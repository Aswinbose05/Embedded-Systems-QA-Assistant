import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from groq import (
    Groq,
    GroqError,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    NotFoundError,
    BadRequestError,
)
from app.core.config import settings
from app.core.logging import get_logger
from app.core.exceptions import LLMGatewayError

logger = get_logger("gateway")

# Attempt importing LangSmith traceable if available
try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@dataclass
class LLMResponse:
    content: str
    model_used: str
    is_fallback: bool = False
    latency_ms: float = 0.0
    usage: Dict[str, int] = field(default_factory=dict)


class LLMGateway:
    """
    Centralized production LLM Gateway managing all interactions with Groq.
    Features:
    - Automatic retries with exponential backoff on transient errors (rate limit, timeout, network).
    - Automatic fallback to secondary model when primary fails or is unavailable.
    - Graceful error containment so internal secrets and raw errors are never leaked.
    - Configurable models via environment variables.
    - Observability and latency tracking.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        primary_model: Optional[str] = None,
        fallback_model: Optional[str] = None,
        max_retries: Optional[int] = None,
        retry_backoff: Optional[float] = None,
        timeout: Optional[float] = None,
        portkey_api_key: Optional[str] = None,
    ):
        self.api_key = api_key or settings.GROQ_API_KEY
        if not self.api_key:
            logger.warning("GROQ_API_KEY is not set. Gateway calls will fail until configured.")

        self.primary_model = primary_model or settings.PRIMARY_MODEL
        self.fallback_model = fallback_model or settings.FALLBACK_MODEL
        self.max_retries = max_retries if max_retries is not None else settings.MAX_RETRIES
        self.retry_backoff = retry_backoff if retry_backoff is not None else settings.RETRY_BACKOFF_FACTOR
        self.timeout = timeout if timeout is not None else settings.REQUEST_TIMEOUT
        self.portkey_api_key = portkey_api_key or getattr(settings, "PORTKEY_API_KEY", None)

        self._client: Optional[Groq] = None
        self._portkey_client = None

        if self.portkey_api_key:
            try:
                from portkey_ai import Portkey
                self._portkey_client = Portkey(
                    api_key=self.portkey_api_key,
                    provider="groq",
                    Authorization=f"Bearer {self.api_key}",
                )
                logger.info("Portkey Gateway client initialized successfully.")
            except Exception as e:
                logger.warning("Could not initialize Portkey client: %s. Using native gateway.", str(e))

    def _get_client(self) -> Groq:
        if self._client is None:
            if not self.api_key:
                raise LLMGatewayError("GROQ_API_KEY is missing. Please configure it in .env.")
            self._client = Groq(api_key=self.api_key, timeout=self.timeout)
        return self._client

    def _is_transient_error(self, exc: Exception) -> bool:
        """Determines if an exception is transient and worthy of a retry."""
        return isinstance(
            exc,
            (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError),
        )

    def _call_model(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> LLMResponse:
        """
        Calls a specific model with exponential backoff retry handling.
        """
        client = self._portkey_client if self._portkey_client is not None else self._get_client()
        last_exception = None

        for attempt in range(1, self.max_retries + 1):
            start_time = time.perf_counter()
            try:
                logger.info(
                    "LLM Gateway invoking model: %s (attempt %d/%d)",
                    model_name,
                    attempt,
                    self.max_retries,
                )

                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                )

                duration_ms = (time.perf_counter() - start_time) * 1000
                content = response.choices[0].message.content or ""

                usage = {}
                if getattr(response, "usage", None):
                    usage = {
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                        "total_tokens": getattr(response.usage, "total_tokens", 0),
                    }

                logger.info(
                    "Model %s succeeded in %.2f ms (tokens: %s)",
                    model_name,
                    duration_ms,
                    usage,
                )

                return LLMResponse(
                    content=content,
                    model_used=model_name,
                    is_fallback=(model_name == self.fallback_model and self.primary_model != self.fallback_model),
                    latency_ms=duration_ms,
                    usage=usage,
                )

            except Exception as exc:
                last_exception = exc
                duration_ms = (time.perf_counter() - start_time) * 1000

                # Non-transient errors like NotFoundError (model invalid) or BadRequestError
                if isinstance(exc, (NotFoundError, BadRequestError)):
                    logger.warning(
                        "Model %s is unavailable or rejected request (%.2f ms): %s",
                        model_name,
                        duration_ms,
                        str(exc),
                    )
                    # Immediate break to allow fallback model attempt
                    break

                if self._is_transient_error(exc):
                    sleep_time = self.retry_backoff ** attempt
                    logger.warning(
                        "Transient error with model %s on attempt %d: %s. Retrying in %.2fs...",
                        model_name,
                        attempt,
                        str(exc),
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error("Non-transient error with model %s: %s", model_name, str(exc))
                    break

        raise last_exception or LLMGatewayError(f"Model {model_name} failed after {self.max_retries} attempts.")

    @traceable(name="llm_gateway_generate", run_type="llm")
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> LLMResponse:
        """
        Executes an LLM chat completion with primary model and automatic fallback.
        """
        try:
            # Step 1: Attempt Primary Model
            return self._call_model(
                model_name=self.primary_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as primary_error:
            logger.warning(
                "Primary model '%s' failed: %s. Attempting fallback to '%s'...",
                self.primary_model,
                str(primary_error),
                self.fallback_model,
            )

            # Step 2: Attempt Fallback Model if distinct
            if self.fallback_model and self.fallback_model != self.primary_model:
                try:
                    fallback_response = self._call_model(
                        model_name=self.fallback_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    fallback_response.is_fallback = True
                    logger.info("Fallback model '%s' succeeded.", self.fallback_model)
                    return fallback_response
                except Exception as fallback_error:
                    logger.error(
                        "Both primary model '%s' and fallback model '%s' failed. Fallback error: %s",
                        self.primary_model,
                        self.fallback_model,
                        str(fallback_error),
                    )
                    raise LLMGatewayError(
                        "All LLM models in the gateway failed. Please try again shortly.",
                        details={
                            "primary_model": self.primary_model,
                            "fallback_model": self.fallback_model,
                        },
                    ) from fallback_error
            else:
                raise LLMGatewayError(
                    f"Primary model '{self.primary_model}' failed and no secondary fallback model is configured."
                ) from primary_error


# Global gateway singleton instance
llm_gateway = LLMGateway()
