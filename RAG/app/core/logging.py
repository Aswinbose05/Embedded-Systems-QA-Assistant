import re
import logging
import sys
from typing import Any

# Patterns to sanitize API keys from logs
SENSITIVE_PATTERNS = [
    re.compile(r"gsk_[a-zA-Z0-9_-]{20,}"),
    re.compile(r"sk-[a-zA-Z0-9_-]{20,}"),
    re.compile(r"lsv2_[a-zA-Z0-9_-]{20,}"),
    re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{20,}", re.IGNORECASE),
]


class SensitiveDataFilter(logging.Filter):
    """
    Log filter that redacts API keys and sensitive tokens from log messages.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self.redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: self.redact(v) if isinstance(v, str) else v for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(self.redact(v) if isinstance(v, str) else v for v in record.args)
        return True

    @staticmethod
    def redact(text: str) -> str:
        for pattern in SENSITIVE_PATTERNS:
            text = pattern.sub("[REDACTED_API_KEY]", text)
        return text


def setup_logging(level: int = logging.INFO) -> None:
    """
    Sets up structured logging with safe token redaction.
    """
    logger = logging.getLogger("app")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        handler.addFilter(SensitiveDataFilter())
        logger.addHandler(handler)

    # Apply filter to root logger as well
    for h in logging.root.handlers:
        h.addFilter(SensitiveDataFilter())


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance with sensitive filter applied.
    """
    logger = logging.getLogger(f"app.{name}")
    logger.addFilter(SensitiveDataFilter())
    return logger
