import re
from enum import Enum
from typing import Dict, Optional, Tuple
from app.core.logging import get_logger

logger = get_logger("router")


class QueryIntent(str, Enum):
    GREETING = "greeting"
    CASUAL = "casual"
    APPRECIATION = "appreciation"
    FAREWELL = "farewell"
    RAG_QUERY = "rag_query"
    UNSAFE = "unsafe"


# Deterministic Natural Responses for Conversational Queries (no RAG/Chroma overhead)
NATURAL_RESPONSES: Dict[QueryIntent, str] = {
    QueryIntent.GREETING: "Hello! 👋 How can I help you with your embedded systems documents today?",
    QueryIntent.CASUAL: "I am your Embedded Systems QA Assistant! 🚀 You can ask me technical questions or upload documents for me to analyze.",
    QueryIntent.APPRECIATION: "You're welcome! 😊 Feel free to ask if you need further explanations or assistance with your documents.",
    QueryIntent.FAREWELL: "Goodbye! 👋 Let me know whenever you have more questions about embedded systems. Have a great day!",
}


class IntentRouter:
    """
    Lightweight, high-speed deterministic router that inspects user queries
    BEFORE RAG retrieval, vector search, or expensive context assembly.

    Routes obvious greetings, casual chit-chat, thanks, and farewells directly,
    while ensuring all technical, embedded systems, and ambiguous queries
    are routed to the RAG pipeline.
    """

    # Exact or near-exact short greetings
    GREETING_PATTERNS = [
        re.compile(r"^(hi|hello|hey|hiya|howdy|hola|namaste)(\s+there|\s+assistant|\s+bot)?[\.!?,]*$", re.IGNORECASE),
        re.compile(r"^(good\s+(morning|afternoon|evening|day))[\.!?,]*$", re.IGNORECASE),
    ]

    # Casual conversation patterns
    CASUAL_PATTERNS = [
        re.compile(r"^(how\s+are\s+you(\s+doing)?|how\'s\s+it\s+going|what\'s\s+up)[\.!?,]*$", re.IGNORECASE),
        re.compile(r"^(who\s+are\s+you|what\s+can\s+you\s+do|tell\s+me\s+about\s+yourself)[\.!?,]*$", re.IGNORECASE),
    ]

    # Appreciation patterns
    APPRECIATION_PATTERNS = [
        re.compile(r"^(thanks|thank\s+you|thx|much\s+appreciated|great\s+job|awesome|perfect|thank\s+you\s+so\s+much)[\.!?,]*$", re.IGNORECASE),
    ]

    # Farewell patterns
    FAREWELL_PATTERNS = [
        re.compile(r"^(bye|goodbye|see\s+you|see\s+ya|have\s+a\s+good\s+day|cya|take\s+care)[\.!?,]*$", re.IGNORECASE),
    ]

    # Specific technical / document keywords that must ALWAYS route to RAG even if greeting words appear
    RAG_OVERRIDE_KEYWORDS = [
        "document", "doc", "pdf", "file", "page", "timer", "uart", "spi", "i2c", "interrupt",
        "gpio", "microcontroller", "register", "firmware", "embedded", "c++", "class", "object",
        "inheritance", "polymorphism", "memory", "dma", "rtos", "what is", "explain", "how does",
        "describe", "define", "summary", "summarize",
    ]

    def route(self, query: str) -> Tuple[QueryIntent, Optional[str]]:
        """
        Determines query intent and returns (QueryIntent, optional_predefined_response).
        If ambiguous or technical, routes to QueryIntent.RAG_QUERY.
        """
        cleaned = query.strip()
        lower = cleaned.lower()

        # Check for RAG override keywords first - ensures technical queries are NEVER mistakenly greeted
        for kw in self.RAG_OVERRIDE_KEYWORDS:
            # Word boundary check
            if re.search(r"\b" + re.escape(kw) + r"\b", lower):
                logger.info("Query routed to RAG due to keyword match '%s': %s", kw, cleaned)
                return QueryIntent.RAG_QUERY, None

        # 1. Greetings
        for pat in self.GREETING_PATTERNS:
            if pat.match(lower):
                logger.info("Routing query to GREETING: %s", cleaned)
                return QueryIntent.GREETING, NATURAL_RESPONSES[QueryIntent.GREETING]

        # 2. Casual
        for pat in self.CASUAL_PATTERNS:
            if pat.match(lower):
                logger.info("Routing query to CASUAL: %s", cleaned)
                return QueryIntent.CASUAL, NATURAL_RESPONSES[QueryIntent.CASUAL]

        # 3. Appreciation
        for pat in self.APPRECIATION_PATTERNS:
            if pat.match(lower):
                logger.info("Routing query to APPRECIATION: %s", cleaned)
                return QueryIntent.APPRECIATION, NATURAL_RESPONSES[QueryIntent.APPRECIATION]

        # 4. Farewell
        for pat in self.FAREWELL_PATTERNS:
            if pat.match(lower):
                logger.info("Routing query to FAREWELL: %s", cleaned)
                return QueryIntent.FAREWELL, NATURAL_RESPONSES[QueryIntent.FAREWELL]

        # Default: Route ambiguous or technical queries to RAG
        logger.info("Routing query to RAG_QUERY: %s", cleaned)
        return QueryIntent.RAG_QUERY, None


intent_router = IntentRouter()
