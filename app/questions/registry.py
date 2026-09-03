import hashlib
import re

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "geopolitics": [
        "war",
        "military",
        "conflict",
        "treaty",
        "sanction",
        "nato",
        "un ",
        "invasion",
        "territory",
    ],
    "economics": [
        "gdp",
        "inflation",
        "recession",
        "fed ",
        "interest rate",
        "trade",
        "tariff",
        "market",
        "stock",
    ],
    "technology": ["ai", "llm", "chip", "semiconductor", "quantum", "software", "cyber", "hack"],
    "health": ["pandemic", "vaccine", "disease", "who ", "outbreak", "virus", "covid"],
    "climate": ["climate", "carbon", "emissions", "renewable", "temperature", "paris agreement"],
    "politics": [
        "election",
        "vote",
        "president",
        "prime minister",
        "parliament",
        "congress",
        "referendum",
    ],
}


def normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for dedup purposes."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def question_id(text: str) -> str:
    """SHA-256 of the normalized question text."""
    normalized = normalize(text)
    return hashlib.sha256(normalized.encode()).hexdigest()


def tag_domains(text: str) -> list[str]:
    """Return a list of domain tags based on keyword matching."""
    lower = text.lower()
    tags: list[str] = []
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            tags.append(domain)
    return tags if tags else ["general"]
