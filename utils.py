import re


def clean_text(text: str) -> str:
    """Requirement 1: lowercase, strip punctuation, collapse extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()                          # "HELLO" -> "hello"
    text = re.sub(r"[^a-z0-9\s]", " ", text)      # remove !, ?, #, etc.
    text = re.sub(r"\s+", " ", text).strip()      # collapse multiple spaces
    return text
