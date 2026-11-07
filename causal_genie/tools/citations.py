"""Quote grounding: is a quote the model gave really in the text it read?

A model can paraphrase or invent a quote, so the harness checks each one against the fetched
text. Small differences (case, spacing, curly quotes, `...` for omitted words) are allowed;
changed numbers are not.
"""

import re

from rapidfuzz import fuzz

QUOTE_MATCH = 90  # rapidfuzz partial_ratio for a quote segment against the source text
SHORT = 20  # segments shorter than this must appear exactly (a fuzzy match is too loose)
ELLIPSIS = re.compile(r"\.\.\.|…")
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def normalize(text: str) -> str:
    """Lowercase with straight quotes, plain hyphens and single spaces (also before `%`)."""
    text = text.lower().translate(str.maketrans("“”‘’–—", "\"\"''--"))
    return re.sub(r"\s+%", "%", " ".join(text.split()))


def quote_found(quote: str, text: str) -> bool:
    """Whether every part of `quote` (split at ellipses) appears in `text`, with each of its
    numbers present in `text` too."""
    source = normalize(text)
    segments = [s.strip() for s in ELLIPSIS.split(normalize(quote)) if s.strip()]
    if not segments:
        return False
    numbers = set(NUMBER.findall(source))
    for segment in segments:
        if not set(NUMBER.findall(segment)) <= numbers:
            return False
        if len(segment) < SHORT:
            if segment not in source:
                return False
        elif fuzz.partial_ratio(segment, source) < QUOTE_MATCH:
            return False
    return True
