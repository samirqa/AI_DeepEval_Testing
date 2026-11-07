"""Local retrieval helpers for the Swag Labs FAQ."""

import re

from rag.document import FAQ_CHUNKS

_STOP_WORDS = {
    "a", "all", "are", "at", "can", "does", "for", "how", "is", "it",
    "much", "of", "on", "the", "to", "what", "why",
}


def retrieve_faq_chunks(question: str, n_results: int = 3) -> list[str]:
    """Rank FAQ chunks by keyword overlap without requiring an embedding API."""
    query_terms = set(re.findall(r"[a-z0-9]+", question.lower())) - _STOP_WORDS
    ranked = sorted(
        FAQ_CHUNKS,
        key=lambda chunk: len(
            query_terms & set(re.findall(r"[a-z0-9]+", chunk.lower()))
        ),
        reverse=True,
    )
    return [
        chunk
        for chunk in ranked[:n_results]
        if query_terms & set(re.findall(r"[a-z0-9]+", chunk.lower()))
    ]
