"""Dedupe by study (PRD §5.3): ten news stories about one paper count as one study.

Sources match by DOI or PMID, then by a close title (or main title, before the colon) with
compatible first author and year. A news story rarely gives the paper's title, so a secondary
source also matches a study with the same first author and year.
"""

import logging

from causal_genie.agent.state import Study
from causal_genie.tools.metadata import titles_match

log = logging.getLogger(__name__)


def dedupe(studies: list[Study]) -> list[Study]:
    """One study per real study, in first-seen order; merged sources go to `also_at`.

    The best source represents the study: the paper itself over a news story, then one with a
    DOI/PMID, then one read in full.
    """
    merged: list[Study] = []
    for study in studies:
        for i, kept in enumerate(merged):
            if same_study(kept, study):
                merged[i] = merge(kept, study)
                log.info("dedupe: %s (%s) is the same study as %s", study.id, study.url, kept.id)
                break
        else:
            merged.append(study)
    return merged


def same_study(a: Study, b: Study) -> bool:
    """Whether two sources report the same study."""
    if a.doi and b.doi:
        return a.doi.lower() == b.doi.lower()
    if a.pmid and b.pmid:
        return a.pmid == b.pmid
    year_ok = a.year is None or b.year is None or a.year == b.year
    author_ok = a.first_author is None or b.first_author is None or same_author(a, b)
    if year_ok and author_ok and titles_match(a.title, b.title):
        return True
    # A news story names the authors and year, rarely the paper's title.
    return (a.secondary_only or b.secondary_only) and same_author(a, b) and a.year == b.year


def same_author(a: Study, b: Study) -> bool:
    """Both name the same first author (surname, ignoring case)."""
    x, y = a.first_author, b.first_author
    return x is not None and y is not None and x.lower() == y.lower()


def merge(kept: Study, new: Study) -> Study:
    """One study from two sources of it; the better source represents it."""
    best, other = (new, kept) if rank(new) > rank(kept) else (kept, new)
    return best.model_copy(
        update={
            "also_at": [*best.also_at, other.url, *other.also_at],
            "secondary_only": best.secondary_only and other.secondary_only,
        }
    )


def rank(study: Study) -> tuple[bool, bool, bool]:
    """How well a source represents its study: the paper, identified, read in full."""
    return (not study.secondary_only, bool(study.doi or study.pmid), study.full_text)
