from collections.abc import Callable

import pytest

from causal_genie.agent.state import Study
from causal_genie.tools.dedupe import dedupe

PAPER_TITLE = (
    "A randomized, double-blind comparison of licorice versus sugar-water gargle for "
    "prevention of postoperative sore throat and postextubation coughing"
)


@pytest.fixture
def make_study(sample_state) -> Callable[..., Study]:
    """A study like the sample trial, with the given fields changed."""
    base = sample_state.studies[0]

    def make(**changes) -> Study:
        return base.model_copy(update=changes)

    return make


@pytest.fixture
def news(make_study) -> list[Study]:
    """Ten news stories about the D1 trial: no DOI, headline titles, same author and year."""
    return [
        make_study(
            id=f"n{i}",
            url=f"https://news{i}.example.com/licorice",
            doi=None,
            pmid=None,
            title=f"Licorice gargle cuts sore throat after surgery, story {i}",
            secondary_only=True,
            full_text=False,
        )
        for i in range(10)
    ]


def test_ten_news_stories_and_one_paper_are_one_study(make_study, news) -> None:
    paper = make_study(id="p", title=PAPER_TITLE)

    (study,) = dedupe([*news[:5], paper, *news[5:]])

    assert study.id == "p"  # the paper itself represents the study
    assert not study.secondary_only
    assert sorted(study.also_at) == sorted(s.url for s in news)


def test_news_stories_alone_stay_one_secondary_only_study(news) -> None:
    (study,) = dedupe(news)

    assert study.secondary_only
    assert len(study.also_at) == 9


def test_same_doi_is_one_study_even_with_different_titles(make_study) -> None:
    a = make_study(id="a", url="https://pubmed.ncbi.nlm.nih.gov/23921656")
    b = make_study(id="b", url="https://doi.org/x", title="Licorice gargle trial", pmid=None)

    (study,) = dedupe([a, b])

    assert study.also_at == ["https://doi.org/x"]


def test_same_title_with_different_case_and_punctuation_is_one_study(make_study) -> None:
    a = make_study(id="a", doi=None, pmid=None, title=PAPER_TITLE)
    b = make_study(id="b", doi=None, pmid=None, url="https://b", title=PAPER_TITLE.upper() + ".")

    assert len(dedupe([a, b])) == 1


def test_same_author_other_year_is_another_study(make_study) -> None:
    a = make_study(id="a", doi=None, pmid=None, secondary_only=True)
    b = make_study(id="b", doi=None, pmid=None, url="https://b", year=2016, title="Follow-up")

    assert [s.id for s in dedupe([a, b])] == ["a", "b"]


def test_different_dois_are_different_studies(make_study) -> None:
    a = make_study(id="a")
    b = make_study(id="b", doi="10.1000/other", pmid=None, url="https://b")

    assert len(dedupe([a, b])) == 2
