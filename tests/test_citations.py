import pytest

from causal_genie.tools.citations import quote_found

SOURCE = """Results: The incidence of sore throat at 30 minutes was 24% in the licorice group
and 61% in the sugar-water group (P < 0.001). Licorice gargling also reduced
“postextubation coughing” in the first 24 hours."""


@pytest.mark.parametrize(
    "quote",
    [
        "The incidence of sore throat at 30 minutes was 24% in the licorice group",
        "the INCIDENCE of sore throat at 30   minutes was 24% in the licorice group",
        'reduced "postextubation coughing"',
        "sore throat at 30 minutes was 24% ... 61% in the sugar-water group",
        "sore throat at 30 minutes was 24% … 61% in the sugar-water group",
        "The incidence of sore throat at 30 minutes was 24 % in the licorice group",
    ],
)
def test_quote_in_source_is_found(quote: str) -> None:
    assert quote_found(quote, SOURCE)


@pytest.mark.parametrize(
    "quote",
    [
        "Licorice gargling eliminated sore throat in all patients",  # made up
        "The incidence of sore throat at 30 minutes was 34% in the licorice group",  # number
        "24% vs 51%",  # short, not exact
        "",
        "...",
    ],
)
def test_quote_not_in_source_is_not_found(quote: str) -> None:
    assert not quote_found(quote, SOURCE)
