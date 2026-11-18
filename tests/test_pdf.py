"""The report as a PDF download: Markdown rendered with a Unicode font, emoji dropped."""

from pathlib import Path

import pytest

from causal_genie.agent.state import RunState
from causal_genie.ui.pdf import dated, printable, report_pdf

GALLERY = sorted((Path(__file__).parents[1] / "gallery").glob("*/state.json"))


def test_printable_keeps_what_the_font_has_and_drops_the_rest() -> None:
    assert printable("**Verdict:** 🔀 Common cause → 25 °C — ‘ok’") == (
        "**Verdict:**  Common cause → 25 °C — ‘ok’"
    )


def _is_pdf(data: bytes) -> bool:
    return data.startswith(b"%PDF-") and data.rstrip().endswith(b"%%EOF")


def test_a_report_with_headings_lists_tables_and_quotes_becomes_a_pdf() -> None:
    md = (
        "# Does X cause Y?\n\n**Verdict:** ✅ Estimated from data\n\n## Findings\n\n"
        "- one\n- two with `code`\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n> “a quote”\n\n"
        "[a link](https://example.org)\n"
    )
    assert _is_pdf(report_pdf(md))


@pytest.mark.parametrize("path", GALLERY, ids=lambda p: p.parent.name)
def test_every_gallery_report_becomes_a_pdf(path: Path) -> None:
    state = RunState.load(path)
    assert state.report
    assert _is_pdf(report_pdf(state.report))


def test_the_answer_date_goes_under_the_title() -> None:
    md = "# Does X cause Y?\n\n**Verdict:** ✅ Estimated from data\n"

    assert dated(md, "Nov 17, 2026") == (
        "# Does X cause Y?\n\n*Answered Nov 17, 2026*\n\n**Verdict:** ✅ Estimated from data\n"
    )
    assert dated(md, "") == md
    assert dated("No title\n", "Nov 17, 2026") == "*Answered Nov 17, 2026*\n\nNo title\n"
