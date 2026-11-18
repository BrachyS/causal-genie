"""A run's Markdown report as a PDF, for the app's download button.

The font is DejaVu, which matplotlib (a dependency of dowhy) ships: it covers the accents,
arrows, dashes and quotes reports use. It has no colour emoji, so characters it can't draw
(the verdict's 🔀, the Genie's 🧞) are dropped rather than printed as empty boxes.
"""

from functools import cache
from pathlib import Path

import matplotlib
from fontTools.ttLib import TTFont
from fpdf import FPDF, FontFace, TextStyle
from markdown_it import MarkdownIt

FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
STYLES = {
    "": "DejaVuSans.ttf",
    "B": "DejaVuSans-Bold.ttf",
    "I": "DejaVuSans-Oblique.ttf",
    "BI": "DejaVuSans-BoldOblique.ttf",
}
MONO = "DejaVuSansMono.ttf"
INK = "#1f2937"  # headings in dark grey, not fpdf2's default maroon
TAG_STYLES = {
    "code": FontFace(family="dejavumono"),
    "pre": TextStyle(font_family="dejavumono", t_margin=2, b_margin=2),
    "h1": TextStyle(color=INK, font_size_pt=18, b_margin=2, t_margin=4),
    "h2": TextStyle(color=INK, font_size_pt=14, b_margin=1, t_margin=5),
    "h3": TextStyle(color=INK, font_size_pt=12, b_margin=1, t_margin=4),
}


@cache
def _glyphs() -> frozenset[int]:
    cmap = TTFont(FONTS / STYLES[""]).getBestCmap()
    assert cmap is not None, "DejaVu Sans has a Unicode cmap"
    return frozenset(cmap)


def printable(text: str) -> str:
    """`text` without the characters the PDF font has no glyph for."""
    glyphs = _glyphs()
    return "".join(c for c in text if c in "\n\t" or ord(c) in glyphs)


def report_pdf(markdown: str) -> bytes:
    """The report as an A4 PDF. Raw HTML in the Markdown is shown as text, never rendered."""
    html = MarkdownIt("commonmark", {"html": False}).enable("table").render(printable(markdown))
    pdf = FPDF(format="A4")
    for style, name in STYLES.items():
        pdf.add_font("dejavu", style, FONTS / name)
    pdf.add_font("dejavumono", "", FONTS / MONO)
    pdf.set_font("dejavu", size=10)
    pdf.add_page()
    pdf.write_html(html, font_family="dejavu", tag_styles=TAG_STYLES, li_prefix_color=INK)
    return bytes(pdf.output())
