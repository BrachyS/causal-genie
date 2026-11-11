import zipfile
from pathlib import Path

import pandas as pd
import pytest

from causal_genie.tools.profile import NotSingleTable, profile, read_table

CSV = "treat,pain,arm\n1,0,zebra\n0,3,okapi\n1,,zebra\n0,5,okapi\n"


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    path = tmp_path / "trial.csv"
    path.write_text(CSV)
    return path


def test_profile_gives_column_summaries_not_rows(csv_path: Path) -> None:
    p = profile(csv_path)

    assert p.rows == 4
    by_name = {c.name: c for c in p.columns}
    assert list(by_name) == ["treat", "pain", "arm"]
    pain = by_name["pain"]
    assert (pain.missing, pain.unique, pain.min, pain.max) == (1, 3, 0, 5)
    assert pain.mean == pytest.approx(8 / 3)
    arm = by_name["arm"]
    assert (arm.unique, arm.min, arm.max, arm.mean) == (2, None, None, None)
    assert "zebra" not in p.model_dump_json()  # no category values leak


def test_tsv_and_xlsx_are_read(tmp_path: Path) -> None:
    tsv = tmp_path / "t.tsv"
    tsv.write_text("x\ty\n1\t2\n")
    xlsx = tmp_path / "t.xlsx"
    pd.DataFrame({"x": [1, 2]}).to_excel(xlsx, index=False)

    assert list(read_table(tsv).columns) == ["x", "y"]
    assert list(read_table(xlsx).columns) == ["x"]


def test_zip_with_one_table_is_read(tmp_path: Path) -> None:
    path = tmp_path / "one.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("data/trial.csv", CSV)
        z.writestr("README.md", "about the data")

    assert len(read_table(path)) == 4


def test_zip_with_several_tables_is_not_usable_in_mvp(tmp_path: Path) -> None:
    path = tmp_path / "nhanes.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("DEMO.csv", "SEQN,age\n1,30\n")
        z.writestr("BPX.csv", "SEQN,sbp\n1,120\n")

    with pytest.raises(NotSingleTable, match=r"2 tables.*BPX.csv"):
        read_table(path)


def test_xlsx_with_several_sheets_is_not_usable_in_mvp(tmp_path: Path) -> None:
    path = tmp_path / "two.xlsx"
    with pd.ExcelWriter(path) as w:
        pd.DataFrame({"x": [1]}).to_excel(w, sheet_name="a", index=False)
        pd.DataFrame({"y": [1]}).to_excel(w, sheet_name="b", index=False)

    with pytest.raises(NotSingleTable, match="2 sheets"):
        read_table(path)
