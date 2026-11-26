import zipfile
from pathlib import Path

import pandas as pd
import pytest

from causal_genie.tools.profile import NotSingleTable, profile, read_table, table_parts

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


# --- One table per sheet or ZIP member (T14.8): each part is its own candidate ---


def workbook(path: Path) -> Path:
    """A workbook like Dryad's: two data sheets and a one-column metadata sheet."""
    with pd.ExcelWriter(path) as w:
        pd.DataFrame({"plot": [1, 2], "herbivory": [0.1, 0.4]}).to_excel(
            w, sheet_name="1. Field data", index=False
        )
        pd.DataFrame({"Notes": ["column plot is..."]}).to_excel(
            w, sheet_name="1. Metadata", index=False
        )
        pd.DataFrame({"plot": [1], "fruits": [3], "deer": [0]}).to_excel(
            w, sheet_name="2. Fruit data", index=False
        )
    return path


def test_parts_of_a_workbook_are_its_data_sheets_with_their_headers(tmp_path: Path) -> None:
    path = workbook(tmp_path / "w.xlsx")

    assert table_parts(path) == {
        "1. Field data": ["plot", "herbivory"],
        "2. Fruit data": ["plot", "fruits", "deer"],
    }  # a sheet with fewer than 2 named columns (notes, metadata) is not a table


def test_one_sheet_of_a_workbook_is_read_and_profiled(tmp_path: Path) -> None:
    path = workbook(tmp_path / "w.xlsx")

    assert list(read_table(path, "2. Fruit data").columns) == ["plot", "fruits", "deer"]
    assert profile(path, "2. Fruit data").rows == 1
    with pytest.raises(NotSingleTable):
        read_table(path)  # still one table per file unless a part is named


def test_parts_of_a_zip_are_its_table_members(tmp_path: Path) -> None:
    path = tmp_path / "two.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("data/a.csv", "x,y\n1,2\n")
        z.writestr("data/b.tsv", "u\tv\tw\n1\t2\t3\n")
        z.writestr("README.txt", "about")

    assert table_parts(path) == {"data/a.csv": ["x", "y"], "data/b.tsv": ["u", "v", "w"]}
    assert list(read_table(path, "data/b.tsv").columns) == ["u", "v", "w"]


def test_parts_of_a_single_table_file_is_that_table(tmp_path: Path) -> None:
    path = tmp_path / "one.xlsx"
    pd.DataFrame({"x": [1], "y": [2]}).to_excel(path, index=False)

    assert table_parts(path) == {None: ["x", "y"]}


def test_a_missing_part_is_an_error(tmp_path: Path) -> None:
    path = workbook(tmp_path / "w.xlsx")

    with pytest.raises(ValueError, match="no sheet"):
        read_table(path, "3. Gone")


@pytest.mark.parametrize("name", ["2. Metadata", "README", "Notes", "Codebook", "data dictionary"])
def test_sheets_named_as_documentation_are_not_tables(tmp_path: Path, name: str) -> None:
    path = tmp_path / "w.xlsx"
    with pd.ExcelWriter(path) as w:
        pd.DataFrame({"a": [1], "b": [2]}).to_excel(w, sheet_name="data", index=False)
        pd.DataFrame({"Patch": ["p"], "Patch number in the field": ["x"]}).to_excel(
            w, sheet_name=name, index=False
        )

    assert list(table_parts(path)) == ["data"]


def test_a_data_sheet_whose_name_merely_starts_like_metadata_is_kept(tmp_path: Path) -> None:
    path = tmp_path / "w.xlsx"
    with pd.ExcelWriter(path) as w:
        pd.DataFrame({"a": [1], "b": [2]}).to_excel(w, sheet_name="Metabolism", index=False)
        pd.DataFrame({"c": [1], "d": [2]}).to_excel(w, sheet_name="Notebook entries", index=False)

    assert list(table_parts(path)) == ["Metabolism", "Notebook entries"]
