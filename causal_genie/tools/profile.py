"""Read a downloaded table and summarize it, so models see column names and summaries only.

Rows never leave this module for a model. The MVP analyses one table per answer: a workbook
with several sheets or a ZIP with several tables is split into parts (`table_parts`), each
a candidate of its own and read by name; parts are never joined (joins are stretch).
"""

import io
import re
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd

from causal_genie.agent.state import ColumnProfile, TableProfile
from causal_genie.log import log_step

TABLE_FORMATS = {".csv", ".tsv", ".xlsx", ".parquet"}


class NotSingleTable(ValueError):
    """The dataset holds several tables; the MVP can't use it (reported, then next candidate)."""


MIN_NAMED_COLUMNS = 2  # a sheet with fewer named columns is notes or metadata, not a table
# Sheet or member names that say they document the data rather than hold it.
DOCUMENTATION = re.compile(r"meta[ _-]?data|readme|\bnotes?\b|codebook|dictionary", re.IGNORECASE)


def read_table(path: Path, part: str | None = None) -> pd.DataFrame:
    """A table in a CSV, TSV, XLSX, Parquet or ZIP file (by extension): the sheet or ZIP
    member named `part`, or the file's only table when `part` is None."""
    suffix = path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            tables = zip_tables(z)
            if part is None and len(tables) != 1:
                raise NotSingleTable(f"{path.name} holds {len(tables)} tables: {sorted(tables)}")
            name = part or tables[0]
            if name not in tables:
                raise ValueError(f"{path.name} has no table {name!r}")
            return read_bytes(z.read(name), PurePosixPath(name).suffix.lower())
    return read_bytes(path.read_bytes(), suffix, part)


def zip_tables(z: zipfile.ZipFile) -> list[str]:
    """Members of a ZIP that are tables, by extension. The `__MACOSX/` folder and `._` files
    that zipping on a Mac adds are file attributes, not tables, so they are left out."""
    return [
        n
        for n in z.namelist()
        if PurePosixPath(n).suffix.lower() in TABLE_FORMATS
        and not n.startswith("__MACOSX/")
        and not PurePosixPath(n).name.startswith("._")
    ]


def read_bytes(body: bytes, suffix: str, sheet: str | None = None) -> pd.DataFrame:
    """Parse one table's bytes according to its file extension; for a workbook, the sheet
    named `sheet`, or its only sheet."""
    if suffix == ".csv":
        return pd.read_csv(io.BytesIO(body))
    if suffix == ".tsv":
        return pd.read_csv(io.BytesIO(body), sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(io.BytesIO(body))
    if suffix == ".xlsx":
        sheets = pd.read_excel(io.BytesIO(body), sheet_name=None)
        if sheet is not None:
            if sheet not in sheets:
                raise ValueError(f"workbook has no sheet {sheet!r}: {list(sheets)}")
            return sheets[sheet]
        if len(sheets) != 1:
            raise NotSingleTable(f"workbook has {len(sheets)} sheets: {list(sheets)}")
        return next(iter(sheets.values()))
    raise ValueError(f"unsupported table format {suffix!r}")


def is_table(columns: list[str]) -> bool:
    """Whether a header looks like a data table: at least `MIN_NAMED_COLUMNS` named columns
    (pandas names blank header cells `Unnamed: n`)."""
    return sum(not c.startswith("Unnamed:") for c in columns) >= MIN_NAMED_COLUMNS


def table_parts(path: Path) -> dict[str | None, list[str]]:
    """Each table in the file with its column names: the sheets of a workbook or the table
    members of a ZIP, by name; a file holding one table gives {None: columns}. Parts that are
    not tables (under 2 named columns) or are named as documentation (metadata, README,
    notes, codebook) are left out."""
    with log_step("table_parts", path=str(path)) as rec:
        suffix = path.suffix.lower()
        if suffix == ".zip":
            with zipfile.ZipFile(path) as z:
                names = zip_tables(z)
                found = {n: header(z.read(n), PurePosixPath(n).suffix.lower()) for n in names}
        elif suffix == ".xlsx":
            sheets = pd.read_excel(path, sheet_name=None, nrows=0)
            found = {name: [str(c) for c in frame.columns] for name, frame in sheets.items()}
        else:
            found = {None: [str(c) for c in read_table(path).columns]}
        if len(found) == 1:
            found = {None: next(iter(found.values()))}
        parts = {
            name: cols
            for name, cols in found.items()
            if is_table(cols) and not (name and DOCUMENTATION.search(name))
        }
        rec.update(parts=list(parts), left_out=[n for n in found if n not in parts])
        return parts


def header(body: bytes, suffix: str) -> list[str]:
    """The column names of one table's bytes."""
    return [str(c) for c in read_bytes(body, suffix).columns]


def profile(path: Path, part: str | None = None) -> TableProfile:
    """Column names, types, missing and distinct counts, and numeric ranges of the table
    (`part` as in `read_table`); no values of text columns, since categories can identify
    people."""
    with log_step("profile", path=str(path), part=part) as rec:
        table = read_table(path, part)
        columns = []
        for name in table.columns:
            col = table[name]
            numeric = pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col)
            columns.append(
                ColumnProfile(
                    name=str(name),
                    dtype=str(col.dtype),
                    missing=int(col.isna().sum()),
                    unique=int(col.nunique()),
                    min=float(col.min()) if numeric and col.notna().any() else None,
                    max=float(col.max()) if numeric and col.notna().any() else None,
                    mean=float(col.mean()) if numeric and col.notna().any() else None,
                )
            )
        result = TableProfile(rows=len(table), columns=columns)
        rec.update(rows=result.rows, columns=[c.name for c in columns])
        return result
