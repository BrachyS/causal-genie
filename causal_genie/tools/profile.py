"""Read a downloaded table and summarize it, so models see column names and summaries only.

Rows never leave this module for a model. The MVP handles one table per dataset: a ZIP with
several tables or a workbook with several sheets is rejected, not joined (joins are stretch).
"""

import io
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd

from causal_genie.agent.state import ColumnProfile, TableProfile
from causal_genie.log import log_step

TABLE_FORMATS = {".csv", ".tsv", ".xlsx", ".parquet"}


class NotSingleTable(ValueError):
    """The dataset holds several tables; the MVP can't use it (reported, then next candidate)."""


def read_table(path: Path) -> pd.DataFrame:
    """The one table in a CSV, TSV, XLSX, Parquet or ZIP file (by extension)."""
    suffix = path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            tables = [n for n in z.namelist() if PurePosixPath(n).suffix.lower() in TABLE_FORMATS]
            if len(tables) != 1:
                raise NotSingleTable(f"{path.name} holds {len(tables)} tables: {sorted(tables)}")
            return read_bytes(z.read(tables[0]), PurePosixPath(tables[0]).suffix.lower())
    return read_bytes(path.read_bytes(), suffix)


def read_bytes(body: bytes, suffix: str) -> pd.DataFrame:
    """Parse one table's bytes according to its file extension."""
    if suffix == ".csv":
        return pd.read_csv(io.BytesIO(body))
    if suffix == ".tsv":
        return pd.read_csv(io.BytesIO(body), sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(io.BytesIO(body))
    if suffix == ".xlsx":
        sheets = pd.read_excel(io.BytesIO(body), sheet_name=None)
        if len(sheets) != 1:
            raise NotSingleTable(f"workbook has {len(sheets)} sheets: {list(sheets)}")
        return next(iter(sheets.values()))
    raise ValueError(f"unsupported table format {suffix!r}")


def profile(path: Path) -> TableProfile:
    """Column names, types, missing and distinct counts, and numeric ranges; no values of
    text columns, since categories can identify people."""
    with log_step("profile", path=str(path)) as rec:
        table = read_table(path)
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
