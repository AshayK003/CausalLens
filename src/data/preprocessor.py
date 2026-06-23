from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PreprocessReport:
    original_rows: int
    original_cols: int
    final_rows: int = 0
    final_cols: int = 0
    steps_applied: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def detect_date_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    col_names_lower = {c.lower() for c in df.columns}
    has_year = bool(col_names_lower & {"year", "yr", "fiscal_year"})
    has_month = bool(col_names_lower & {"month", "mo", "mon"})
    for col in df.columns:
        try:
            series = df[col].astype(str) if not pd.api.types.is_string_dtype(df[col]) else df[col]
            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().sum() > len(df) * 0.5:
                # If both year and month columns exist, skip "year" — let
                # try_construct_date_from_year_month handle it instead.
                if has_year and has_month and col.lower() in {"year", "yr", "fiscal_year"}:
                    continue
                return col
        except Exception:
            continue
    return None


def detect_numeric_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    exclude = exclude or []
    numeric_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
            continue
        try:
            cleaned = df[col].astype(str).str.replace(r"[^\d.\-]", "", regex=True)
            converted = pd.to_numeric(cleaned, errors="coerce")
            if converted.notna().sum() > len(df) * 0.5:
                numeric_cols.append(col)
        except Exception:
            continue
    return numeric_cols


def clean_column_names(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    steps = []
    new_cols = []
    for col in df.columns:
        new_name = col.strip().lower().replace(" ", "_").replace("-", "_")
        new_name = "".join(c for c in new_name if c.isalnum() or c == "_")
        new_name = new_name.strip("_")
        if not new_name:
            new_name = f"col_{len(new_cols)}"
        if new_name != col:
            steps.append(f"Renamed '{col}' -> '{new_name}'")
        new_cols.append(new_name)

    if steps:
        df = df.copy()
        df.columns = new_cols
    return df, steps


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    steps = []
    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)
    if n_removed > 0:
        steps.append(f"Removed {n_removed} duplicate rows")
    return df, steps


def drop_empty_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    steps = []
    n_before = len(df)
    df = df.dropna(how="all")
    n_removed = n_before - len(df)
    if n_removed > 0:
        steps.append(f"Removed {n_removed} completely empty rows")
    return df, steps


def drop_empty_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    steps = []
    empty_cols = [col for col in df.columns if df[col].isna().all()]
    if empty_cols:
        steps.append(f"Dropped empty columns: {', '.join(empty_cols)}")
        df = df.drop(columns=empty_cols)
    return df, steps


def parse_dates(df: pd.DataFrame, date_col: str) -> tuple[pd.DataFrame, list[str]]:
    steps = []
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        return df, steps

    df = df.copy()

    if df[date_col].dtype in [np.int64, np.float64]:
        min_val = df[date_col].min()
        max_val = df[date_col].max()
        if min_val < -1e10 or max_val > 1e18:
            steps.append(
                f"Skipped date parsing for '{date_col}': "
                f"numeric values out of datetime range"
            )
            return df, steps

    try:
        df[date_col] = pd.to_datetime(
            df[date_col].astype(str), errors="coerce", format="mixed"
        )
    except Exception:
        try:
            df[date_col] = pd.to_datetime(df[date_col].astype(str), errors="coerce")
        except Exception as e:
            steps.append(f"Date parsing failed: {e}")
            return df, steps

    n_failed = df[date_col].isna().sum()
    n_total = len(df)
    success_rate = (n_total - n_failed) / n_total if n_total > 0 else 0

    if n_failed > 0:
        if n_failed == n_total:
            steps.append(f"Date parsing failed: all {n_total} values invalid")
        elif success_rate < 0.5:
            steps.append(
                f"Date parsing: only {int(success_rate * 100)}% success rate, skipped"
            )
        else:
            steps.append(f"Parsed dates: dropped {n_failed}/{n_total} invalid rows")
            df = df.dropna(subset=[date_col])
    else:
        steps.append(f"Parsed '{date_col}' as datetime")

    return df, steps


def try_construct_date_from_year_month(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None, list[str]]:
    steps = []
    year_col = None
    month_col = None

    for col in df.columns:
        if col in ("year", "yr", "fiscal_year"):
            if pd.api.types.is_numeric_dtype(df[col]):
                year_col = col
            else:
                try:
                    converted = pd.to_numeric(df[col], errors="coerce")
                    if converted.notna().sum() > len(df) * 0.5:
                        year_col = col
                except Exception:
                    pass

    for col in df.columns:
        if col in ("month", "mo", "mon") and pd.api.types.is_numeric_dtype(df[col]):
            values = df[col].dropna()
            if values.min() >= 1 and values.max() <= 12:
                month_col = col

    if year_col and month_col:
        df = df.copy()
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
        df[month_col] = pd.to_numeric(df[month_col], errors="coerce")
        df = df.dropna(subset=[year_col, month_col])
        df["__constructed_date"] = pd.to_datetime(
            df[year_col].astype(int).astype(str) + "-" +
            df[month_col].astype(int).astype(str).str.zfill(2) + "-01",
            errors="coerce",
        )
        df = df.dropna(subset=["__constructed_date"])
        steps.append(f"Constructed date from '{year_col}' + '{month_col}' columns")
        return df, "__constructed_date", steps

    if year_col:
        df = df.copy()
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
        df = df.dropna(subset=[year_col])
        df["__constructed_date"] = pd.to_datetime(
            df[year_col].astype(int).astype(str) + "-01-01",
            errors="coerce",
        )
        df = df.dropna(subset=["__constructed_date"])
        steps.append(f"Constructed date from '{year_col}' column (year only)")
        return df, "__constructed_date", steps

    return df, None, steps


def parse_numeric(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, list[str]]:
    steps = []
    if pd.api.types.is_numeric_dtype(df[col]):
        return df, steps

    try:
        df = df.copy()
        original = df[col].copy()
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(r"[^\d.\-]", "", regex=True),
            errors="coerce",
        )
        n_failed = df[col].isna().sum()
        n_original = original.notna().sum()
        if n_failed > 0 and n_original > 0:
            steps.append(
                f"Converted '{col}' to numeric: {n_failed}/{n_original} values became NaN"
            )
    except Exception as e:
        steps.append(f"Numeric conversion failed for '{col}': {e}")

    return df, steps


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "drop",
) -> tuple[pd.DataFrame, list[str]]:
    steps = []
    if strategy == "drop":
        n_before = len(df)
        df = df.dropna()
        n_removed = n_before - len(df)
        if n_removed > 0:
            steps.append(f"Dropped {n_removed} rows with missing values")
    elif strategy == "forward_fill":
        n_filled = df.isna().sum().sum()
        df = df.ffill()
        df = df.bfill()
        steps.append(f"Forward-filled {int(n_filled)} missing values")
    elif strategy == "mean":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
                steps.append(f"Filled {n_missing} missing values in '{col}' with mean ({mean_val:.2f})")
    return df, steps


def remove_outliers(
    df: pd.DataFrame,
    col: str,
    method: str = "iqr",
    threshold: float = 3.0,
) -> tuple[pd.DataFrame, list[str]]:
    steps = []
    if not pd.api.types.is_numeric_dtype(df[col]):
        return df, steps

    n_before = len(df)

    if method == "iqr":
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    elif method == "zscore":
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df = df[np.abs((df[col] - mean) / std) <= threshold]

    n_removed = n_before - len(df)
    if n_removed > 0:
        steps.append(f"Removed {n_removed} outliers from '{col}' ({method} method)")
    return df, steps


def preprocess_data(
    df: pd.DataFrame,
    date_col: str | None = None,
    metric_col: str | None = None,
    missing_strategy: str = "drop",
    remove_outliers_flag: bool = False,
    outlier_method: str = "iqr",
) -> tuple[pd.DataFrame, PreprocessReport]:
    report = PreprocessReport(
        original_rows=len(df),
        original_cols=len(df.columns),
    )

    df, steps = clean_column_names(df)
    report.steps_applied.extend(steps)

    df, steps = drop_empty_columns(df)
    report.steps_applied.extend(steps)

    df, steps = drop_empty_rows(df)
    report.steps_applied.extend(steps)

    df, steps = drop_duplicates(df)
    report.steps_applied.extend(steps)

    if date_col is None:
        date_col = detect_date_column(df)
        if date_col:
            report.warnings.append(f"Auto-detected date column: '{date_col}'")

    date_constructed = False
    if date_col and date_col in df.columns:
        df, steps = parse_dates(df, date_col)
        report.steps_applied.extend(steps)

        n_dt = df[date_col].notna().sum() if date_col in df.columns else 0
        if n_dt < len(df) * 0.5:
            report.warnings.append(
                f"Date column '{date_col}' has too few valid dates ({n_dt}/{len(df)}). "
                "Attempting to construct date from year/month columns."
            )
            df_restored = df.drop(columns=[date_col])
            df_constructed, new_date_col, construct_steps = try_construct_date_from_year_month(df_restored)
            report.steps_applied.extend(construct_steps)
            if new_date_col:
                date_col = new_date_col
                df = df_constructed
                date_constructed = True
                report.warnings.append(f"Using constructed date column: '{date_col}'")

    if date_col is None and not date_constructed:
        df_constructed, new_date_col, construct_steps = try_construct_date_from_year_month(df)
        report.steps_applied.extend(construct_steps)
        if new_date_col:
            date_col = new_date_col
            df = df_constructed
            date_constructed = True
            report.warnings.append("Constructed date from year/month columns")

    if metric_col is None:
        exclude = [date_col] if date_col else []
        numeric_cols = detect_numeric_columns(df, exclude=exclude)
        if numeric_cols:
            metric_col = numeric_cols[0]
            report.warnings.append(f"Auto-detected metric column: '{metric_col}'")

    if metric_col and metric_col in df.columns:
        df, steps = parse_numeric(df, metric_col)
        report.steps_applied.extend(steps)

    if date_col and date_col in df.columns:
        df = df.sort_values(date_col).reset_index(drop=True)

    df, steps = handle_missing_values(df, strategy=missing_strategy)
    report.steps_applied.extend(steps)

    if remove_outliers_flag and metric_col and metric_col in df.columns:
        df, steps = remove_outliers(df, metric_col, method=outlier_method)
        report.steps_applied.extend(steps)

    report.final_rows = len(df)
    report.final_cols = len(df.columns)

    if len(df) < 30:
        report.warnings.append(
            f"Only {len(df)} rows remaining after preprocessing. "
            "Results may be unreliable with less than 30 data points."
        )

    logger.info(
        f"Preprocessing complete: {report.original_rows} -> {report.final_rows} rows, "
        f"{len(report.steps_applied)} steps applied"
    )

    return df, report
