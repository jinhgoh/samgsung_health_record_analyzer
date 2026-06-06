from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np
import pandas as pd


SLEEP_STAGE_MAP = {
    40001: "Awake",
    40002: "Light",
    40003: "Deep",
    40004: "REM",
}

SLEEP_STAGE_ORDER = {
    "Awake": 4,
    "REM": 3,
    "Light": 2,
    "Deep": 1,
    "Unknown": 0,
}

BODY_METRICS = {
    "weight": {"label": "Weight", "unit": "kg"},
    "height": {"label": "Height", "unit": "cm"},
    "body_fat": {"label": "Body fat", "unit": "%"},
    "body_fat_mass": {"label": "Body fat mass", "unit": "kg"},
    "muscle_mass": {"label": "Muscle mass", "unit": "kg"},
    "skeletal_muscle": {"label": "Skeletal muscle", "unit": "%"},
    "skeletal_muscle_mass": {"label": "Skeletal muscle mass", "unit": "kg"},
    "fat_free": {"label": "Fat free", "unit": "%"},
    "fat_free_mass": {"label": "Fat free mass", "unit": "kg"},
    "total_body_water": {"label": "Total body water", "unit": "kg"},
    "basal_metabolic_rate": {"label": "Basal metabolic rate", "unit": "kcal"},
    "vfa_level": {"label": "VFA level", "unit": ""},
}


@dataclass(frozen=True)
class HealthRecord:
    source_name: str
    record_type: str
    metadata: tuple[str, ...]
    frame: pd.DataFrame


def load_health_csv(
    source: str | Path | bytes | BinaryIO,
    source_name: str | None = None,
    record_type: str | None = None,
) -> HealthRecord:
    raw = _read_bytes(source)
    name = source_name or getattr(source, "name", None) or "uploaded.csv"
    text = raw.decode("utf-8-sig", errors="replace")
    metadata = _read_first_csv_row(text)
    skip_rows = 1 if metadata and metadata[0].startswith("com.samsung.health.") else 0
    inferred_record_type = record_type or _infer_record_type(name, metadata)

    frame = _read_csv_frame(text, skip_rows)
    frame = frame.replace(r"^\s*$", pd.NA, regex=True).dropna(how="all")
    frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed")]
    frame = frame.reset_index(drop=True)
    if record_type is None and inferred_record_type == "unknown":
        inferred_record_type = _infer_record_type(name, metadata, frame.columns)
    frame.attrs["source_name"] = name
    frame.attrs["record_type"] = inferred_record_type

    return HealthRecord(
        source_name=name,
        record_type=inferred_record_type,
        metadata=tuple(metadata),
        frame=frame,
    )


def combine_frames(records: Iterable[HealthRecord], record_type: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for record in records:
        if record.record_type != record_type:
            continue
        frame = record.frame.copy()
        frame["source_name"] = record.source_name
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def prepare_sleep_stages(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "stage" not in frame.columns:
        return pd.DataFrame()

    work = frame.copy()
    work["_row_order"] = range(len(work))
    work["stage_code"] = pd.to_numeric(work["stage"], errors="coerce").astype("Int64")
    work["stage_label"] = work["stage_code"].map(SLEEP_STAGE_MAP).fillna("Unknown")
    work["stage_order"] = work["stage_label"].map(SLEEP_STAGE_ORDER).fillna(0)

    if "sleep_id" not in work.columns:
        work["sleep_id"] = work.get("source_name", pd.Series("Sleep session", index=work.index))
    work["sleep_id"] = work["sleep_id"].fillna("Sleep session").astype(str)

    work["start_dt"] = parse_datetime_series(work.get("start_time"))
    work["end_dt"] = parse_datetime_series(work.get("end_time"))
    work["has_absolute_time"] = work["start_dt"].notna() & work["end_dt"].notna()

    work["start_fragment_min"] = work.get("start_time", pd.Series(pd.NA, index=work.index)).map(
        _clock_fragment_minutes
    )
    work["end_fragment_min"] = work.get("end_time", pd.Series(pd.NA, index=work.index)).map(
        _clock_fragment_minutes
    )
    work["clock_period_min"] = work.get("start_time", pd.Series(pd.NA, index=work.index)).map(
        _clock_period_minutes
    )

    absolute_duration = (work["end_dt"] - work["start_dt"]).dt.total_seconds() / 60
    absolute_duration = absolute_duration.mask(absolute_duration < 0, absolute_duration + 1440)

    fragment_duration = work["end_fragment_min"] - work["start_fragment_min"]
    fragment_duration = fragment_duration.mask(
        fragment_duration < 0,
        fragment_duration + work["clock_period_min"].fillna(1440),
    )

    work["duration_min"] = absolute_duration.where(work["has_absolute_time"], fragment_duration)
    work = work[work["duration_min"].notna() & (work["duration_min"] > 0)].copy()
    if work.empty:
        return work

    sort_columns = ["sleep_id", "has_absolute_time", "_row_order"]
    if work["has_absolute_time"].any():
        sort_columns = ["sleep_id", "start_dt", "_row_order"]
    work = work.sort_values(sort_columns, kind="stable")

    work["session_number"] = pd.factorize(work["sleep_id"])[0] + 1
    work["session_label"] = work["session_number"].map(lambda value: f"Session {value}")

    work["x_start_min"] = 0.0
    for sleep_id, group in work.groupby("sleep_id", sort=False):
        if group["has_absolute_time"].all():
            first_start = group["start_dt"].min()
            starts = (group["start_dt"] - first_start).dt.total_seconds() / 60
            work.loc[group.index, "x_start_min"] = starts
        else:
            work.loc[group.index, "x_start_min"] = group["duration_min"].cumsum() - group["duration_min"]

    work["x_end_min"] = work["x_start_min"] + work["duration_min"]
    work["hover_start"] = _format_time_value(work.get("start_time"))
    work["hover_end"] = _format_time_value(work.get("end_time"))
    return work.reset_index(drop=True)


def summarize_sleep_sessions(sleep: pd.DataFrame) -> pd.DataFrame:
    if sleep.empty:
        return pd.DataFrame()

    rows = []
    for (_, group) in sleep.groupby("sleep_id", sort=False):
        stage_minutes = group.groupby("stage_label")["duration_min"].sum().to_dict()
        total_minutes = group["duration_min"].sum()
        awake_minutes = stage_minutes.get("Awake", 0.0)
        asleep_minutes = max(total_minutes - awake_minutes, 0.0)
        rows.append(
            {
                "Session": group["session_label"].iloc[0],
                "Sleep ID": group["sleep_id"].iloc[0],
                "Total recorded": total_minutes,
                "Asleep": asleep_minutes,
                "Awake": awake_minutes,
                "Light": stage_minutes.get("Light", 0.0),
                "Deep": stage_minutes.get("Deep", 0.0),
                "REM": stage_minutes.get("REM", 0.0),
                "Sleep efficiency": (asleep_minutes / total_minutes * 100) if total_minutes else pd.NA,
                "Stage rows": len(group),
            }
        )
    return pd.DataFrame(rows)


def prepare_sleep_records(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "start_time" not in frame.columns or "end_time" not in frame.columns:
        return pd.DataFrame()

    work = frame.copy()
    work["_row_order"] = range(len(work))

    id_column = next((column for column in ["sleep_id", "datauuid"] if column in work.columns), None)
    if id_column:
        work["sleep_id"] = work[id_column].fillna("").astype(str).str.strip()
    else:
        work["sleep_id"] = ""
    fallback_id = (
        work.get("source_name", pd.Series("Sleep session", index=work.index))
        .fillna("Sleep session")
        .astype(str)
        + " #"
        + (work["_row_order"] + 1).astype(str)
    )
    missing_id = work["sleep_id"].eq("") | work["sleep_id"].str.lower().eq("<na>")
    work["sleep_id"] = work["sleep_id"].mask(missing_id, fallback_id)

    work["start_dt"] = parse_datetime_series(work["start_time"])
    work["end_dt"] = parse_datetime_series(work["end_time"])
    work["has_absolute_time"] = work["start_dt"].notna() & work["end_dt"].notna()

    work["start_fragment_min"] = work["start_time"].map(_clock_fragment_minutes)
    work["end_fragment_min"] = work["end_time"].map(_clock_fragment_minutes)
    work["clock_period_min"] = work["start_time"].map(_clock_period_minutes)

    absolute_duration = (work["end_dt"] - work["start_dt"]).dt.total_seconds() / 60
    absolute_duration = absolute_duration.mask(absolute_duration < 0, absolute_duration + 1440)

    fragment_duration = work["end_fragment_min"] - work["start_fragment_min"]
    fragment_duration = fragment_duration.mask(
        fragment_duration < 0,
        fragment_duration + work["clock_period_min"].fillna(1440),
    )

    work["duration_min"] = absolute_duration.where(work["has_absolute_time"], fragment_duration)
    work = work[work["duration_min"].notna() & (work["duration_min"] > 0)].copy()
    if work.empty:
        return work

    sort_columns = ["start_dt", "_row_order"] if work["start_dt"].notna().any() else ["_row_order"]
    work = work.sort_values(sort_columns, kind="stable")
    work["session_number"] = range(1, len(work) + 1)
    work["session_label"] = work["session_number"].map(lambda value: f"Session {value}")
    work["hover_start"] = _format_time_value(work.get("start_time"))
    work["hover_end"] = _format_time_value(work.get("end_time"))
    return work.reset_index(drop=True)


def prepare_body_records(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    metric_columns = [column for column in BODY_METRICS if column in frame.columns]
    if not metric_columns:
        return pd.DataFrame()

    work = frame.copy()
    for column in metric_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce").astype("float64")

    if "weight" in metric_columns and work["weight"].notna().any():
        work = work[work["weight"].notna()].copy()
    else:
        work = work.dropna(subset=metric_columns, how="all").copy()
    if work.empty:
        return work

    time_column = "start_time" if "start_time" in work.columns else "create_time" if "create_time" in work.columns else None
    work["record_dt"] = parse_datetime_series(work[time_column]) if time_column else pd.NaT
    work["_row_order"] = range(len(work))
    if work["record_dt"].notna().any():
        work = work.sort_values(["record_dt", "_row_order"], kind="stable")
    work["record_index"] = range(1, len(work) + 1)

    if "weight" in work.columns and "height" in work.columns:
        height_m = work["height"].where(work["height"] < 3, work["height"] / 100)
        work["bmi"] = work["weight"] / (height_m**2)

    return work.reset_index(drop=True)


def metric_label(column: str) -> str:
    if column == "bmi":
        return "BMI"
    config = BODY_METRICS.get(column)
    if not config:
        return column.replace("_", " ").title()
    unit = config["unit"]
    return f"{config['label']} ({unit})" if unit else config["label"]


def parse_datetime_series(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="datetime64[ns]")

    values = series.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    numeric = pd.to_numeric(values, errors="coerce")
    numeric_mask = numeric.notna() & values.str.fullmatch(r"-?\d+(\.\d+)?").fillna(False)
    if numeric_mask.any():
        median = numeric[numeric_mask].abs().median()
        if median > 1e11:
            parsed.loc[numeric_mask] = pd.to_datetime(numeric[numeric_mask], unit="ms", errors="coerce")
        elif median > 1e9:
            parsed.loc[numeric_mask] = pd.to_datetime(numeric[numeric_mask], unit="s", errors="coerce")

    text_mask = values.notna() & values.map(_looks_date_like)
    if text_mask.any():
        parsed.loc[text_mask] = pd.to_datetime(values[text_mask], errors="coerce")

    return parsed


def available_body_metrics(frame: pd.DataFrame) -> list[str]:
    metrics = [column for column in BODY_METRICS if column in frame.columns and frame[column].notna().any()]
    if "bmi" in frame.columns and frame["bmi"].notna().any():
        metrics.append("bmi")
    return metrics


def time_weighted_mean(
    frame: pd.DataFrame,
    value_column: str = "weight",
    time_column: str = "record_dt",
) -> float | None:
    """Lifetime average of a metric weighted by how long each value was held.

    Integrates the value over time with the trapezoidal rule and divides by the
    total elapsed span: ``(integral of value dt) / (t_last - t_first)``. Unlike a
    plain ``mean()`` this accounts for uneven spacing between measurements, so a
    weight kept for months counts more than one logged for a single day. Falls
    back to a simple mean when usable timestamps are missing or span zero time.
    """
    if frame.empty or value_column not in frame.columns:
        return None

    values = pd.to_numeric(frame[value_column], errors="coerce")
    has_time = time_column in frame.columns and frame[time_column].notna().any()
    if not has_time:
        clean = values.dropna()
        return float(clean.mean()) if not clean.empty else None

    work = pd.DataFrame({"value": values, "time": pd.to_datetime(frame[time_column], errors="coerce")})
    work = work.dropna().sort_values("time", kind="stable")
    if work.empty:
        return None
    if len(work) == 1:
        return float(work["value"].iloc[0])

    value_array = work["value"].to_numpy(dtype="float64")
    seconds = (work["time"] - work["time"].iloc[0]).dt.total_seconds().to_numpy()
    total = seconds[-1] - seconds[0]
    if total <= 0:
        return float(value_array.mean())

    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(value_array, seconds) / total)


def _read_bytes(source: str | Path | bytes | BinaryIO) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    if hasattr(source, "getvalue"):
        value = source.getvalue()
        return value if isinstance(value, bytes) else bytes(value)
    return source.read()


def _read_first_csv_row(text: str) -> list[str]:
    try:
        return next(csv.reader(io.StringIO(text)))
    except StopIteration:
        return []


def _read_csv_frame(text: str, skip_rows: int) -> pd.DataFrame:
    rows = list(csv.reader(io.StringIO(text)))
    rows = rows[skip_rows:]
    if not rows:
        return pd.DataFrame()

    header = [str(column).strip() for column in rows[0]]
    data_rows = [_align_csv_row(header, row) for row in rows[1:]]
    return pd.DataFrame(data_rows, columns=header, dtype="string")


def _align_csv_row(header: list[str], row: list[str]) -> list[str]:
    column_count = len(header)
    if len(row) < column_count:
        return row + [""] * (column_count - len(row))
    if len(row) == column_count:
        return row
    if all(str(value).strip() == "" for value in row[column_count:]):
        return row[:column_count]

    extra_count = len(row) - column_count
    normalized_header = [column.strip().lower() for column in header]
    merge_candidates = [
        normalized_header.index(column)
        for column in ["comment", "custom"]
        if column in normalized_header
    ]
    merge_candidates.append(column_count - 1)

    best_row = None
    best_score = None
    for candidate in merge_candidates:
        if candidate + extra_count >= len(row):
            continue
        aligned = (
            row[:candidate]
            + [",".join(row[candidate : candidate + extra_count + 1])]
            + row[candidate + extra_count + 1 :]
        )
        score = _score_aligned_row(header, aligned)
        if best_score is None or score > best_score:
            best_row = aligned
            best_score = score

    return best_row if best_row is not None else row[: column_count - 1] + [",".join(row[column_count - 1 :])]


def _score_aligned_row(header: list[str], row: list[str]) -> int:
    score = 0
    for column, value in zip(header, row):
        value = str(value).strip()
        if not value:
            continue
        match = _value_matches_column(column, value)
        if match is True:
            score += 3
        elif match is False:
            score -= 4
    return score


def _value_matches_column(column: str, value: str) -> bool | None:
    column = column.strip().lower()
    if column in {"comment", "custom", "client_data_id", "deviceuuid"}:
        return None
    if column in BODY_METRICS or column in {
        "stage",
        "create_sh_ver",
        "modify_sh_ver",
        "client_data_ver",
    }:
        return bool(re.fullmatch(r"-?\d+(\.\d+)?", value))
    if column in {"start_time", "end_time", "create_time", "update_time"} or column.endswith("_time"):
        return (
            _looks_date_like(value)
            or _clock_fragment_minutes(value) is not None
            or bool(re.fullmatch(r"-?\d{10,13}(\.\d+)?", value))
        )
    if column == "time_offset":
        return value.upper().startswith(("UTC", "GMT"))
    if column in {"datauuid", "sleep_id"}:
        return bool(re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", value))
    if column == "pkg_name":
        return "." in value
    return None


def _infer_record_type(name: str, metadata: list[str], columns: Iterable[str] | None = None) -> str:
    candidates = [metadata[0] if metadata else "", name]
    joined = " ".join(candidates).lower()
    if "sleep_stage" in joined:
        return "sleep_stage"
    if "weight" in joined:
        return "weight"
    match = re.search(r"com\.samsung\.health\.([a-z0-9_]+)", joined)
    if match:
        return match.group(1)
    normalized_columns = {str(column).strip().lower() for column in ([] if columns is None else columns)}
    if {"stage", "start_time", "end_time"}.issubset(normalized_columns):
        return "sleep_stage"
    sleep_markers = {"sleep_score", "score", "sleep_time", "sleep_duration", "efficiency"}
    if {"start_time", "end_time"}.issubset(normalized_columns) and normalized_columns & sleep_markers:
        return "sleep"
    if "weight" in normalized_columns:
        return "weight"
    return "unknown"


def _looks_date_like(value: object) -> bool:
    text = str(value).strip()
    if not text:
        return False
    return bool(
        re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", text)
        or re.search(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", text)
        or "T" in text
    )


def _clock_fragment_minutes(value: object) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or _looks_date_like(text):
        return None
    parts = text.split(":")
    try:
        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes + seconds / 60
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 60 + minutes + seconds / 60
    except ValueError:
        return None
    return None


def _clock_period_minutes(value: object) -> float | None:
    if pd.isna(value):
        return None
    parts = str(value).strip().split(":")
    if len(parts) == 2:
        return 60.0
    if len(parts) == 3:
        return 1440.0
    return None


def _format_time_value(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="string")
    return series.astype("string").fillna("")
