from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from samsung_health_parser import (
    BODY_METRICS,
    SLEEP_STAGE_ORDER,
    available_body_metrics,
    combine_frames,
    load_health_csv,
    metric_label,
    prepare_body_records,
    prepare_sleep_records,
    prepare_sleep_stages,
    summarize_sleep_sessions,
    time_weighted_mean,
)


APP_DIR = Path(__file__).parent
SAMPLE_DIR = APP_DIR / "sample_dataset"

STAGE_COLORS = {
    "Awake": "#f59e0b",
    "REM": "#8b5cf6",
    "Light": "#38bdf8",
    "Deep": "#1d4ed8",
    "Unknown": "#94a3b8",
}
CHART_TEXT_COLOR = "#111827"
CHART_MUTED_TEXT_COLOR = "#475569"
CHART_GRID_COLOR = "#e5e7eb"
CHART_AXIS_COLOR = "#cbd5e1"
TIME_GRAINS = {
    "Day": "D",
    "Week": "W-SUN",
    "Month": "M",
}


st.set_page_config(
    page_title="Samsung health record analyzer",
    page_icon=":material/monitor_heart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --surface: #ffffff;
        --line: #d8dee8;
        --muted: #64748b;
        --ink: #111827;
    }
    .main .block-container {
        padding-top: 1.4rem;
        max-width: 1380px;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
        min-height: 94px;
    }
    [data-testid="stMetricLabel"] p {
        color: var(--muted);
        font-size: 0.82rem;
    }
    [data-testid="stMetricValue"] {
        color: var(--ink);
    }
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stDataFrame"]) {
        border-radius: 8px;
    }
    .small-note {
        color: var(--muted);
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_sample_records(sample_paths: tuple[str, ...]):
    return [load_health_csv(Path(path)) for path in sample_paths]


@st.cache_data(show_spinner=False)
def load_uploaded_record(name: str, content: bytes, record_type: str | None):
    try:
        return load_health_csv(content, source_name=name, record_type=record_type)
    except TypeError as exc:
        if "record_type" not in str(exc):
            raise
        record = load_health_csv(content, source_name=name)
        return replace(record, record_type=record_type) if record_type else record


def format_minutes(minutes: float | int | pd.NA) -> str:
    if pd.isna(minutes):
        return ""
    minutes = float(minutes)
    hours = int(minutes // 60)
    mins = int(round(minutes % 60))
    if mins == 60:
        hours += 1
        mins = 0
    return f"{hours}h {mins:02d}m"


def latest_non_null(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = frame[column].dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def apply_chart_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        font=dict(color=CHART_TEXT_COLOR),
        legend=dict(
            font=dict(color=CHART_TEXT_COLOR),
            title_font=dict(color=CHART_TEXT_COLOR),
        ),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor=CHART_AXIS_COLOR,
            font=dict(color=CHART_TEXT_COLOR),
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    axis_style = dict(
        color=CHART_TEXT_COLOR,
        title_font=dict(color=CHART_TEXT_COLOR),
        tickfont=dict(color=CHART_MUTED_TEXT_COLOR),
        gridcolor=CHART_GRID_COLOR,
        linecolor=CHART_AXIS_COLOR,
        zerolinecolor=CHART_AXIS_COLOR,
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    fig.update_traces(
        textfont=dict(color=CHART_TEXT_COLOR),
        outsidetextfont=dict(color=CHART_TEXT_COLOR),
        selector=dict(type="pie"),
    )
    return fig


def has_usable_time(frame: pd.DataFrame, column: str = "record_dt") -> bool:
    return column in frame.columns and frame[column].notna().any()


def time_bucket(series: pd.Series, grain: str) -> pd.Series:
    if grain == "Day":
        return series.dt.floor("D")
    return series.dt.to_period(TIME_GRAINS[grain]).dt.start_time


def time_axis_title(grain: str) -> str:
    if grain == "Week":
        return "Week"
    if grain == "Month":
        return "Month"
    return "Date"


def render_record_summary(records) -> None:
    summary = pd.DataFrame(
        [
            {
                "File": record.source_name,
                "Type": record.record_type,
                "Rows": len(record.frame),
                "Columns": len(record.frame.columns),
                "Samsung type": record.metadata[0] if record.metadata else "",
            }
            for record in records
        ]
    )
    st.dataframe(summary, width="stretch", hide_index=True)


def render_sleep_timeline(sleep: pd.DataFrame) -> None:
    fig = go.Figure()
    use_absolute_time = sleep["has_absolute_time"].all() and sleep["start_dt"].notna().any()
    ordered_labels = sorted(
        sleep["stage_label"].dropna().unique(),
        key=lambda label: SLEEP_STAGE_ORDER.get(label, 0),
        reverse=True,
    )
    for label in ordered_labels:
        stage_rows = sleep[sleep["stage_label"] == label]
        bar_x = (
            stage_rows["duration_min"] * 60 * 1000
            if use_absolute_time
            else stage_rows["duration_min"]
        )
        bar_base = stage_rows["start_dt"] if use_absolute_time else stage_rows["x_start_min"]
        fig.add_trace(
            go.Bar(
                x=bar_x,
                y=stage_rows["session_label"],
                base=bar_base,
                orientation="h",
                name=label,
                marker_color=STAGE_COLORS.get(label, "#94a3b8"),
                customdata=stage_rows[["hover_start", "hover_end", "duration_min"]],
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Start: %{customdata[0]}<br>"
                    "End: %{customdata[1]}<br>"
                    "Duration: %{customdata[2]:.1f} min"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        barmode="stack",
        height=max(340, min(720, 72 + sleep["session_label"].nunique() * 42)),
        margin=dict(l=12, r=12, t=10, b=42),
        xaxis_title="Time" if use_absolute_time else "Elapsed minutes within session",
        yaxis_title=None,
        legend_title=None,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    if use_absolute_time:
        fig.update_xaxes(type="date")
    fig.update_yaxes(autorange="reversed")
    apply_chart_theme(fig)
    st.plotly_chart(fig, width="stretch")


def render_sleep_records(sleep_records: pd.DataFrame) -> None:
    session_count = len(sleep_records)
    total_recorded = sleep_records["duration_min"].sum()
    average_session = total_recorded / session_count if session_count else 0
    longest_session = sleep_records["duration_min"].max() if session_count else 0
    has_start_time = has_usable_time(sleep_records, "start_dt")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Sleep sessions", f"{session_count:,}")
    metric_columns[1].metric("Recorded sleep", format_minutes(total_recorded))
    metric_columns[2].metric("Average session", format_minutes(average_session))
    metric_columns[3].metric("Longest session", format_minutes(longest_session))

    if has_start_time:
        time_grain = st.radio(
            "Sleep x-axis",
            ["Session", *TIME_GRAINS.keys()],
            horizontal=True,
            index=1,
            key="sleep_records_time_grain",
        )
    else:
        time_grain = "Session"

    if has_start_time and time_grain != "Session":
        chart_data = sleep_records.dropna(subset=["start_dt"]).copy()
        chart_data["time_bucket"] = time_bucket(chart_data["start_dt"], time_grain)
        chart_data = (
            chart_data.groupby("time_bucket", as_index=False)["duration_min"]
            .sum()
            .sort_values("time_bucket", kind="stable")
        )
        chart_x = chart_data["time_bucket"]
        chart_y = chart_data["duration_min"]
        x_title = time_axis_title(time_grain)
        hovertemplate = f"{time_axis_title(time_grain)}: %{{x|%Y-%m-%d}}<br>Total sleep: %{{y:.1f}} min<extra></extra>"
    elif has_start_time:
        chart_data = sleep_records.dropna(subset=["start_dt"]).sort_values(["start_dt", "session_number"])
        chart_x = chart_data["start_dt"]
        chart_y = chart_data["duration_min"]
        x_title = "Session start"
        hovertemplate = "Start: %{x|%Y-%m-%d %H:%M}<br>Duration: %{y:.1f} min<extra></extra>"
    else:
        chart_data = sleep_records
        chart_x = chart_data["session_label"]
        chart_y = chart_data["duration_min"]
        x_title = "Session"
        hovertemplate = "Session: %{x}<br>Duration: %{y:.1f} min<extra></extra>"

    fig = go.Figure(
        data=[
            go.Bar(
                x=chart_x,
                y=chart_y,
                marker_color="#2563eb",
                hovertemplate=hovertemplate,
            )
        ]
    )
    fig.update_layout(
        height=330,
        margin=dict(l=12, r=12, t=10, b=42),
        xaxis_title=x_title,
        yaxis_title="Minutes",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    if has_start_time:
        fig.update_xaxes(type="date")
    apply_chart_theme(fig)
    st.plotly_chart(fig, width="stretch")

    display = pd.DataFrame(
        {
            "Session": sleep_records["session_label"],
            "Start": sleep_records["start_dt"].dt.strftime("%Y-%m-%d %H:%M").fillna(sleep_records["hover_start"]),
            "End": sleep_records["end_dt"].dt.strftime("%Y-%m-%d %H:%M").fillna(sleep_records["hover_end"]),
            "Duration": sleep_records["duration_min"].map(format_minutes),
        }
    )
    for source_column, label in [
        ("sleep_score", "Sleep score"),
        ("score", "Score"),
        ("efficiency", "Efficiency"),
    ]:
        if source_column in sleep_records.columns and sleep_records[source_column].notna().any():
            display[label] = sleep_records[source_column]

    st.subheader("Sleep Sessions")
    st.dataframe(display, width="stretch", hide_index=True)


def render_sleep_tab(sleep: pd.DataFrame, sleep_records: pd.DataFrame, loaded_sleep_records) -> None:
    if sleep.empty and sleep_records.empty:
        if loaded_sleep_records:
            st.info("Sleep files were loaded, but no usable sleep-stage or sleep-session rows were found.")
            render_record_summary(loaded_sleep_records)
        else:
            st.info("No sleep records were loaded.")
        return

    if not sleep.empty:
        session_summary = summarize_sleep_sessions(sleep)
        total_recorded = sleep["duration_min"].sum()
        asleep = total_recorded - sleep.loc[sleep["stage_label"] == "Awake", "duration_min"].sum()
        deep_pct = sleep.loc[sleep["stage_label"] == "Deep", "duration_min"].sum() / total_recorded * 100
        rem_pct = sleep.loc[sleep["stage_label"] == "REM", "duration_min"].sum() / total_recorded * 100

        metric_columns = st.columns(4)
        metric_columns[0].metric("Sleep sessions", f"{sleep['sleep_id'].nunique():,}")
        metric_columns[1].metric("Recorded sleep", format_minutes(total_recorded))
        metric_columns[2].metric("Asleep time", format_minutes(asleep))
        metric_columns[3].metric("Deep + REM", f"{deep_pct + rem_pct:.1f}%")

        sessions = session_summary["Session"].tolist()
        selected = st.multiselect("Sessions", sessions, default=sessions[: min(8, len(sessions))])
        filtered = sleep[sleep["session_label"].isin(selected)] if selected else sleep.iloc[0:0]

        render_sleep_timeline(filtered)

        left, right = st.columns([1.1, 1])
        with left:
            st.subheader("Session Summary")
            display = session_summary.copy()
            for column in ["Total recorded", "Asleep", "Awake", "Light", "Deep", "REM"]:
                display[column] = display[column].map(format_minutes)
            display["Sleep efficiency"] = display["Sleep efficiency"].map(
                lambda value: "" if pd.isna(value) else f"{value:.1f}%"
            )
            st.dataframe(display.drop(columns=["Sleep ID"]), width="stretch", hide_index=True)

        with right:
            stage_totals = (
                sleep.groupby("stage_label", as_index=False)["duration_min"]
                .sum()
                .sort_values("duration_min", ascending=False)
            )
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=stage_totals["stage_label"],
                        values=stage_totals["duration_min"],
                        hole=0.58,
                        marker_colors=[STAGE_COLORS.get(label, "#94a3b8") for label in stage_totals["stage_label"]],
                        textinfo="label+percent",
                        textposition="outside",
                        hovertemplate="%{label}<br>%{value:.1f} min<extra></extra>",
                    )
                ]
            )
            fig.update_layout(height=360, margin=dict(l=44, r=44, t=16, b=16), legend_title=None)
            apply_chart_theme(fig)
            st.plotly_chart(fig, width="stretch")

    if not sleep_records.empty:
        if not sleep.empty:
            st.divider()
        render_sleep_records(sleep_records)


def render_body_tab(body: pd.DataFrame) -> None:
    if body.empty:
        st.info("No weight or body-composition records were loaded.")
        return

    metrics = available_body_metrics(body)
    default_metrics = [metric for metric in ["weight", "body_fat", "total_body_water", "bmi"] if metric in metrics]
    selected_metrics = st.multiselect(
        "Body Metrics",
        metrics,
        default=default_metrics or metrics[: min(3, len(metrics))],
        format_func=metric_label,
    )

    latest_weight = latest_non_null(body, "weight")
    latest_bmi = latest_non_null(body, "bmi")
    latest_body_fat = latest_non_null(body, "body_fat")
    latest_water = latest_non_null(body, "total_body_water")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Latest weight", f"{latest_weight:.1f} kg" if latest_weight else "-")
    metric_columns[1].metric("Latest BMI", f"{latest_bmi:.1f}" if latest_bmi else "-")
    metric_columns[2].metric("Latest body fat", f"{latest_body_fat:.1f}%" if latest_body_fat else "-")
    metric_columns[3].metric("Latest body water", f"{latest_water:.1f} kg" if latest_water else "-")

    if "weight" in body.columns and body["weight"].notna().any():
        weighted = body[body["weight"].notna()]
        weight_has_time = has_usable_time(weighted, "record_dt")
        lifetime_weight = time_weighted_mean(weighted, "weight", "record_dt")
        simple_weight = float(weighted["weight"].mean())
        span_days = None
        if weight_has_time:
            span = weighted["record_dt"].max() - weighted["record_dt"].min()
            span_days = span.total_seconds() / 86400

        st.markdown("##### Lifetime body weight")
        lifetime_columns = st.columns(3)
        if weight_has_time and lifetime_weight is not None:
            lifetime_columns[0].metric("Time-weighted average", f"{lifetime_weight:.1f} kg")
        else:
            lifetime_columns[0].metric("Average weight", f"{simple_weight:.1f} kg")
        lifetime_columns[1].metric("Simple average", f"{simple_weight:.1f} kg")
        if span_days is not None:
            lifetime_columns[2].metric("Measured span", f"{span_days:,.0f} days")

        if weight_has_time:
            st.caption(
                "Time-weighted average = integral of weight over time div by total elapsed time "
                "(trapezoidal integration across measurement timestamps), so periods you held a "
                "weight longer count proportionally rather than each record counting equally."
            )
        else:
            st.caption("No usable timestamps, so only the simple average of records is available.")

    if selected_metrics:
        has_record_time = has_usable_time(body, "record_dt")
        if has_record_time:
            time_grain = st.radio(
                "Body x-axis",
                list(TIME_GRAINS.keys()),
                horizontal=True,
                key="body_time_grain",
            )
            chart_data = body.dropna(subset=["record_dt"]).copy()
            chart_data["time_bucket"] = time_bucket(chart_data["record_dt"], time_grain)
            chart_data = (
                chart_data.groupby("time_bucket", as_index=False)[selected_metrics]
                .mean(numeric_only=True)
                .sort_values("time_bucket", kind="stable")
            )
            x_values = chart_data["time_bucket"]
            x_title = time_axis_title(time_grain)
        else:
            chart_data = body
            x_values = chart_data["record_index"]
            x_title = "Record number"

        fig = go.Figure()
        for metric in selected_metrics:
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=chart_data[metric],
                    mode="lines+markers",
                    name=metric_label(metric),
                    line=dict(width=2),
                    marker=dict(size=7),
                    connectgaps=False,
                    hovertemplate=f"{metric_label(metric)}: %{{y:.2f}}<extra></extra>",
                )
            )
        fig.update_layout(
            height=430,
            margin=dict(l=12, r=12, t=10, b=42),
            xaxis_title=x_title,
            yaxis_title="Value",
            legend_title=None,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
        )
        if has_record_time:
            fig.update_xaxes(type="date")
        apply_chart_theme(fig)
        st.plotly_chart(fig, width="stretch")

    table_columns = [
        column
        for column in ["record_index", "record_dt", *BODY_METRICS.keys(), "bmi"]
        if column in body.columns
    ]
    display = body[table_columns].copy()
    if "record_dt" in display.columns:
        display["record_dt"] = display["record_dt"].astype("string").replace("<NA>", "")
    st.subheader("Body Records")
    st.dataframe(display, width="stretch", hide_index=True)


def render_explorer(records) -> None:
    if not records:
        return
    render_record_summary(records)
    source_names = [record.source_name for record in records]
    selected_source = st.selectbox("Source File", source_names)
    record = next(item for item in records if item.source_name == selected_source)
    st.dataframe(record.frame, width="stretch", hide_index=True)
    st.download_button(
        "Download Normalized CSV",
        record.frame.to_csv(index=False).encode("utf-8"),
        file_name=f"normalized_{Path(record.source_name).stem}.csv",
        mime="text/csv",
    )


sample_paths = tuple(str(path) for path in sorted(SAMPLE_DIR.glob("*.csv")))

st.sidebar.header("Data")
sleep_uploads = st.sidebar.file_uploader(
    "Sleep / Sleep Stage CSV Files",
    type=["csv"],
    accept_multiple_files=True,
    key="sleep_stage_uploads",
)
body_uploads = st.sidebar.file_uploader(
    "Weight / Body CSV Files",
    type=["csv"],
    accept_multiple_files=True,
    key="body_uploads",
)
other_uploads = st.sidebar.file_uploader(
    "Other CSV Files",
    type=["csv"],
    accept_multiple_files=True,
    key="other_uploads",
)
sleep_uploads = sleep_uploads or []
body_uploads = body_uploads or []
other_uploads = other_uploads or []
has_uploads = bool(sleep_uploads or body_uploads or other_uploads)
include_samples = st.sidebar.toggle("Include Sample Dataset", value=not has_uploads)

records = []
errors = []
if include_samples and sample_paths:
    records.extend(load_sample_records(sample_paths))
for uploaded, record_type in [
    *[(file, None) for file in sleep_uploads],
    *[(file, None) for file in body_uploads],
    *[(file, None) for file in other_uploads],
]:
    try:
        records.append(load_uploaded_record(uploaded.name, uploaded.getvalue(), record_type))
    except Exception as exc:  # pragma: no cover - visible app error path
        errors.append((uploaded.name, str(exc)))

st.title("Samsung health record analyzer")

for file_name, message in errors:
    st.error(f"{file_name}: {message}")

if not records:
    st.warning("No Samsung Health CSV records loaded.")
    st.stop()

sleep_stage_frame = combine_frames(records, "sleep_stage")
sleep_record_frame = combine_frames(records, "sleep")
body_frame = combine_frames(records, "weight")
sleep = prepare_sleep_stages(sleep_stage_frame)
sleep_records = prepare_sleep_records(sleep_record_frame)
body = prepare_body_records(body_frame)
loaded_sleep_records = [record for record in records if record.record_type in {"sleep", "sleep_stage"}]

overview, sleep_tab, body_tab, explorer_tab = st.tabs(["Overview", "Sleep", "Body", "Data Explorer"])

with overview:
    loaded_rows = sum(len(record.frame) for record in records)
    known_types = pd.Series([record.record_type for record in records]).value_counts()
    latest_weight = latest_non_null(body, "weight")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Files loaded", f"{len(records):,}")
    metric_columns[1].metric("Parsed rows", f"{loaded_rows:,}")
    metric_columns[2].metric("Record types", f"{len(known_types):,}")
    metric_columns[3].metric("Latest weight", f"{latest_weight:.1f} kg" if latest_weight else "-")

    left, right = st.columns([0.9, 1.1])
    with left:
        st.subheader("Loaded Records")
        render_record_summary(records)
    with right:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=known_types.index,
                    y=known_types.values,
                    marker_color=["#2563eb", "#16a34a", "#f59e0b", "#64748b"][: len(known_types)],
                    hovertemplate="%{x}: %{y} file(s)<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            height=330,
            margin=dict(l=10, r=10, t=10, b=40),
            xaxis_title="Record type",
            yaxis_title="Files",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
        )
        apply_chart_theme(fig)
        st.plotly_chart(fig, width="stretch")

with sleep_tab:
    render_sleep_tab(sleep, sleep_records, loaded_sleep_records)

with body_tab:
    render_body_tab(body)

with explorer_tab:
    render_explorer(records)
