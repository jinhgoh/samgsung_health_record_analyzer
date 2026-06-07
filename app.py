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
    daily_sleep_summary,
    load_health_csv,
    metric_label,
    prepare_body_records,
    prepare_sleep_records,
    prepare_sleep_stages,
    summarize_sleep_sessions,
    time_weighted_mean,
    trailing_rate_per_day,
    weight_band_durations,
    weight_range_by_period,
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


def date_window_slider(dates: pd.Series, key: str, label: str = "Date range"):
    """Render a start-end date slider spanning the data's dates.

    Returns the chosen ``(start, end)`` dates, or ``None`` when there are no
    usable dates so callers skip filtering. With a single date it returns that
    day without rendering a slider.
    """
    valid = pd.to_datetime(dates, errors="coerce").dropna()
    if valid.empty:
        return None
    min_date = valid.min().date()
    max_date = valid.max().date()
    if min_date == max_date:
        return (min_date, max_date)
    return st.slider(
        label,
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD",
        key=key,
        help="Filters the time-series charts in this tab to the selected dates.",
    )


def filter_by_date(frame: pd.DataFrame, column: str, window) -> pd.DataFrame:
    """Keep rows whose ``column`` timestamp falls within the inclusive ``window``.

    ``window`` is a ``(start, end)`` pair of dates. A no-op when ``window`` is
    ``None`` or ``column`` is missing, so unfiltered call sites stay unchanged.
    """
    if window is None or frame.empty or column not in frame.columns:
        return frame
    start, end = window
    times = pd.to_datetime(frame[column], errors="coerce")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
    return frame[times.notna() & (times >= start_ts) & (times < end_ts)]


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


def render_sleep_records(sleep_records: pd.DataFrame, date_window=None) -> None:
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

    chart_source = filter_by_date(sleep_records, "start_dt", date_window)
    if has_start_time and time_grain != "Session":
        chart_data = chart_source.dropna(subset=["start_dt"]).copy()
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
        chart_data = chart_source.dropna(subset=["start_dt"]).sort_values(["start_dt", "session_number"])
        chart_x = chart_data["start_dt"]
        chart_y = chart_data["duration_min"]
        x_title = "Session start"
        hovertemplate = "Start: %{x|%Y-%m-%d %H:%M}<br>Duration: %{y:.1f} min<extra></extra>"
    else:
        chart_data = chart_source
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


def _clock_hours(times: pd.Series) -> pd.Series:
    """Hour-of-day on a noon-anchored 12:00->12:00 scale.

    A night's sleep-in (late evening) and wake-up (next morning) sit on opposite
    sides of midnight, so plotting raw hour-of-day makes the line jump between ~23
    and ~1. Mapping after-midnight times to 24-36 keeps each night continuous.
    """
    hours = times.dt.hour + times.dt.minute / 60
    return hours.where(hours >= 12, hours + 24)


def render_daily_sleep(daily: pd.DataFrame) -> None:
    st.subheader("Daily Sleep")

    dated = daily[daily["sleep_in_dt"].notna()]
    if len(dated) >= 2:
        x = dated["sleep_in_dt"].dt.normalize()
        sleep_in_hours = _clock_hours(dated["sleep_in_dt"])
        wake_up_hours = _clock_hours(dated["wake_up_dt"])

        low_hour = int(min(sleep_in_hours.min(), wake_up_hours.min()))
        high_hour = int(max(sleep_in_hours.max(), wake_up_hours.max())) + 1
        step = 2 if high_hour - low_hour > 12 else 1
        tickvals = list(range(low_hour, high_hour + 1, step))
        ticktext = [f"{value % 24:02d}:00" for value in tickvals]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=sleep_in_hours,
                name="Sleep-in",
                mode="lines",
                line=dict(color="#6366f1", width=2),
                customdata=dated["sleep_in"],
                hovertemplate="Sleep-in: %{customdata}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=wake_up_hours,
                name="Wake-up",
                mode="lines",
                line=dict(color="#f59e0b", width=2),
                customdata=dated["wake_up"],
                hovertemplate="Wake-up: %{customdata}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=dated["sleep_minutes"] / 60,
                name="Sleep hours",
                mode="lines",
                line=dict(color="#10b981", width=2, dash="dot"),
                yaxis="y2",
                hovertemplate="Sleep: %{y:.1f} h<extra></extra>",
            )
        )
        fig.update_layout(
            height=360,
            margin=dict(l=12, r=12, t=10, b=42),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis=dict(title="Date", type="date"),
            yaxis=dict(title="Clock time", tickmode="array", tickvals=tickvals, ticktext=ticktext),
            yaxis2=dict(
                title="Sleep hours",
                overlaying="y",
                side="right",
                rangemode="tozero",
                showgrid=False,
            ),
        )
        apply_chart_theme(fig)
        st.plotly_chart(fig, width="stretch")

    table = pd.DataFrame(
        {
            "Date": daily["date"].dt.strftime("%Y-%m-%d").fillna("—"),
            "Sleep-in time": daily["sleep_in"].replace("", "—"),
            "Wake-up time": daily["wake_up"].replace("", "—"),
            "Sleep hours": daily["sleep_minutes"].map(format_minutes),
        }
    )
    st.dataframe(table, width="stretch", hide_index=True)
    if not daily["has_clock_time"].any():
        st.caption(
            "Sleep-in and wake-up clock times need absolute timestamps in the sleep export. "
            "This data only has elapsed-time fragments, so only sleep length is shown."
        )


def render_sleep_tab(sleep: pd.DataFrame, sleep_records: pd.DataFrame, loaded_sleep_records) -> None:
    if sleep.empty and sleep_records.empty:
        if loaded_sleep_records:
            st.info("Sleep files were loaded, but no usable sleep-stage or sleep-session rows were found.")
            render_record_summary(loaded_sleep_records)
        else:
            st.info("No sleep records were loaded.")
        return

    date_parts = []
    if not sleep.empty and "start_dt" in sleep.columns:
        date_parts.append(sleep["start_dt"])
    if not sleep_records.empty and "start_dt" in sleep_records.columns:
        date_parts.append(sleep_records["start_dt"])
    date_window = date_window_slider(
        pd.concat(date_parts) if date_parts else pd.Series(dtype="datetime64[ns]"),
        key="sleep_date_window",
    )

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

        all_daily = daily_sleep_summary(sleep)
        daily = filter_by_date(all_daily, "sleep_in_dt", date_window)
        if not daily.empty:
            render_daily_sleep(daily)
            st.divider()
        elif not all_daily.empty:
            st.info("No sleep sessions fall in the selected date range.")
            st.divider()

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
        render_sleep_records(sleep_records, date_window)


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

    body_window = (
        date_window_slider(body["record_dt"], key="body_date_window")
        if has_usable_time(body, "record_dt")
        else None
    )

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

        st.markdown("##### Time spent in each weight range")
        bands = weight_band_durations(weighted, "weight", "record_dt", bin_size=5.0) if weight_has_time else pd.DataFrame()
        if not bands.empty:
            bands = bands.assign(share=bands["days"] / bands["days"].sum() * 100)
            band_chart, band_table = st.columns([1.4, 1])
            with band_chart:
                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=bands["days"],
                            y=bands["band_label"],
                            orientation="h",
                            marker_color="#2563eb",
                            customdata=bands["share"],
                            hovertemplate="%{y}<br>Time: %{x:.1f} days (%{customdata:.1f}%)<extra></extra>",
                        )
                    ]
                )
                fig.update_layout(
                    height=max(240, 60 + len(bands) * 46),
                    margin=dict(l=12, r=12, t=10, b=42),
                    xaxis_title="Days",
                    yaxis_title="Weight range",
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="#ffffff",
                )
                fig.update_yaxes(categoryorder="array", categoryarray=list(bands["band_label"]))
                apply_chart_theme(fig)
                st.plotly_chart(fig, width="stretch")
            with band_table:
                band_display = pd.DataFrame(
                    {
                        "Weight range": bands["band_label"],
                        "Days": bands["days"].map(lambda value: f"{value:,.1f}"),
                        "Share": bands["share"].map(lambda value: f"{value:.1f}%"),
                        "Cumulative %": bands["share"].cumsum().map(lambda value: f"{value:.1f}%"),
                    }
                )
                st.dataframe(band_display, width="stretch", hide_index=True)
            st.caption(
                "Time between consecutive weigh-ins is credited to the 5 kg band holding that "
                "interval's midpoint weight, then summed per band. Cumulative % is the share of "
                "time spent at or below each band."
            )
        elif weight_has_time:
            st.caption("Need at least two timestamped weigh-ins to total time per weight range.")
        else:
            st.caption("No usable timestamps, so time spent in each weight range can't be computed.")

        st.markdown("##### Body weight range over time")
        if weight_has_time:
            range_grain = st.radio(
                "Range period",
                ["Month", "Year"],
                horizontal=True,
                key="weight_range_period",
            )
            ranges = weight_range_by_period(
                filter_by_date(weighted, "record_dt", body_window),
                "weight",
                "record_dt",
                freq="M" if range_grain == "Month" else "Y",
            )
        else:
            range_grain = "Month"
            ranges = pd.DataFrame()

        if not ranges.empty:
            tick_format = "%Y-%m" if range_grain == "Month" else "%Y"
            ranges = ranges.assign(spread=ranges["high"] - ranges["low"])
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=ranges["period_start"],
                    y=ranges["spread"],
                    base=ranges["low"],
                    name="Range",
                    marker_color="#bfdbfe",
                    marker_line_color="#2563eb",
                    marker_line_width=1,
                    customdata=ranges[["low", "high", "mean", "spread"]],
                    hovertemplate=(
                        f"{range_grain}: %{{x|{tick_format}}}<br>"
                        "Low: %{customdata[0]:.1f} kg<br>"
                        "High: %{customdata[1]:.1f} kg<br>"
                        "Average: %{customdata[2]:.1f} kg<br>"
                        "Spread: %{customdata[3]:.1f} kg"
                        "<extra></extra>"
                    ),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=ranges["period_start"],
                    y=ranges["mean"],
                    mode="lines+markers",
                    name="Average",
                    line=dict(color="#1d4ed8", width=2),
                    marker=dict(size=6),
                    hovertemplate=f"{range_grain}: %{{x|{tick_format}}}<br>Average: %{{y:.1f}} kg<extra></extra>",
                )
            )
            fig.update_layout(
                height=380,
                margin=dict(l=12, r=12, t=10, b=42),
                xaxis_title=range_grain,
                yaxis_title="Weight (kg)",
                legend_title=None,
                barmode="overlay",
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
            )
            fig.update_xaxes(type="date")
            apply_chart_theme(fig)
            st.plotly_chart(fig, width="stretch")
            st.caption(
                f"Each bar spans the lowest-to-highest weight recorded in that {range_grain.lower()}; "
                "the line tracks the period average."
            )
        elif weight_has_time:
            st.caption("Need timestamped weigh-ins to show the weight range over time.")
        else:
            st.caption("No usable timestamps, so the weight range over time can't be computed.")

        st.markdown("##### Estimate future weight")
        proj_basis_days = {"Week": 7, "Month": 30}
        proj_horizons = {"1 month": 30, "3 months": 90, "6 months": 180, "1 year": 365}
        proj_colors = {"Week": "#f59e0b", "Month": "#16a34a"}
        forecast_source = (
            weighted.dropna(subset=["record_dt"]).sort_values("record_dt", kind="stable")
            if weight_has_time
            else pd.DataFrame()
        )

        if len(forecast_source) >= 2:
            selected_bases = st.multiselect(
                "Continue the trend from the past…",
                list(proj_basis_days.keys()),
                default=list(proj_basis_days.keys()),
                key="weight_projection_bases",
            )
            horizon_label = st.radio(
                "Project ahead",
                list(proj_horizons.keys()),
                horizontal=True,
                index=1,
                key="weight_projection_horizon",
            )
            horizon_days = proj_horizons[horizon_label]

            anchor_time = forecast_source["record_dt"].iloc[-1]
            anchor_value = float(forecast_source["weight"].iloc[-1])
            target_time = anchor_time + pd.Timedelta(days=horizon_days)

            forecasts = []
            missing = []
            for basis in selected_bases:
                window_days = proj_basis_days[basis]
                rate = trailing_rate_per_day(
                    forecast_source,
                    "weight",
                    "record_dt",
                    window_days=window_days,
                    min_span_days=window_days * 0.5,
                )
                if rate is None:
                    missing.append(basis)
                    continue
                forecasts.append((basis, rate, anchor_value + rate * horizon_days))

            if forecasts:
                metric_cols = st.columns(len(forecasts))
                for col, (basis, rate, target_value) in zip(metric_cols, forecasts):
                    col.metric(
                        f"If past {basis.lower()}'s trend holds",
                        f"{target_value:.1f} kg",
                        delta=f"{target_value - anchor_value:+.1f} kg in {horizon_label}",
                        delta_color="off",
                    )

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=forecast_source["record_dt"],
                    y=forecast_source["weight"],
                    mode="lines+markers",
                    name="Recorded",
                    line=dict(color="#2563eb", width=2),
                    marker=dict(size=6),
                    hovertemplate="Date: %{x|%Y-%m-%d}<br>Weight: %{y:.1f} kg<extra></extra>",
                )
            )
            for basis, rate, target_value in forecasts:
                fig.add_trace(
                    go.Scatter(
                        x=[anchor_time, target_time],
                        y=[anchor_value, target_value],
                        mode="lines",
                        name=f"Past {basis.lower()} trend",
                        line=dict(color=proj_colors.get(basis, "#64748b"), width=2, dash="dash"),
                        hovertemplate="%{x|%Y-%m-%d}<br>Projected: %{y:.1f} kg<extra></extra>",
                    )
                )
            fig.update_layout(
                height=380,
                margin=dict(l=12, r=12, t=10, b=42),
                xaxis_title="Date",
                yaxis_title="Weight (kg)",
                legend_title=None,
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
            )
            fig.update_xaxes(type="date")
            apply_chart_theme(fig)
            st.plotly_chart(fig, width="stretch")

            if missing:
                st.caption(
                    "Not enough weigh-ins in the past "
                    + " or ".join(basis.lower() for basis in missing)
                    + " to estimate that trend."
                )
            st.caption(
                "Projections continue the average kg-per-day change measured across the past "
                "week/month in a straight line from your latest weigh-in. It assumes the trend "
                "holds unchanged — a simple extrapolation, not a medical prediction."
            )
        elif weight_has_time:
            st.caption("Need at least two timestamped weigh-ins to project future weight.")
        else:
            st.caption("No usable timestamps, so future weight can't be projected.")

    if selected_metrics:
        has_record_time = has_usable_time(body, "record_dt")
        if has_record_time:
            time_grain = st.radio(
                "Body x-axis",
                list(TIME_GRAINS.keys()),
                horizontal=True,
                key="body_time_grain",
            )
            chart_data = filter_by_date(body, "record_dt", body_window).dropna(subset=["record_dt"]).copy()
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

        if has_record_time:
            x_hover = f"{x_title}: %{{x|%Y-%m-%d}}<br>"
        else:
            x_hover = f"{x_title}: %{{x}}<br>"

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
                    hovertemplate=f"{x_hover}{metric_label(metric)}: %{{y:.2f}}<extra></extra>",
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
