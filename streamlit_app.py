from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="NLR ESIF PUE / ERE Explorer",
    page_icon="",
    layout="wide",
)

DATASET_PAGE = "https://data.nlr.gov/submissions/300"
POWER_METRICS_URL = "https://data.nlr.gov/system/files/300/1757103411-esif.influx.buildingData.PUE.combined.parquet"
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
PARQUET_PATH = DATA_DIR / "esif_dc_power_metrics.parquet"

SOURCE_COLUMNS = [
    "ts",
    "pue",
    "ere",
    "energy_reuse",
    "it_power_kw",
    "cooling_kw",
    "hvac_kw",
    "plug_and_light_kw",
    "pump_kw",
]

METRIC_LABELS = {
    "pue": "PUE",
    "ere": "ERE",
    "energy_reuse": "Energy reuse",
    "erf": "ERF (%)",
    "it_power_kw": "IT power (kW)",
    "cooling_kw": "Cooling (kW)",
    "hvac_kw": "HVAC (kW)",
    "plug_and_light_kw": "Plug and light (kW)",
    "pump_kw": "Pump (kW)",
}

RESAMPLE_OPTIONS = {
    "Raw": None,
    "Hourly": "h",
    "Daily": "D",
    "Weekly": "W",
    "Monthly": "ME",
    "Quarterly": "QE",
}

AGGREGATION_OPTIONS = {
    "Mean": "mean",
    "Median": "median",
}


def metric_label(column: str) -> str:
    return METRIC_LABELS.get(column, column)


def download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_suffix(destination.suffix + ".part")

    if destination.exists() and destination.stat().st_size > 0:
        return destination

    with requests.get(url, stream=True, timeout=(10, 120)) as response:
        response.raise_for_status()
        with partial_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)

    partial_path.replace(destination)
    return destination


@st.cache_data(show_spinner=False)
def load_power_metrics(download_if_missing: bool) -> pd.DataFrame:
    if not PARQUET_PATH.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"Data file not found at {PARQUET_PATH}. Enable download in the sidebar or place the Parquet file there."
            )
        download_file(POWER_METRICS_URL, PARQUET_PATH)

    raw = pd.read_parquet(PARQUET_PATH, columns=SOURCE_COLUMNS)
    df = raw.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    for column in df.columns:
        if column != "ts":
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["ts"]).sort_values("ts").set_index("ts")

    required = ["pue", "ere"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Required source columns are missing: {missing}")

    valid_erf_inputs = df["pue"].gt(0) & df["pue"].notna() & df["ere"].notna()
    df["erf"] = ((1 - (df["ere"] / df["pue"])) * 100).where(valid_erf_inputs)
    return df


def available_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column]) and df[column].notna().any()
    ]


def resample_timeseries(
    df: pd.DataFrame,
    selected_columns: Iterable[str],
    resample_rule: str | None,
    aggregation: str,
    rolling_points: int,
) -> pd.DataFrame:
    selected_columns = list(selected_columns)
    plot_data = df[selected_columns].copy()

    if resample_rule:
        resampler = plot_data.resample(resample_rule)
        if aggregation == "median":
            plot_data = resampler.median(numeric_only=True)
        else:
            plot_data = resampler.mean(numeric_only=True)

    if rolling_points > 1:
        plot_data = plot_data.rolling(window=rolling_points, min_periods=1).mean()

    return plot_data.dropna(how="all")


def normalized_to_first_valid(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty or valid.iloc[0] == 0:
        return series
    return series / valid.iloc[0] * 100


def chart_download_frame(plot_data: pd.DataFrame, normalized: bool) -> pd.DataFrame:
    download_df = plot_data.copy()
    if normalized:
        for column in download_df.columns:
            download_df[column] = normalized_to_first_valid(download_df[column])

    download_df = download_df.reset_index()
    download_df = download_df.rename(columns={"ts": "Timestamp"})
    download_df.columns = [
        "Timestamp" if column == "Timestamp" else f"{metric_label(column)} ({column})"
        for column in download_df.columns
    ]
    return download_df


def build_timeseries_figure(
    plot_data: pd.DataFrame,
    selected_columns: list[str],
    normalized: bool,
    show_markers: bool,
    fixed_pue_ere_scale: bool,
) -> go.Figure:
    mode = "lines+markers" if show_markers else "lines"
    selected_set = set(selected_columns)
    pue_ere_only = bool(selected_set) and selected_set.issubset({"pue", "ere"})

    def add_trace(fig: go.Figure, column: str, secondary_y: bool | None = None) -> None:
        y = normalized_to_first_valid(plot_data[column]) if normalized else plot_data[column]
        value_label = "Indexed value" if normalized else "Value"
        trace = go.Scatter(
            x=plot_data.index,
            y=y,
            name=metric_label(column),
            mode=mode,
            line=dict(width=2.5),
            hovertemplate=(
                f"{metric_label(column)}"
                f"<br>Source column: <b>{column}</b>"
                "<br>%{x|%Y-%m-%d %H:%M}"
                f"<br>{value_label}=%{{y:.3f}}"
                "<extra></extra>"
            ),
        )
        if secondary_y is None:
            fig.add_trace(trace)
        else:
            fig.add_trace(trace, secondary_y=secondary_y)

    if pue_ere_only and not normalized and len(selected_columns) == 2:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if "pue" in selected_columns:
            add_trace(fig, "pue", secondary_y=False)
        if "ere" in selected_columns:
            add_trace(fig, "ere", secondary_y=True)
        fig.update_yaxes(title_text="PUE from source column 'pue'", secondary_y=False)
        fig.update_yaxes(title_text="ERE from source column 'ere'", secondary_y=True)
        if fixed_pue_ere_scale:
            fig.update_yaxes(range=[1.0, 2.0], secondary_y=False)
            fig.update_yaxes(range=[-1.0, 2.0], secondary_y=True)
    else:
        fig = go.Figure()
        for column in selected_columns:
            add_trace(fig, column)

        if normalized:
            y_title = "Indexed value (first valid point = 100)"
        elif len(selected_columns) == 1:
            y_title = f"{metric_label(selected_columns[0])} from source column '{selected_columns[0]}'"
        else:
            y_title = "Value"
        fig.update_yaxes(title_text=y_title)

        if fixed_pue_ere_scale and pue_ere_only and not normalized:
            if selected_columns == ["pue"]:
                fig.update_yaxes(range=[1.0, 2.0])
            elif selected_columns == ["ere"]:
                fig.update_yaxes(range=[-1.0, 2.0])

    selected_labels = ", ".join(metric_label(column) for column in selected_columns)
    fig.update_layout(
        title=f"Time-Series View: {selected_labels}",
        template="plotly_white",
        hovermode="x unified",
        height=650,
        margin=dict(l=50, r=50, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title="Timestamp", rangeslider_visible=True)
    return fig


def build_annual_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    annual = df[columns].groupby(df.index.year).agg(["mean", "median", "min", "max", "count"])
    annual.index.name = "Year"
    return annual


def flatten_annual_summary(annual: pd.DataFrame) -> pd.DataFrame:
    flattened = annual.copy()
    flattened.columns = [f"{metric_label(metric)} ({metric}) {stat}" for metric, stat in flattened.columns]
    return flattened.reset_index()


def build_annual_median_figure(annual: pd.DataFrame) -> go.Figure:
    annual_plot = annual[[("pue", "median"), ("ere", "median")]].copy()
    annual_plot.columns = ["PUE annual median from pue", "ERE annual median from ere"]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=annual_plot.index,
            y=annual_plot["PUE annual median from pue"],
            name="PUE annual median",
            mode="lines+markers",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=8),
            hovertemplate="Year=%{x}<br>PUE median=%{y:.3f}<br>Source column: pue<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=annual_plot.index,
            y=annual_plot["ERE annual median from ere"],
            name="ERE annual median",
            mode="lines+markers",
            line=dict(color="#d62728", width=3),
            marker=dict(size=8),
            hovertemplate="Year=%{x}<br>ERE median=%{y:.3f}<br>Source column: ere<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Annual Median PUE and ERE",
        template="plotly_white",
        height=500,
        hovermode="x unified",
        margin=dict(l=50, r=50, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title="Year")
    fig.update_yaxes(title_text="Annual median PUE from source column 'pue'", secondary_y=False)
    fig.update_yaxes(
        title_text="Annual median ERE from source column 'ere'",
        range=[-1.0, 2.0],
        secondary_y=True,
    )
    return fig


st.title("NLR ESIF PUE / ERE Explorer")
st.caption("Interactive time-series and annual summary views for the NLR ESIF power metrics dataset.")

with st.sidebar:
    st.header("Data")
    st.markdown(f"[NLR dataset page]({DATASET_PAGE})")
    download_if_missing = st.checkbox("Download data if missing", value=True)

try:
    df = load_power_metrics(download_if_missing=download_if_missing)
except Exception as exc:
    st.error(str(exc))
    st.stop()

available_columns = available_numeric_columns(df)
default_columns = [column for column in ["pue", "ere"] if column in available_columns]

with st.sidebar:
    st.header("Time Series")
    selected_columns = st.multiselect(
        "Series",
        options=available_columns,
        default=default_columns,
        format_func=metric_label,
    )
    resample_label = st.selectbox("Resample", options=list(RESAMPLE_OPTIONS), index=4)
    aggregation_label = st.selectbox("Aggregation", options=list(AGGREGATION_OPTIONS), index=0)
    rolling_points = st.slider("Rolling window", min_value=1, max_value=24, value=1, step=1)
    show_markers = st.checkbox("Show markers", value=False)
    normalized = st.checkbox("Index to 100", value=False)
    fixed_scale = st.checkbox("Use fixed PUE/ERE y-scale", value=False)

source_mapping = pd.DataFrame(
    [
        {"Display label": metric_label(column), "Source column": column, "Type": "calculated" if column == "erf" else "imported"}
        for column in available_columns
    ]
)

summary_cols = st.columns(3)
summary_cols[0].metric("Rows", f"{len(df):,}")
summary_cols[1].metric("Start", df.index.min().strftime("%Y-%m-%d"))
summary_cols[2].metric("End", df.index.max().strftime("%Y-%m-%d"))

with st.expander("Source column mapping", expanded=False):
    st.dataframe(source_mapping, hide_index=True, use_container_width=True)
    st.markdown(
        "PUE is plotted from `pue`; ERE is plotted from `ere`. "
        "The `energy_reuse` field remains available as a separate imported series. "
        "ERF is calculated as `(1 - ere / pue) * 100`."
    )

if not selected_columns:
    st.warning("Select at least one time-series column in the sidebar.")
    st.stop()

plot_data = resample_timeseries(
    df=df,
    selected_columns=selected_columns,
    resample_rule=RESAMPLE_OPTIONS[resample_label],
    aggregation=AGGREGATION_OPTIONS[aggregation_label],
    rolling_points=rolling_points,
)

st.subheader("Interactive Time-Series")
st.plotly_chart(
    build_timeseries_figure(
        plot_data=plot_data,
        selected_columns=selected_columns,
        normalized=normalized,
        show_markers=show_markers,
        fixed_pue_ere_scale=fixed_scale,
    ),
    use_container_width=True,
)

current_chart_download = chart_download_frame(plot_data, normalized=normalized)
st.download_button(
    "Download current time-series view as CSV",
    data=current_chart_download.to_csv(index=False).encode("utf-8"),
    file_name="nlr_esif_current_timeseries_view.csv",
    mime="text/csv",
)

st.subheader("Annual Summary")
annual_columns = [column for column in ["pue", "ere", "erf"] if column in available_columns]
annual_summary = build_annual_summary(df, annual_columns)
st.plotly_chart(build_annual_median_figure(annual_summary), use_container_width=True)

annual_download = flatten_annual_summary(annual_summary)
st.dataframe(annual_download, hide_index=True, use_container_width=True)
st.download_button(
    "Download annual summary as CSV",
    data=annual_download.to_csv(index=False).encode("utf-8"),
    file_name="nlr_esif_annual_summary.csv",
    mime="text/csv",
)
