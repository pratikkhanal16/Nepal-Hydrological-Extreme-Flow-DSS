
import os
import re
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pydeck as pdk

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Nepal Hydrological Extreme Flow DSS Built by Pratik and Saugat",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# USER PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER = os.path.join(
    BASE_DIR,
    "data"
)

RESULTS_FOLDER = os.path.join(
    BASE_DIR,
    "results"
)

STATION_CSV = os.path.join(
    DATA_FOLDER,
    "Nepal_DHM_103_Streamflow_Stations.csv"
)

HYDROEVT_FILE = os.path.join(
    RESULTS_FOLDER,
    "Nepal_HydroEVT_All_Stations.xlsx"
)

ANNUAL_EXCEL = os.path.join(
    DATA_FOLDER,
    "Maximum Yearly Discharge.xlsx"
)

STATION_CSV = os.path.join(
    DATA_FOLDER,
    "Nepal_DHM_103_Streamflow_Stations.csv"
)

HYDROEVT_FILE = os.path.join(
    RESULTS_FOLDER,
    "Nepal_HydroEVT_All_Stations.xlsx"
)

ANNUAL_EXCEL = os.path.join(
    DATA_FOLDER,
    "Maximum Yearly Discharge.xlsx"
)

# ============================================================
# HELPERS
# ============================================================
def normalize_station_id(value):
    """
    Converts station IDs to a consistent string representation.
    Examples:
        420 -> '420'
        420.0 -> '420'
        '420.0' -> '420'
        ' 251.6 ' -> '251.6'
    """
    if pd.isna(value):
        return ""

    s = str(value).strip()

    try:
        num = float(s)
        if float(num).is_integer():
            return str(int(num))
        return str(num).rstrip("0").rstrip(".")
    except:
        return s


def normalize_station_columns(df):
    df = df.copy()
    new_cols = []
    for c in df.columns:
        if str(c).strip().lower() == "year":
            new_cols.append("Year")
        else:
            new_cols.append(normalize_station_id(c))
    df.columns = new_cols
    return df


def find_sheet(sheet_names, candidates):
    lower = {s.lower(): s for s in sheet_names}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def safe_number(value, digits=2):
    if pd.isna(value):
        return "NA"
    try:
        return f"{float(value):,.{digits}f}"
    except:
        return str(value)


def get_col(df, candidates):
    """
    Return first matching column name, case-insensitive.
    """
    mapping = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in mapping:
            return mapping[candidate.lower()]
    return None


def station_exists(df, station):
    if df is None or df.empty:
        return False
    return station in set(df["Station"].astype(str))


def build_download_excel(
    station_meta,
    summary_row,
    design_row,
    ranking_df,
    annual_df,
    analysis_name
):
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:

        pd.DataFrame([station_meta]).to_excel(
            writer,
            sheet_name="Station_Info",
            index=False
        )

        if summary_row is not None:
            pd.DataFrame([summary_row]).to_excel(
                writer,
                sheet_name="Summary",
                index=False
            )

        if design_row is not None:
            pd.DataFrame([design_row]).to_excel(
                writer,
                sheet_name="Design_Flows",
                index=False
            )

        if ranking_df is not None and not ranking_df.empty:
            ranking_df.to_excel(
                writer,
                sheet_name="Distribution_Ranking",
                index=False
            )

        if annual_df is not None and not annual_df.empty:
            annual_df.to_excel(
                writer,
                sheet_name="Annual_Series",
                index=False
            )

    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_station_metadata(path):
    df = pd.read_csv(path, dtype={"Station": str})
    df["Station"] = df["Station"].apply(normalize_station_id)
    return df


@st.cache_data
def load_hydroevt(path):
    xls = pd.ExcelFile(path)
    sheets = xls.sheet_names

    names = {
        "max_summary": find_sheet(sheets, ["MAX_Summary"]),
        "max_design": find_sheet(sheets, ["MAX_Design_Flows"]),
        "max_rank": find_sheet(sheets, ["MAX_Distribution_Ranking"]),
        "max_all": find_sheet(sheets, ["MAX_All_Distributions"]),

        "min_summary": find_sheet(sheets, ["MIN_Summary"]),
        "min_design": find_sheet(sheets, ["MIN_Design_Flows"]),
        "min_rank": find_sheet(sheets, ["MIN_Distribution_Ranking"]),
        "min_all": find_sheet(sheets, ["MIN_All_Distributions"]),
    }

    out = {}

    for key, sheet in names.items():
        if sheet is None:
            out[key] = pd.DataFrame()
            continue

        df = pd.read_excel(path, sheet_name=sheet)

        if "Station" in df.columns:
            df["Station"] = df["Station"].apply(normalize_station_id)

        out[key] = df

    return out, sheets


@st.cache_data
def load_annual_data(path):
    xls = pd.ExcelFile(path)
    sheets = xls.sheet_names

    max_sheet = find_sheet(sheets, ["Maximum", "maximum"])
    min_sheet = find_sheet(sheets, ["minimum", "Minimum"])

    max_df = pd.DataFrame()
    min_df = pd.DataFrame()

    if max_sheet:
        max_df = pd.read_excel(path, sheet_name=max_sheet)
        max_df = normalize_station_columns(max_df)

        if "Year" not in max_df.columns and len(max_df.columns) > 0:
            max_df = max_df.rename(columns={max_df.columns[0]: "Year"})

    if min_sheet:
        min_df = pd.read_excel(path, sheet_name=min_sheet)
        min_df = normalize_station_columns(min_df)

        if "Year" not in min_df.columns and len(min_df.columns) > 0:
            min_df = min_df.rename(columns={min_df.columns[0]: "Year"})

    return max_df, min_df


# ============================================================
# FILE CHECK
# ============================================================
missing = []

if not os.path.exists(STATION_CSV):
    missing.append(STATION_CSV)

if not os.path.exists(HYDROEVT_FILE):
    missing.append(HYDROEVT_FILE)

if not os.path.exists(ANNUAL_EXCEL):
    missing.append(ANNUAL_EXCEL)

if missing:
    st.error("Required file(s) not found:")
    for p in missing:
        st.code(p)
    st.stop()


stations = load_station_metadata(STATION_CSV)
evt, evt_sheets = load_hydroevt(HYDROEVT_FILE)
max_raw, min_raw = load_annual_data(ANNUAL_EXCEL)

# ============================================================
# CSS
# ============================================================
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.35rem;
            font-weight: 800;
            color: #0f3557;
            margin-bottom: 0.1rem;
        }

        .sub-title {
            color: #5c6770;
            margin-bottom: 1.2rem;
        }

        .station-card {
            border: 1px solid #dfe5eb;
            border-radius: 12px;
            padding: 1rem 1.1rem;
            background: #ffffff;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        }

        .small-note {
            font-size: 0.85rem;
            color: #667085;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# TITLE
# ============================================================
st.markdown(
    '<div class="main-title">🌊 Nepal Hydrological Extreme Flow DSS</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Flood-frequency and low-flow frequency decision-support system '
    'for DHM streamflow stations'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("DSS Controls")

river_values = (
    stations["River"]
    .dropna()
    .astype(str)
    .sort_values()
    .unique()
    .tolist()
)

selected_river = st.sidebar.selectbox(
    "River",
    ["All Rivers"] + river_values
)

if selected_river == "All Rivers":
    station_subset = stations.copy()
else:
    station_subset = stations[
        stations["River"].astype(str) == selected_river
    ].copy()

station_subset = station_subset.sort_values(
    ["River", "Station"]
)

station_options = {}

for _, row in station_subset.iterrows():
    sid = normalize_station_id(row["Station"])

    label = (
        f"{sid} | "
        f"{row.get('River', 'NA')} | "
        f"{row.get('Location', 'NA')}"
    )

    station_options[label] = sid

if not station_options:
    st.warning("No stations available for this river selection.")
    st.stop()

station_label = st.sidebar.selectbox(
    "Station",
    list(station_options.keys())
)

station_id = station_options[station_label]

analysis_type = st.sidebar.radio(
    "Analysis",
    ["Maximum Flow", "Minimum Flow"],
    horizontal=False
)

return_period = st.sidebar.selectbox(
    "Return Period (years)",
    [2, 5, 10, 25, 50, 100, 200],
    index=5
)

show_all_stations = st.sidebar.checkbox(
    "Show all stations on map",
    value=True
)

st.sidebar.divider()

st.sidebar.caption(
    "Station IDs are normalized automatically, so 420, 420.0 "
    "and '420' are treated as the same station."
)

# ============================================================
# ACTIVE TABLES
# ============================================================
if analysis_type == "Maximum Flow":
    summary_df = evt["max_summary"]
    design_df = evt["max_design"]
    ranking_df = evt["max_rank"]
    all_models_df = evt["max_all"]
    annual_source = max_raw
    analysis_short = "MAX"
else:
    summary_df = evt["min_summary"]
    design_df = evt["min_design"]
    ranking_df = evt["min_rank"]
    all_models_df = evt["min_all"]
    annual_source = min_raw
    analysis_short = "MIN"

# ============================================================
# SELECT STATION META
# ============================================================
meta_match = stations[
    stations["Station"] == station_id
].copy()

if meta_match.empty:
    st.error(
        f"Station metadata not found for station {station_id}. "
        "This should not occur because the dropdown is generated from the metadata table."
    )
    st.stop()

meta = meta_match.iloc[0].to_dict()

# ============================================================
# TOP INFORMATION
# ============================================================
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Station", station_id)
c2.metric("River", str(meta.get("River", "NA")))
c3.metric("Location", str(meta.get("Location", "NA")))

drainage = meta.get("Drainage_Area_km2", np.nan)
elev = meta.get("Elevation_m", np.nan)

c4.metric(
    "Drainage Area",
    "NA" if pd.isna(drainage) else f"{float(drainage):,.0f} km²"
)

c5.metric(
    "Elevation",
    "NA" if pd.isna(elev) else f"{float(elev):,.0f} m"
)

# ============================================================
# FIND HYDROEVT ROWS
# ============================================================
summary_match = pd.DataFrame()
design_match = pd.DataFrame()
rank_match = pd.DataFrame()
all_model_match = pd.DataFrame()

if not summary_df.empty and "Station" in summary_df.columns:
    summary_match = summary_df[
        summary_df["Station"] == station_id
    ].copy()

if not design_df.empty and "Station" in design_df.columns:
    design_match = design_df[
        design_df["Station"] == station_id
    ].copy()

if not ranking_df.empty and "Station" in ranking_df.columns:
    rank_match = ranking_df[
        ranking_df["Station"] == station_id
    ].copy()

if not all_models_df.empty and "Station" in all_models_df.columns:
    all_model_match = all_models_df[
        all_models_df["Station"] == station_id
    ].copy()

summary_row = (
    summary_match.iloc[0].to_dict()
    if not summary_match.empty
    else None
)

design_row = (
    design_match.iloc[0].to_dict()
    if not design_match.empty
    else None
)

# ============================================================
# TABS
# ============================================================
tab_overview, tab_frequency, tab_models, tab_map, tab_download = st.tabs(
    [
        "📊 Overview",
        "📈 Frequency Analysis",
        "🏆 Distribution Models",
        "🗺️ Station Map",
        "⬇️ Download"
    ]
)

# ============================================================
# OVERVIEW TAB
# ============================================================
with tab_overview:

    left, right = st.columns([1.05, 1.35])

    with left:

        st.subheader("Station Information")

        station_info = pd.DataFrame(
            {
                "Attribute": [
                    "Station",
                    "River",
                    "Location",
                    "Latitude",
                    "Longitude",
                    "Elevation (m)",
                    "Drainage Area (km²)",
                    "DHM Published From",
                    "DHM Published To",
                ],
                "Value": [
                    station_id,
                    meta.get("River", "NA"),
                    meta.get("Location", "NA"),
                    safe_number(meta.get("Latitude"), 5),
                    safe_number(meta.get("Longitude"), 5),
                    safe_number(meta.get("Elevation_m"), 0),
                    safe_number(meta.get("Drainage_Area_km2"), 1),
                    meta.get("Published_From", "NA"),
                    meta.get("Published_To", "NA"),
                ]
            }
        )

        st.dataframe(
            station_info,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("HydroEVT Summary")

        if summary_row is None:
            st.warning(
                f"No {analysis_type.lower()} HydroEVT result was found "
                f"for station {station_id}. This can occur if the station "
                "had fewer than the required number of valid years."
            )
        else:
            n_col = get_col(summary_match, ["N"])
            start_col = get_col(summary_match, ["Start_Year"])
            end_col = get_col(summary_match, ["End_Year"])
            mean_col = get_col(summary_match, ["Mean"])
            best_col = get_col(summary_match, ["Best_Distribution"])
            trend_col = get_col(summary_match, ["MK_Trend"])
            p_col = get_col(summary_match, ["MK_p_value"])
            slope_col = get_col(summary_match, ["Sen_Slope"])

            a, b = st.columns(2)

            with a:
                if n_col:
                    st.metric("Valid Years", summary_row[n_col])

                if mean_col:
                    st.metric(
                        "Mean Extreme Flow",
                        f"{float(summary_row[mean_col]):,.2f} m³/s"
                    )

                if best_col:
                    st.metric(
                        "Best Distribution",
                        str(summary_row[best_col])
                    )

            with b:
                if start_col and end_col:
                    st.metric(
                        "Analysis Record",
                        f"{int(summary_row[start_col])}–"
                        f"{int(summary_row[end_col])}"
                    )

                if trend_col:
                    st.metric(
                        "Mann–Kendall Trend",
                        str(summary_row[trend_col])
                    )

                if p_col:
                    st.metric(
                        "MK p-value",
                        f"{float(summary_row[p_col]):.4f}"
                    )

                if slope_col:
                    st.caption(
                        f"Sen's slope: "
                        f"{float(summary_row[slope_col]):.4f}"
                    )

    with right:

        st.subheader(
            f"Annual {analysis_type} Series"
        )

        annual_station = pd.DataFrame()

        if (
            annual_source is not None
            and not annual_source.empty
            and station_id in annual_source.columns
        ):

            annual_station = annual_source[
                ["Year", station_id]
            ].copy()

            annual_station["Year"] = pd.to_numeric(
                annual_station["Year"],
                errors="coerce"
            )

            annual_station[station_id] = pd.to_numeric(
                annual_station[station_id],
                errors="coerce"
            )

            annual_station = annual_station.dropna()

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=annual_station["Year"],
                    y=annual_station[station_id],
                    mode="lines+markers",
                    name="Observed",
                    hovertemplate=(
                        "Year=%{x}<br>"
                        "Discharge=%{y:.2f} m³/s<extra></extra>"
                    )
                )
            )

            if len(annual_station) >= 2:
                z = np.polyfit(
                    annual_station["Year"].values,
                    annual_station[station_id].values,
                    1
                )

                trend_y = np.polyval(
                    z,
                    annual_station["Year"].values
                )

                fig.add_trace(
                    go.Scatter(
                        x=annual_station["Year"],
                        y=trend_y,
                        mode="lines",
                        name="Linear trend"
                    )
                )

            fig.update_layout(
                template="plotly_white",
                height=500,
                xaxis_title="Year",
                yaxis_title="Discharge (m³/s)",
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h")
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.warning(
                f"Station {station_id} was not found in the "
                f"{analysis_type.lower()} annual-series sheet."
            )

# ============================================================
# FREQUENCY TAB
# ============================================================
with tab_frequency:

    st.subheader(
        f"{analysis_type} – Return Period Results"
    )

    if design_row is None:

        st.warning(
            "No design-flow table is available for this station."
        )

    else:

        # Identify return period columns robustly
        if analysis_type == "Maximum Flow":
            return_map = {
                2: ["Q2"],
                5: ["Q5"],
                10: ["Q10"],
                25: ["Q25"],
                50: ["Q50"],
                100: ["Q100"],
                200: ["Q200"],
            }
        else:
            return_map = {
                2: ["Qmin2", "Q2"],
                5: ["Qmin5", "Q5"],
                10: ["Qmin10", "Q10"],
                25: ["Qmin25", "Q25"],
                50: ["Qmin50", "Q50"],
                100: ["Qmin100", "Q100"],
                200: ["Qmin200", "Q200"],
            }

        rp_values = {}

        for T, candidates in return_map.items():
            col = get_col(design_match, candidates)
            rp_values[T] = (
                design_row[col]
                if col is not None
                else np.nan
            )

        # Selected design event
        selected_q = rp_values.get(
            return_period,
            np.nan
        )

        st.info(
            f"Selected return period: **{return_period} years**  |  "
            f"Design discharge: **"
            f"{'NA' if pd.isna(selected_q) else f'{float(selected_q):,.2f} m³/s'}**"
        )

        cols = st.columns(7)

        for i, T in enumerate([2, 5, 10, 25, 50, 100, 200]):
            q = rp_values[T]

            cols[i].metric(
                f"{'Q' if analysis_type == 'Maximum Flow' else 'Qmin'}{T}",
                "NA" if pd.isna(q) else f"{float(q):,.1f}"
            )

        rp_df = pd.DataFrame(
            {
                "Return Period": list(rp_values.keys()),
                "Design Discharge": list(rp_values.values())
            }
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=rp_df["Return Period"],
                y=rp_df["Design Discharge"],
                mode="lines+markers",
                name="Best-fit design flow",
                hovertemplate=(
                    "T=%{x} years<br>"
                    "Q=%{y:.2f} m³/s<extra></extra>"
                )
            )
        )

        if not pd.isna(selected_q):
            fig.add_trace(
                go.Scatter(
                    x=[return_period],
                    y=[selected_q],
                    mode="markers",
                    marker=dict(size=16, symbol="diamond"),
                    name=f"Selected T={return_period}"
                )
            )

        fig.update_layout(
            template="plotly_white",
            height=520,
            xaxis_type="log",
            xaxis_title="Return Period (years)",
            yaxis_title="Design Discharge (m³/s)",
            margin=dict(l=20, r=20, t=20, b=20)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            rp_df,
            use_container_width=True,
            hide_index=True
        )

    # all distributions return levels if available
    if not all_model_match.empty:

        st.subheader(
            "Return Levels from All Candidate Distributions"
        )

        dist_col = get_col(
            all_model_match,
            ["Distribution"]
        )

        rp_col = get_col(
            all_model_match,
            ["Return_Period", "Return Period"]
        )

        q_col = get_col(
            all_model_match,
            ["Discharge", "Design Flood"]
        )

        if dist_col and rp_col and q_col:

            plot_df = all_model_match[
                [dist_col, rp_col, q_col]
            ].copy()

            plot_df[rp_col] = pd.to_numeric(
                plot_df[rp_col],
                errors="coerce"
            )

            plot_df[q_col] = pd.to_numeric(
                plot_df[q_col],
                errors="coerce"
            )

            plot_df = plot_df.dropna()

            fig2 = px.line(
                plot_df,
                x=rp_col,
                y=q_col,
                color=dist_col,
                markers=True,
                log_x=True,
                labels={
                    rp_col: "Return Period (years)",
                    q_col: "Discharge (m³/s)",
                    dist_col: "Distribution"
                }
            )

            fig2.update_layout(
                template="plotly_white",
                height=520
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

# ============================================================
# MODEL TAB
# ============================================================
with tab_models:

    st.subheader("Probability Distribution Ranking")

    if rank_match.empty:

        st.warning(
            "No distribution ranking is available for this station."
        )

    else:

        total_rank_col = get_col(
            rank_match,
            ["Total_Rank"]
        )

        distribution_col = get_col(
            rank_match,
            ["Distribution"]
        )

        if total_rank_col and distribution_col:

            rank_plot = rank_match.copy()

            rank_plot[total_rank_col] = pd.to_numeric(
                rank_plot[total_rank_col],
                errors="coerce"
            )

            rank_plot = rank_plot.sort_values(
                total_rank_col
            )

            hover_cols = [
                c for c in [
                    get_col(rank_plot, ["RMSE"]),
                    get_col(rank_plot, ["MAE"]),
                    get_col(rank_plot, ["KS"]),
                    get_col(rank_plot, ["KS_p_value"])
                ]
                if c is not None
            ]

            fig = px.bar(
                rank_plot,
                x=distribution_col,
                y=total_rank_col,
                hover_data=hover_cols
            )

            fig.update_layout(
                template="plotly_white",
                height=480,
                xaxis_title="Distribution",
                yaxis_title="Total Rank"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.dataframe(
            rank_match,
            use_container_width=True,
            hide_index=True
        )

# ============================================================
# MAP TAB
# ============================================================
with tab_map:

    st.subheader("DHM Streamflow Station Map")

    map_df = stations.copy()

    if not show_all_stations:
        map_df = map_df[
            map_df["Station"] == station_id
        ].copy()

    map_df["selected"] = (
        map_df["Station"] == station_id
    )

    map_df["radius"] = np.where(
        map_df["selected"],
        14000,
        5000
    )

    # station label for tooltip
    map_df["label"] = (
        "Station " +
        map_df["Station"].astype(str) +
        " | " +
        map_df["River"].astype(str) +
        " | " +
        map_df["Location"].astype(str)
    )

    lat = float(meta["Latitude"])
    lon = float(meta["Longitude"])

map_df["fill_color"] = map_df["selected"].apply(
    lambda sel: [220, 38, 38, 220] if sel else [37, 99, 235, 150]
)

scatter_layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_df,
    get_position="[Longitude, Latitude]",
    get_radius="radius",
    pickable=True,
    auto_highlight=True,
    get_fill_color="fill_color"
)

    text_layer = pdk.Layer(
        "TextLayer",
        data=map_df[map_df["selected"]],
        get_position="[Longitude, Latitude]",
        get_text="label",
        get_size=16,
        get_alignment_baseline="'bottom'",
        pickable=False
    )

    view_state = pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=7.5,
        pitch=0
    )

    deck = pdk.Deck(
        layers=[scatter_layer, text_layer],
        initial_view_state=view_state,
        tooltip={
            "html": (
                "<b>{label}</b><br/>"
                "Elevation: {Elevation_m} m<br/>"
                "Drainage area: {Drainage_Area_km2} km²<br/>"
                "Lat: {Latitude}<br/>"
                "Lon: {Longitude}"
            )
        }
    )

    st.pydeck_chart(
        deck,
        use_container_width=True
    )

    st.caption(
        "Red point = selected station. Blue points = other DHM stations."
    )

# ============================================================
# DOWNLOAD TAB
# ============================================================
with tab_download:

    st.subheader("Export Selected Station Results")

    annual_station = pd.DataFrame()

    if (
        annual_source is not None
        and not annual_source.empty
        and station_id in annual_source.columns
    ):
        annual_station = annual_source[
            ["Year", station_id]
        ].copy()

        annual_station = annual_station.rename(
            columns={station_id: "Discharge"}
        )

        annual_station["Year"] = pd.to_numeric(
            annual_station["Year"],
            errors="coerce"
        )

        annual_station["Discharge"] = pd.to_numeric(
            annual_station["Discharge"],
            errors="coerce"
        )

        annual_station = annual_station.dropna()

    export_bytes = build_download_excel(
        station_meta=meta,
        summary_row=summary_row,
        design_row=design_row,
        ranking_df=rank_match,
        annual_df=annual_station,
        analysis_name=analysis_type
    )

    filename = (
        f"Station_{station_id}_"
        f"{analysis_short}_HydroEVT_DSS.xlsx"
    )

    st.download_button(
        label="⬇️ Download selected station analysis",
        data=export_bytes,
        file_name=filename,
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True
    )

    st.subheader("Data Quality Check")

    qc = {
        "Station metadata available": True,
        "HydroEVT summary available": summary_row is not None,
        "Design-flow results available": design_row is not None,
        "Distribution ranking available": not rank_match.empty,
        "Annual series available": not annual_station.empty,
    }

    qc_df = pd.DataFrame(
        {
            "Check": qc.keys(),
            "Available": qc.values()
        }
    )

    st.dataframe(
        qc_df,
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# FOOTER
# ============================================================
st.divider()

st.caption(
    "Nepal Hydrological Extreme Flow DSS | "
    "Station metadata from DHM Streamflow Summary | "
    "Frequency-analysis outputs from HydroEVT workflow."
)
