import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import geopandas as gpd
import folium

from folium.plugins import Fullscreen, MarkerCluster
from streamlit_folium import st_folium

# Optional on-demand uncertainty engine. The app still runs if the helper
# is missing, but the bootstrap tab will ask for the precomputed workbook.
try:
    from hydroevt_uncertainty_v2 import analyze_station
    UNCERTAINTY_ENGINE_AVAILABLE = True
except Exception:
    analyze_station = None
    UNCERTAINTY_ENGINE_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Nepal Hydrological Extreme Flow DSS | Pratik & Saugat",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "data")
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")

STATION_CSV = os.path.join(DATA_FOLDER, "Nepal_DHM_103_Streamflow_Stations.csv")
HYDROEVT_FILE = os.path.join(RESULTS_FOLDER, "Nepal_HydroEVT_All_Stations.xlsx")
ANNUAL_EXCEL = os.path.join(DATA_FOLDER, "Maximum Yearly Discharge.xlsx")
UNCERTAINTY_FILE = os.path.join(RESULTS_FOLDER, "Nepal_HydroEVT_Uncertainty_v2.xlsx")

# Prefer a single GeoPackage. Fallbacks allow the user's current uploaded names.
BASIN_CANDIDATES = [
    os.path.join(DATA_FOLDER, "All_DHM_Watersheds_Sorted.gpkg"),
    os.path.join(DATA_FOLDER, "All_DHM_Watersheds_Sorted(1).gpkg"),
    os.path.join(DATA_FOLDER, "All_DHM_Watersheds.shp"),
    os.path.join(DATA_FOLDER, "All_DHM_Watersheds(1).shp"),
]
BASIN_FILE = next((p for p in BASIN_CANDIDATES if os.path.exists(p)), None)

DEFAULT_MIN_VALID_YEARS = 20
DEFAULT_BOOTSTRAPS = 500
FINAL_BOOTSTRAPS = 2000
RETURN_PERIODS = [2, 5, 10, 25, 50, 100, 200]
PLOTLY_TEMPLATE = "plotly_dark"


# ============================================================
# THEME / CSS
# ============================================================
st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at 15% 10%, #0f2438 0%, #0a1420 45%, #060b12 100%);
        }
        section[data-testid="stSidebar"] {
            background: #0b1622;
            border-right: 1px solid #16324a;
        }
        h1, h2, h3, h4 { color: #e8f1fb !important; }
        .main-title {
            font-size: 2.6rem;
            font-weight: 800;
            background: linear-gradient(90deg, #38bdf8, #6ee7b7 60%, #38bdf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.1rem;
            letter-spacing: -0.5px;
        }
        .sub-title {
            color: #93a5b8;
            margin-bottom: 1.4rem;
            font-size: 1.02rem;
        }
        .kpi-card {
            background: linear-gradient(145deg, rgba(56,189,248,0.10), rgba(110,231,183,0.05));
            border: 1px solid rgba(148,197,255,0.18);
            border-radius: 16px;
            padding: 0.95rem 1.1rem;
            box-shadow: 0 4px 18px rgba(0,0,0,0.25);
            height: 100%;
        }
        .kpi-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 1.1px;
            color: #7fa0c0;
            margin-bottom: 0.25rem;
        }
        .kpi-value {
            font-size: 1.55rem;
            font-weight: 750;
            color: #f1f8ff;
        }
        .kpi-sub {
            font-size: 0.78rem;
            color: #93a5b8;
            margin-top: 0.15rem;
        }
        .kpi-sub.good { color: #34d399; }
        .kpi-sub.bad { color: #f87171; }
        .kpi-sub.warn { color: #facc15; }
        .badge {
            display: inline-block;
            padding: 0.18rem 0.65rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.4px;
        }
        .badge-green { background: rgba(52,211,153,0.15); color: #34d399; }
        .badge-red { background: rgba(248,113,113,0.15); color: #f87171; }
        .badge-blue { background: rgba(56,189,248,0.15); color: #38bdf8; }
        .badge-yellow { background: rgba(250,204,21,0.15); color: #facc15; }
        .badge-gray { background: rgba(148,163,184,0.15); color: #94a3b8; }
        [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi_card(label, value, sub=None, sub_class="neutral"):
    sub_html = f'<div class="kpi-sub {sub_class}">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(text, kind="blue"):
    return f'<span class="badge badge-{kind}">{text}</span>'


# ============================================================
# GENERIC HELPERS
# ============================================================
def normalize_station_id(value):
    if pd.isna(value):
        return ""
    s = str(value).strip()
    try:
        num = float(s)
        if float(num).is_integer():
            return str(int(num))
        return str(num).rstrip("0").rstrip(".")
    except Exception:
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
    except Exception:
        return str(value)


def get_col(df, candidates):
    if df is None or df.empty:
        return None
    mapping = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in mapping:
            return mapping[candidate.lower()]
    return None


def longest_consecutive_run(values):
    values = sorted(set(int(v) for v in values))
    if not values:
        return 0
    longest = current = 1
    for a, b in zip(values[:-1], values[1:]):
        if b == a + 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def record_class(n):
    if n < 10:
        return "Insufficient"
    if n < 20:
        return "Limited"
    if n < 30:
        return "Moderate"
    if n < 50:
        return "Good"
    return "Strong"


def annual_record_quality(annual_df):
    """One row per station using actual observed years; missing years are never filled."""
    if annual_df is None or annual_df.empty or "Year" not in annual_df.columns:
        return pd.DataFrame()

    years_all = pd.to_numeric(annual_df["Year"], errors="coerce")
    rows = []
    for station in [c for c in annual_df.columns if c != "Year"]:
        q = pd.to_numeric(annual_df[station], errors="coerce")
        m = years_all.notna() & q.notna()
        observed_years = sorted(set(years_all[m].astype(int).tolist()))
        n = len(observed_years)

        if n == 0:
            rows.append({
                "Station": normalize_station_id(station),
                "Valid_Years": 0,
                "First_Year": np.nan,
                "Last_Year": np.nan,
                "Span_Years": 0,
                "Missing_Years": 0,
                "Completeness_pct": 0.0,
                "Longest_Gap_Years": 0,
                "Record_Class": "Insufficient",
            })
            continue

        first_y, last_y = observed_years[0], observed_years[-1]
        span_years = last_y - first_y + 1
        expected = set(range(first_y, last_y + 1))
        missing = sorted(expected.difference(observed_years))
        completeness = 100.0 * n / span_years if span_years else np.nan

        rows.append({
            "Station": normalize_station_id(station),
            "Valid_Years": n,
            "First_Year": first_y,
            "Last_Year": last_y,
            "Span_Years": span_years,
            "Missing_Years": len(missing),
            "Completeness_pct": completeness,
            "Longest_Gap_Years": longest_consecutive_run(missing),
            "Record_Class": record_class(n),
        })

    return pd.DataFrame(rows)


def design_return_map(analysis_type):
    if analysis_type == "Maximum Flow":
        return {T: [f"Q{T}"] for T in RETURN_PERIODS}
    return {T: [f"Qmin{T}", f"1Q{T}", f"Q{T}"] for T in RETURN_PERIODS}


def extract_design_values(design_match, design_row, analysis_type):
    out = {}
    if design_row is None or design_match is None or design_match.empty:
        return {T: np.nan for T in RETURN_PERIODS}
    for T, candidates in design_return_map(analysis_type).items():
        c = get_col(design_match, candidates)
        out[T] = design_row[c] if c is not None else np.nan
    return out


# ============================================================
# LOADERS
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

    max_df, min_df = pd.DataFrame(), pd.DataFrame()
    if max_sheet:
        max_df = normalize_station_columns(pd.read_excel(path, sheet_name=max_sheet))
        if "Year" not in max_df.columns and len(max_df.columns) > 0:
            max_df = max_df.rename(columns={max_df.columns[0]: "Year"})
    if min_sheet:
        min_df = normalize_station_columns(pd.read_excel(path, sheet_name=min_sheet))
        if "Year" not in min_df.columns and len(min_df.columns) > 0:
            min_df = min_df.rename(columns={min_df.columns[0]: "Year"})
    return max_df, min_df


@st.cache_data
def load_basins(path):
    gdf = gpd.read_file(path)
    station_col = next((c for c in gdf.columns if str(c).strip().lower() == "station"), None)
    if station_col is None:
        raise ValueError("Watershed layer must contain a Station field.")
    if station_col != "Station":
        gdf = gdf.rename(columns={station_col: "Station"})
    gdf["Station"] = gdf["Station"].apply(normalize_station_id)

    area_col = next((c for c in gdf.columns if str(c).strip().lower() in ["area_km2", "area", "basin_area_km2"]), None)
    if area_col and area_col != "area_km2":
        gdf = gdf.rename(columns={area_col: "area_km2"})
    if "area_km2" not in gdf.columns:
        # Equal-area calculation fallback; only used if no area field exists.
        gdf_eq = gdf.to_crs(6933)
        gdf["area_km2"] = gdf_eq.geometry.area / 1e6

    if gdf.crs is None:
        raise ValueError("Watershed file has no CRS. Define it before using the DSS.")
    gdf = gdf.to_crs(4326)
    return gdf[[c for c in ["Station", "area_km2", "geometry"] if c in gdf.columns]].copy()


@st.cache_data
def load_uncertainty_workbook(path):
    if not os.path.exists(path):
        return {"max_fit": pd.DataFrame(), "max_boot": pd.DataFrame(), "max_choice": pd.DataFrame(),
                "min_fit": pd.DataFrame(), "min_boot": pd.DataFrame(), "min_choice": pd.DataFrame()}
    xls = pd.ExcelFile(path)
    mapping = {
        "max_fit": "MAX_Model_Fit_Uncertainty",
        "max_boot": "MAX_Quantile_Bootstrap",
        "max_choice": "MAX_Distribution_Choice",
        "min_fit": "MIN_Model_Fit_Uncertainty",
        "min_boot": "MIN_Quantile_Bootstrap",
        "min_choice": "MIN_Distribution_Choice",
    }
    out = {}
    for key, sheet in mapping.items():
        if sheet in xls.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet)
            if "Station" in df.columns:
                df["Station"] = df["Station"].apply(normalize_station_id)
            out[key] = df
        else:
            out[key] = pd.DataFrame()
    return out


@st.cache_data(show_spinner=False)
def cached_uncertainty(values_tuple, analysis, return_periods_tuple, n_boot):
    if not UNCERTAINTY_ENGINE_AVAILABLE:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    values = np.asarray(values_tuple, dtype=float)
    return analyze_station(
        values,
        analysis=analysis,
        return_periods=list(return_periods_tuple),
        n_boot=int(n_boot),
        seed=42,
    )


# ============================================================
# EXPORT
# ============================================================
def build_download_excel(
    station_meta,
    summary_row,
    design_row,
    annual_df,
    record_quality_row=None,
    basin_info=None,
    fit_df=None,
    boot_df=None,
    model_choice_df=None,
):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame([station_meta]).to_excel(writer, sheet_name="Station_Info", index=False)
        if basin_info:
            pd.DataFrame([basin_info]).to_excel(writer, sheet_name="Basin_Info", index=False)
        if record_quality_row:
            pd.DataFrame([record_quality_row]).to_excel(writer, sheet_name="Record_Quality", index=False)
        if summary_row is not None:
            pd.DataFrame([summary_row]).to_excel(writer, sheet_name="Summary", index=False)
        if design_row is not None:
            pd.DataFrame([design_row]).to_excel(writer, sheet_name="Design_Flows", index=False)
        if annual_df is not None and not annual_df.empty:
            annual_df.to_excel(writer, sheet_name="Annual_Series", index=False)
        if fit_df is not None and not fit_df.empty:
            fit_df.to_excel(writer, sheet_name="Distribution_Fit", index=False)
        if boot_df is not None and not boot_df.empty:
            boot_df.to_excel(writer, sheet_name="Bootstrap_Uncertainty", index=False)
        if model_choice_df is not None and not model_choice_df.empty:
            model_choice_df.to_excel(writer, sheet_name="Model_Choice_Uncertainty", index=False)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# REQUIRED FILES
# ============================================================
missing = [p for p in [STATION_CSV, HYDROEVT_FILE, ANNUAL_EXCEL] if not os.path.exists(p)]
if missing:
    st.error("Required file(s) not found:")
    for p in missing:
        st.code(p)
    st.stop()

stations = load_station_metadata(STATION_CSV)
evt, evt_sheets = load_hydroevt(HYDROEVT_FILE)
max_raw, min_raw = load_annual_data(ANNUAL_EXCEL)
unc_wb = load_uncertainty_workbook(UNCERTAINTY_FILE)

basins = pd.DataFrame()
if BASIN_FILE:
    try:
        basins = load_basins(BASIN_FILE)
    except Exception as e:
        st.warning(f"Watershed file found but could not be loaded: {e}")
else:
    st.warning("No watershed GeoPackage/Shapefile found in data/. The DSS will run without basin polygons.")

if isinstance(basins, gpd.GeoDataFrame) and not basins.empty:
    basin_area = basins[["Station", "area_km2"]].drop_duplicates("Station").rename(
        columns={"area_km2": "Delineated_Area_km2"}
    )
    stations = stations.merge(basin_area, on="Station", how="left")
else:
    stations["Delineated_Area_km2"] = np.nan


# ============================================================
# TITLE
# ============================================================
st.markdown('<div class="main-title">🌊 Nepal Hydrological Extreme Flow DSS</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Flood-frequency & low-flow decision-support system · '
    f'{len(stations)} DHM stations · record screening · watershed intelligence · bootstrap uncertainty</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR — ANALYSIS FIRST, THEN RECORD SCREENING, THEN STATION
# ============================================================
st.sidebar.header("⚙️ DSS Controls")
analysis_type = st.sidebar.radio("Analysis", ["Maximum Flow", "Minimum Flow"], horizontal=False)
annual_source = max_raw if analysis_type == "Maximum Flow" else min_raw

st.sidebar.markdown("#### Record screening")
min_valid_years = st.sidebar.slider(
    "Minimum valid annual years",
    min_value=10,
    max_value=50,
    value=DEFAULT_MIN_VALID_YEARS,
    step=1,
    help="Stations below this number of observed annual extremes are removed from analysis. Missing years are not interpolated.",
)

record_quality = annual_record_quality(annual_source)
if record_quality.empty:
    st.error("Could not calculate annual record quality from the active annual series.")
    st.stop()
record_quality["Eligible"] = record_quality["Valid_Years"] >= min_valid_years
eligible_ids = set(record_quality.loc[record_quality["Eligible"], "Station"])

eligible_stations = stations[stations["Station"].isin(eligible_ids)].copy()
excluded_count = int((~record_quality["Eligible"]).sum())

st.sidebar.caption(
    f"Eligible: {len(eligible_stations)} stations · Excluded: {excluded_count} stations with < {min_valid_years} valid years"
)

if eligible_stations.empty:
    st.error("No stations meet the selected minimum record length. Reduce the threshold.")
    st.stop()

river_values = eligible_stations["River"].dropna().astype(str).sort_values().unique().tolist()
selected_river = st.sidebar.selectbox("River", ["All Rivers"] + river_values)

station_subset = eligible_stations.copy() if selected_river == "All Rivers" else eligible_stations[
    eligible_stations["River"].astype(str) == selected_river
].copy()
station_subset = station_subset.sort_values(["River", "Station"])

station_options = {}
for _, row in station_subset.iterrows():
    sid = normalize_station_id(row["Station"])
    rq = record_quality[record_quality["Station"] == sid]
    nyrs = int(rq.iloc[0]["Valid_Years"]) if not rq.empty else 0
    label = f"{sid} | {row.get('River', 'NA')} | {row.get('Location', 'NA')} | n={nyrs}"
    station_options[label] = sid

if not station_options:
    st.warning("No eligible stations available for this river selection.")
    st.stop()

station_label = st.sidebar.selectbox("Station", list(station_options.keys()))
station_id = station_options[station_label]
return_period = st.sidebar.selectbox("Return Period (years)", RETURN_PERIODS, index=5)

show_all_stations = st.sidebar.checkbox("Show all eligible stations on map", value=True)
show_all_basins = st.sidebar.checkbox("Show all watershed boundaries", value=True)

st.sidebar.divider()
with st.sidebar.expander("📐 Active-data snapshot", expanded=False):
    st.metric("Eligible stations", len(eligible_stations))
    st.metric("Excluded stations", excluded_count)
    st.metric("Minimum valid years", min_valid_years)
    if isinstance(basins, gpd.GeoDataFrame):
        st.metric("Watersheds loaded", len(basins))


# ============================================================
# ACTIVE TABLES / MATCHES
# ============================================================
if analysis_type == "Maximum Flow":
    summary_df, design_df, ranking_df, all_models_df = (
        evt["max_summary"], evt["max_design"], evt["max_rank"], evt["max_all"]
    )
    analysis_short = "MAX"
    analysis_engine = "maximum"
    accent = "#38bdf8"
    ufit_all, uboot_all, uchoice_all = unc_wb["max_fit"], unc_wb["max_boot"], unc_wb["max_choice"]
else:
    summary_df, design_df, ranking_df, all_models_df = (
        evt["min_summary"], evt["min_design"], evt["min_rank"], evt["min_all"]
    )
    analysis_short = "MIN"
    analysis_engine = "minimum"
    accent = "#f59e0b"
    ufit_all, uboot_all, uchoice_all = unc_wb["min_fit"], unc_wb["min_boot"], unc_wb["min_choice"]


def match_station(df, sid=station_id):
    if df is None or df.empty or "Station" not in df.columns:
        return pd.DataFrame()
    return df[df["Station"] == sid].copy()


meta_match = stations[stations["Station"] == station_id].copy()
if meta_match.empty:
    st.error(f"Station metadata not found for station {station_id}.")
    st.stop()
meta = meta_match.iloc[0].to_dict()

summary_match = match_station(summary_df)
design_match = match_station(design_df)
summary_row = summary_match.iloc[0].to_dict() if not summary_match.empty else None
design_row = design_match.iloc[0].to_dict() if not design_match.empty else None

rq_match = record_quality[record_quality["Station"] == station_id]
rq_row = rq_match.iloc[0].to_dict() if not rq_match.empty else {}

basin_match = basins[basins["Station"] == station_id].copy() if isinstance(basins, gpd.GeoDataFrame) else pd.DataFrame()
basin_area_km2 = float(basin_match.iloc[0]["area_km2"]) if not basin_match.empty else np.nan

annual_station = pd.DataFrame()
if annual_source is not None and not annual_source.empty and station_id in annual_source.columns:
    annual_station = annual_source[["Year", station_id]].copy()
    annual_station["Year"] = pd.to_numeric(annual_station["Year"], errors="coerce")
    annual_station[station_id] = pd.to_numeric(annual_station[station_id], errors="coerce")
    annual_station = annual_station.dropna().sort_values("Year")

rp_values = extract_design_values(design_match, design_row, analysis_type)
selected_q = rp_values.get(return_period, np.nan)
specific_q = selected_q / basin_area_km2 if pd.notna(selected_q) and pd.notna(basin_area_km2) and basin_area_km2 > 0 else np.nan

# Precomputed uncertainty, if available.
fit_match = match_station(ufit_all)
boot_match = match_station(uboot_all)
choice_match = match_station(uchoice_all)


# ============================================================
# KPI ROW
# ============================================================
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    kpi_card("Station", station_id, str(meta.get("River", "NA")), "neutral")
with k2:
    kpi_card(
        "Delineated Basin Area",
        "NA" if pd.isna(basin_area_km2) else f"{basin_area_km2:,.0f} km²",
        "watershed polygon",
        "neutral",
    )
with k3:
    nyrs = int(rq_row.get("Valid_Years", 0))
    comp = rq_row.get("Completeness_pct", np.nan)
    kpi_card(
        "Valid Annual Extremes",
        f"{nyrs} yr",
        "NA" if pd.isna(comp) else f"{comp:.1f}% complete within observed span",
        "good" if nyrs >= min_valid_years else "bad",
    )
with k4:
    if summary_row is not None:
        mean_col = get_col(summary_match, ["Mean"])
        val = f"{float(summary_row[mean_col]):,.1f} m³/s" if mean_col else "NA"
        kpi_card(f"Mean {analysis_short}", val, str(rq_row.get("Record_Class", "")), "neutral")
    else:
        kpi_card(f"Mean {analysis_short}", "NA", "no HydroEVT match", "bad")
with k5:
    label = f"Q{return_period}" if analysis_type == "Maximum Flow" else f"1Q{return_period}"
    kpi_card(label, "NA" if pd.isna(selected_q) else f"{float(selected_q):,.1f} m³/s", "selected return level", "neutral")
with k6:
    trend_val, sig_text, sub_class = "NA", "", "neutral"
    if summary_row is not None:
        trend_col = get_col(summary_match, ["MK_Trend"])
        p_col = get_col(summary_match, ["MK_p_value"])
        if trend_col:
            trend_val = str(summary_row[trend_col])
        if p_col and pd.notna(summary_row[p_col]):
            pv = float(summary_row[p_col])
            sig_text = f"p={pv:.3f} · {'significant' if pv < 0.05 else 'not significant'}"
            if pv < 0.05:
                if analysis_type == "Maximum Flow" and "increas" in trend_val.lower():
                    sub_class = "bad"
                elif analysis_type == "Minimum Flow" and "decreas" in trend_val.lower():
                    sub_class = "bad"
                else:
                    sub_class = "good"
    kpi_card("Mann–Kendall Trend", trend_val, sig_text, sub_class)

st.write("")


# ============================================================
# TABS
# ============================================================
(
    tab_overview,
    tab_frequency,
    tab_uncertainty,
    tab_compare,
    tab_basin_quality,
    tab_map,
    tab_download,
) = st.tabs([
    "📊 Overview",
    "📈 Frequency Analysis",
    "🎯 Fit & Uncertainty",
    "🔀 Compare Stations",
    "🌐 Basin & Record Quality",
    "🗺️ Watershed Map",
    "⬇️ Export",
])


# ============================================================
# OVERVIEW
# ============================================================
with tab_overview:
    left, right = st.columns([1.0, 1.5])
    with left:
        st.markdown("#### Station & Basin Information")
        published_area = meta.get("Drainage_Area_km2", np.nan)
        area_diff_pct = np.nan
        if pd.notna(published_area) and pd.notna(basin_area_km2) and float(published_area) != 0:
            area_diff_pct = 100.0 * (basin_area_km2 - float(published_area)) / float(published_area)

        station_info = pd.DataFrame({
            "Attribute": [
                "Station", "River", "Location", "Latitude", "Longitude", "Elevation (m)",
                "Published drainage area (km²)", "Delineated basin area (km²)",
                "Area difference (%)", "Published From", "Published To",
            ],
            "Value": [
                station_id, meta.get("River", "NA"), meta.get("Location", "NA"),
                safe_number(meta.get("Latitude"), 5), safe_number(meta.get("Longitude"), 5),
                safe_number(meta.get("Elevation_m"), 0), safe_number(published_area, 1),
                safe_number(basin_area_km2, 1), safe_number(area_diff_pct, 1),
                meta.get("Published_From", "NA"), meta.get("Published_To", "NA"),
            ],
        })
        st.dataframe(station_info, use_container_width=True, hide_index=True)

        st.markdown("#### Record Quality")
        rq_show = pd.DataFrame({
            "Metric": ["Valid annual values", "Observed span", "Missing years within span", "Completeness", "Longest gap", "Record class", "Eligible"],
            "Value": [
                rq_row.get("Valid_Years", "NA"),
                f"{rq_row.get('First_Year', 'NA')}–{rq_row.get('Last_Year', 'NA')}",
                rq_row.get("Missing_Years", "NA"),
                f"{rq_row.get('Completeness_pct', np.nan):.1f}%" if pd.notna(rq_row.get("Completeness_pct", np.nan)) else "NA",
                f"{rq_row.get('Longest_Gap_Years', 'NA')} yr",
                rq_row.get("Record_Class", "NA"),
                f"Yes (≥ {min_valid_years} yr)",
            ],
        })
        st.dataframe(rq_show, use_container_width=True, hide_index=True)
        st.caption("Missing annual extremes are kept missing; the DSS does not interpolate annual maxima/minima.")

    with right:
        st.markdown(f"#### Annual {analysis_type} Series")
        if annual_station.empty:
            st.warning("No annual series is available for this station.")
        else:
            vals = annual_station[station_id].values
            years = annual_station["Year"].values
            roll = pd.Series(vals).rolling(window=5, min_periods=2, center=True).mean()
            extreme_idx = np.argsort(vals)[-3:] if analysis_type == "Maximum Flow" else np.argsort(vals)[:3]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=years, y=vals, name="Observed annual extreme",
                marker_color=accent, opacity=0.55,
                hovertemplate="Year=%{x}<br>Discharge=%{y:.2f} m³/s<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=years, y=roll, name="5-yr rolling mean", mode="lines",
                line=dict(color="#f1f8ff", width=2.5),
            ))
            if len(years) >= 2:
                z = np.polyfit(years, vals, 1)
                fig.add_trace(go.Scatter(
                    x=years, y=np.polyval(z, years), name="Linear trend", mode="lines",
                    line=dict(color="#f87171", width=2, dash="dash"),
                ))
            fig.add_trace(go.Scatter(
                x=years[extreme_idx], y=vals[extreme_idx], mode="markers",
                name="Largest floods" if analysis_type == "Maximum Flow" else "Severe low-flow years",
                marker=dict(size=13, color="#facc15", symbol="diamond", line=dict(width=1, color="#0b1622")),
            ))
            fig.update_layout(
                template=PLOTLY_TEMPLATE, height=480,
                xaxis_title="Year", yaxis_title="Discharge (m³/s)",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📌 Extreme-year detail"):
                extreme_df = annual_station.iloc[extreme_idx].copy()
                extreme_df = extreme_df.sort_values(station_id, ascending=(analysis_type == "Minimum Flow"))
                extreme_df = extreme_df.rename(columns={station_id: "Discharge (m³/s)"})
                st.dataframe(extreme_df, use_container_width=True, hide_index=True)


# ============================================================
# FREQUENCY ANALYSIS
# ============================================================
with tab_frequency:
    st.markdown(f"#### {analysis_type} — Return-Period Results")
    if design_row is None:
        st.warning("No design-flow table is available for this station.")
    else:
        selected_label = f"Q{return_period}" if analysis_type == "Maximum Flow" else f"1Q{return_period}"
        st.info(
            f"Selected return period: **{return_period} years**  |  {selected_label}: "
            f"**{'NA' if pd.isna(selected_q) else f'{float(selected_q):,.2f} m³/s'}**"
        )

        cols = st.columns(len(RETURN_PERIODS))
        for i, T in enumerate(RETURN_PERIODS):
            q = rp_values[T]
            metric_label = f"Q{T}" if analysis_type == "Maximum Flow" else f"1Q{T}"
            cols[i].metric(metric_label, "NA" if pd.isna(q) else f"{float(q):,.1f}")

        rp_df = pd.DataFrame({
            "Return Period (yr)": RETURN_PERIODS,
            "Design Discharge (m³/s)": [rp_values[T] for T in RETURN_PERIODS],
        })
        if pd.notna(basin_area_km2) and basin_area_km2 > 0:
            rp_df["Specific Discharge (m³/s/km²)"] = rp_df["Design Discharge (m³/s)"] / basin_area_km2

        c_curve, c_table = st.columns([1.35, 1.0])
        with c_curve:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rp_df["Return Period (yr)"], y=rp_df["Design Discharge (m³/s)"],
                mode="lines+markers", name="Selected design curve",
                line=dict(color=accent, width=3),
                hovertemplate="T=%{x} yr<br>Q=%{y:.2f} m³/s<extra></extra>",
            ))
            if not pd.isna(selected_q):
                fig.add_trace(go.Scatter(
                    x=[return_period], y=[selected_q], mode="markers",
                    marker=dict(size=17, symbol="diamond", color="#facc15", line=dict(width=1.5, color="#0b1622")),
                    name=f"Selected T={return_period}",
                ))
            fig.update_layout(
                template=PLOTLY_TEMPLATE, height=460, xaxis_type="log",
                xaxis_title="Return Period (years)", yaxis_title="Design Discharge (m³/s)",
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
        with c_table:
            st.dataframe(rp_df, use_container_width=True, hide_index=True)
            if pd.notna(specific_q):
                st.metric(
                    f"Specific {'flood' if analysis_type == 'Maximum Flow' else 'low-flow'} discharge at T={return_period}",
                    f"{specific_q:.4f} m³/s/km²",
                )
                st.caption("This normalizes discharge by the delineated watershed area for basin-to-basin comparison.")

        if analysis_type == "Minimum Flow":
            st.caption(
                "Here 1Q10 means a 1-day annual-minimum flow with 10-year recurrence/nonexceedance behavior. "
                "Do not interpret it as 7Q10 unless the annual series was first built from 7-day moving-average minima."
            )


# ============================================================
# FIT + BOOTSTRAP UNCERTAINTY
# ============================================================
def render_uncertainty_panel(fit_df, boot_df, choice_df):
    st.markdown("#### Distribution Fit & Bootstrap Uncertainty")
    st.caption(
        "Goodness of fit and return-level uncertainty are reported separately. "
        "The best-fitting distribution can still have a wide bootstrap confidence interval."
    )

    if fit_df is None or fit_df.empty:
        st.warning("No uncertainty result is available yet.")
        return

    f = fit_df.copy()
    b = boot_df.copy() if boot_df is not None else pd.DataFrame()
    c = choice_df.copy() if choice_df is not None else pd.DataFrame()

    if "Return_Period" in b.columns:
        b["Return_Period"] = pd.to_numeric(b["Return_Period"], errors="coerce")
    bt = b[b["Return_Period"] == int(return_period)].copy() if not b.empty and "Return_Period" in b.columns else pd.DataFrame()

    best_fit = "NA"
    if "Fit_Rank" in f.columns and f["Fit_Rank"].notna().any():
        best_fit = str(f.sort_values(["Fit_Rank", "AD"], na_position="last").iloc[0]["Distribution"])
    best_unc = "NA"
    if not bt.empty and "Relative_CI_HalfWidth_pct" in bt.columns and bt["Relative_CI_HalfWidth_pct"].notna().any():
        best_unc = str(bt.sort_values("Relative_CI_HalfWidth_pct").iloc[0]["Distribution"])

    a, bcol, ccol = st.columns(3)
    a.metric("Best observed-data fit", best_fit)
    bcol.metric(f"Lowest uncertainty at T={return_period}", best_unc)
    if best_fit != "NA" and not bt.empty:
        bf = bt[bt["Distribution"].astype(str) == best_fit]
        if not bf.empty and pd.notna(bf.iloc[0].get("Relative_CI_HalfWidth_pct", np.nan)):
            ccol.metric("Best-fit relative 95% CI half-width", f"{bf.iloc[0]['Relative_CI_HalfWidth_pct']:.1f}%")
        else:
            ccol.metric("Best-fit uncertainty", "NA")
    else:
        ccol.metric("Best-fit uncertainty", "NA")

    if best_fit != "NA" and best_unc != "NA" and best_fit != best_unc:
        st.warning(
            f"**Fit–uncertainty trade-off:** {best_fit} has the best observed-data fit, but {best_unc} has the "
            f"narrowest bootstrap uncertainty for the {return_period}-year return level."
        )
    elif best_fit != "NA" and best_fit == best_unc:
        st.success(f"{best_fit} is both the best-fitting and least-uncertain candidate at T={return_period}.")

    st.markdown("##### 1. Goodness of fit — lower is better")
    fit_cols = [
        "Distribution", "AD", "CvM", "KS", "AICc", "Rank_AD", "Rank_CvM", "Rank_KS",
        "Fit_Rank", "Median_Relative_CI_HalfWidth_pct", "Overall_Uncertainty_Rank",
        "Mean_Bootstrap_Failure_pct",
    ]
    fit_cols = [x for x in fit_cols if x in f.columns]
    st.dataframe(f[fit_cols].sort_values([x for x in ["Fit_Rank", "AD"] if x in fit_cols]), use_container_width=True, hide_index=True)
    st.caption("RMSE and MAE are intentionally not used for primary distribution selection in this updated panel.")

    st.markdown(f"##### 2. Sampling uncertainty — T={return_period} years")
    if bt.empty:
        st.info("No bootstrap values are available for the selected return period.")
    else:
        show_cols = [
            "Distribution", "Point_Estimate", "Bootstrap_Median", "Bootstrap_SE",
            "CI95_Lower", "CI95_Upper", "Relative_CI_HalfWidth_pct", "Bootstrap_CV_pct",
            "Bootstrap_Bias", "Bootstrap_Failure_pct", "Fit_Rank", "Uncertainty_Rank",
        ]
        show_cols = [x for x in show_cols if x in bt.columns]
        st.dataframe(bt[show_cols].sort_values([x for x in ["Fit_Rank", "Uncertainty_Rank"] if x in show_cols]), use_container_width=True, hide_index=True)

        bt_plot = bt.dropna(subset=[x for x in ["Point_Estimate", "CI95_Lower", "CI95_Upper"] if x in bt.columns]).copy()
        if not bt_plot.empty:
            fig_ci = go.Figure()
            fig_ci.add_trace(go.Scatter(
                x=bt_plot["Distribution"], y=bt_plot["Point_Estimate"], mode="markers",
                marker=dict(size=12),
                error_y=dict(
                    type="data", symmetric=False,
                    array=bt_plot["CI95_Upper"] - bt_plot["Point_Estimate"],
                    arrayminus=bt_plot["Point_Estimate"] - bt_plot["CI95_Lower"],
                    thickness=1.5, width=5,
                ),
                name="Return level ± 95% bootstrap CI",
                hovertemplate="%{x}<br>Q=%{y:.2f} m³/s<extra></extra>",
            ))
            fig_ci.update_layout(
                template=PLOTLY_TEMPLATE, height=430,
                xaxis_title="Distribution", yaxis_title="Discharge (m³/s)",
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_ci, use_container_width=True)

        if "Fit_Rank_Sum" in bt.columns and "Relative_CI_HalfWidth_pct" in bt.columns:
            pu = bt.dropna(subset=["Fit_Rank_Sum", "Relative_CI_HalfWidth_pct"]).copy()
            if not pu.empty:
                fig_trade = px.scatter(
                    pu, x="Fit_Rank_Sum", y="Relative_CI_HalfWidth_pct", text="Distribution",
                    hover_data=[x for x in ["Point_Estimate", "CI95_Lower", "CI95_Upper", "Bootstrap_Failure_pct"] if x in pu.columns],
                    template=PLOTLY_TEMPLATE,
                    labels={
                        "Fit_Rank_Sum": "GOF rank sum (lower = better fit)",
                        "Relative_CI_HalfWidth_pct": "Relative 95% CI half-width (%) — lower = more precise",
                    },
                )
                fig_trade.update_traces(textposition="top center", marker=dict(size=12))
                fig_trade.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_trade, use_container_width=True)
                st.caption("The desirable region is the lower-left: strong fit with relatively narrow uncertainty.")

    st.markdown("##### 3. Distribution-choice uncertainty")
    if c.empty or "Return_Period" not in c.columns:
        st.info("No across-distribution uncertainty table is available.")
    else:
        c["Return_Period"] = pd.to_numeric(c["Return_Period"], errors="coerce")
        cr = c[c["Return_Period"] == int(return_period)]
        if cr.empty:
            st.info("No distribution-choice uncertainty result for this return period.")
        else:
            r = cr.iloc[0]
            x1, x2, x3 = st.columns(3)
            x1.metric(
                "All-model estimate range",
                f"{r['All_Model_Min']:.2f}–{r['All_Model_Max']:.2f} m³/s"
                if pd.notna(r.get("All_Model_Min")) and pd.notna(r.get("All_Model_Max")) else "NA",
            )
            x2.metric("All-model spread", f"{r['All_Model_Range']:.2f} m³/s" if pd.notna(r.get("All_Model_Range")) else "NA")
            x3.metric("Top-3 fit spread", f"{r['Top3_Model_Range']:.2f} m³/s" if pd.notna(r.get("Top3_Model_Range")) else "NA")
            st.caption(f"Top-three fit candidates: {r.get('Top3_Distributions', 'NA')}")


with tab_uncertainty:
    if not fit_match.empty:
        st.success("Using precomputed uncertainty workbook from results/Nepal_HydroEVT_Uncertainty_v2.xlsx")
        render_uncertainty_panel(fit_match, boot_match, choice_match)
    else:
        st.info(
            "No precomputed uncertainty workbook was found for this station. You can compute bootstrap uncertainty on demand below. "
            "For a deployed app, precomputing all eligible stations is faster."
        )
        if not UNCERTAINTY_ENGINE_AVAILABLE:
            st.error("hydroevt_uncertainty_v2.py / lmoments3 is not available. Add the helper file and requirements.txt.")
        elif annual_station.empty:
            st.warning("Annual series unavailable; bootstrap analysis cannot be run.")
        else:
            n_boot = st.select_slider(
                "Bootstrap resamples",
                options=[200, 500, 1000, 2000, 5000],
                value=DEFAULT_BOOTSTRAPS,
                help="Use 500 for quick testing. Use 2000 or more for final reporting.",
            )
            st.caption(
                "Maximum candidates: GEV, Gumbel, Kappa, Generalized Logistic, Generalized Pareto, Log-Pearson III. "
                "Minimum candidates: Log-Pearson III, Lognormal, Gamma, Weibull, GEV-minimum, Gumbel-minimum."
            )
            run_key = f"unc_{analysis_short}_{station_id}_{n_boot}"
            if st.button("Run bootstrap uncertainty for selected station", type="primary"):
                with st.spinner(f"Running {n_boot} bootstrap resamples across candidate distributions..."):
                    fit_calc, boot_calc, choice_calc = cached_uncertainty(
                        tuple(annual_station[station_id].astype(float).tolist()),
                        analysis_engine,
                        tuple(RETURN_PERIODS),
                        int(n_boot),
                    )
                    for d in [fit_calc, boot_calc, choice_calc]:
                        if not d.empty and "Station" not in d.columns:
                            d.insert(0, "Station", station_id)
                    st.session_state[run_key] = (fit_calc, boot_calc, choice_calc)
            if run_key in st.session_state:
                fit_calc, boot_calc, choice_calc = st.session_state[run_key]
                fit_match, boot_match, choice_match = fit_calc, boot_calc, choice_calc
                render_uncertainty_panel(fit_calc, boot_calc, choice_calc)


# ============================================================
# COMPARE STATIONS
# ============================================================
with tab_compare:
    st.markdown("#### Multi-Station Comparison — Eligible Stations Only")
    st.caption(f"Only stations with ≥ {min_valid_years} observed annual {analysis_type.lower()} values are available.")

    all_station_ids = eligible_stations["Station"].dropna().unique().tolist()
    default_sel = [station_id] if station_id in all_station_ids else all_station_ids[:1]
    compare_ids = st.multiselect("Select stations to compare", options=all_station_ids, default=default_sel, max_selections=8)

    if compare_ids:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("###### Annual series overlay")
            fig_cmp = go.Figure()
            has_any = False
            for sid in compare_ids:
                if sid in annual_source.columns:
                    s = pd.to_numeric(annual_source[sid], errors="coerce")
                    yrs = pd.to_numeric(annual_source["Year"], errors="coerce")
                    m = ~s.isna() & ~yrs.isna()
                    if m.any():
                        has_any = True
                        fig_cmp.add_trace(go.Scatter(x=yrs[m], y=s[m], mode="lines+markers", name=f"Station {sid}"))
            if has_any:
                fig_cmp.update_layout(
                    template=PLOTLY_TEMPLATE, height=440,
                    xaxis_title="Year", yaxis_title="Discharge (m³/s)",
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_cmp, use_container_width=True)
            else:
                st.caption("No annual series available for selected stations.")

        with c2:
            st.markdown(f"###### Basin area vs. Q{return_period if analysis_type == 'Maximum Flow' else 'min'+str(return_period)}")
            rows = []
            for sid in compare_ids:
                dm = design_df[design_df["Station"] == sid] if not design_df.empty and "Station" in design_df.columns else pd.DataFrame()
                smeta = stations[stations["Station"] == sid]
                if dm.empty or smeta.empty:
                    continue
                row = dm.iloc[0].to_dict()
                cands = design_return_map(analysis_type)[return_period]
                qcol = get_col(dm, cands)
                area = smeta.iloc[0].get("Delineated_Area_km2", np.nan)
                qv = row.get(qcol, np.nan) if qcol else np.nan
                rows.append({"Station": sid, "Area_km2": area, "Q": qv, "Specific_Q": qv / area if pd.notna(qv) and pd.notna(area) and area > 0 else np.nan})
            reg = pd.DataFrame(rows).dropna(subset=["Area_km2", "Q"])
            if not reg.empty:
                fig_reg = px.scatter(
                    reg, x="Area_km2", y="Q", text="Station",
                    hover_data=["Specific_Q"], template=PLOTLY_TEMPLATE,
                    labels={"Area_km2": "Delineated basin area (km²)", "Q": "Design discharge (m³/s)", "Specific_Q": "Specific discharge"},
                )
                fig_reg.update_traces(textposition="top center")
                fig_reg.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_reg, use_container_width=True)
            else:
                st.caption("Basin area/design discharge data unavailable for this selection.")

        rows = []
        for sid in compare_ids:
            sm = summary_df[summary_df["Station"] == sid] if not summary_df.empty and "Station" in summary_df.columns else pd.DataFrame()
            m2 = stations[stations["Station"] == sid]
            rq = record_quality[record_quality["Station"] == sid]
            dm = design_df[design_df["Station"] == sid] if not design_df.empty and "Station" in design_df.columns else pd.DataFrame()
            qv = np.nan
            if not dm.empty:
                qc = get_col(dm, design_return_map(analysis_type)[return_period])
                if qc:
                    qv = dm.iloc[0][qc]
            area = m2.iloc[0].get("Delineated_Area_km2", np.nan) if not m2.empty else np.nan
            rows.append({
                "Station": sid,
                "River": m2.iloc[0]["River"] if not m2.empty else "NA",
                "Valid Years": int(rq.iloc[0]["Valid_Years"]) if not rq.empty else np.nan,
                "Completeness %": rq.iloc[0]["Completeness_pct"] if not rq.empty else np.nan,
                "Basin Area (km²)": area,
                f"Q{return_period} (m³/s)" if analysis_type == "Maximum Flow" else f"1Q{return_period} (m³/s)": qv,
                "Specific Q (m³/s/km²)": qv / area if pd.notna(qv) and pd.notna(area) and area > 0 else np.nan,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Select at least one station above to compare.")


# ============================================================
# BASIN + RECORD QUALITY
# ============================================================
with tab_basin_quality:
    st.markdown("#### Basin & Record-Sufficiency Diagnostics")
    q_all = record_quality.merge(
        stations[[c for c in ["Station", "River", "Location", "Delineated_Area_km2"] if c in stations.columns]],
        on="Station", how="left",
    )
    q_all["Eligible"] = q_all["Valid_Years"] >= min_valid_years

    a, b, c, d = st.columns(4)
    a.metric("Stations in annual sheet", len(q_all))
    b.metric("Eligible", int(q_all["Eligible"].sum()))
    c.metric("Excluded", int((~q_all["Eligible"]).sum()))
    d.metric("Median valid years", f"{q_all['Valid_Years'].median():.0f}")

    c_hist, c_scatter = st.columns(2)
    with c_hist:
        fig_hist = px.histogram(
            q_all, x="Valid_Years", nbins=20, template=PLOTLY_TEMPLATE,
            labels={"Valid_Years": "Observed annual extremes"},
        )
        fig_hist.add_vline(x=min_valid_years, line_dash="dash", annotation_text=f"Eligibility = {min_valid_years} yr")
        fig_hist.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_hist, use_container_width=True)
    with c_scatter:
        qa = q_all.dropna(subset=["Delineated_Area_km2"]).copy()
        if not qa.empty:
            fig_q = px.scatter(
                qa, x="Delineated_Area_km2", y="Valid_Years", color="Eligible",
                hover_name="Station", hover_data=[x for x in ["River", "Location", "Completeness_pct", "Longest_Gap_Years"] if x in qa.columns],
                log_x=True, template=PLOTLY_TEMPLATE,
                labels={"Delineated_Area_km2": "Delineated basin area (km²)", "Valid_Years": "Valid annual extremes"},
            )
            fig_q.add_hline(y=min_valid_years, line_dash="dash")
            fig_q.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_q, use_container_width=True)

    st.markdown("##### Station screening table")
    show_cols = [
        "Station", "River", "Location", "Delineated_Area_km2", "Valid_Years", "First_Year", "Last_Year",
        "Span_Years", "Missing_Years", "Completeness_pct", "Longest_Gap_Years", "Record_Class", "Eligible",
    ]
    show_cols = [x for x in show_cols if x in q_all.columns]
    st.dataframe(q_all[show_cols].sort_values(["Eligible", "Valid_Years"], ascending=[False, False]), use_container_width=True, hide_index=True)

    excluded = q_all[~q_all["Eligible"]].copy()
    if not excluded.empty:
        with st.expander(f"🚫 Excluded stations (< {min_valid_years} valid years)"):
            st.dataframe(excluded[show_cols].sort_values("Valid_Years"), use_container_width=True, hide_index=True)


# ============================================================
# WATERSHED MAP
# ============================================================
with tab_map:
    st.markdown("#### DHM Stations & Delineated Watersheds")

    lat = float(meta["Latitude"]) if pd.notna(meta.get("Latitude")) else 28.0
    lon = float(meta["Longitude"]) if pd.notna(meta.get("Longitude")) else 84.0
    m = folium.Map(location=[lat, lon], zoom_start=8, tiles="OpenStreetMap", control_scale=True)

    Fullscreen(position="topright", title="Full Screen", title_cancel="Exit Full Screen", force_separate_button=True).add_to(m)
    folium.TileLayer(tiles="CartoDB positron", name="Light Map", control=True).add_to(m)
    folium.TileLayer(tiles="CartoDB dark_matter", name="Dark Map", control=True).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="Map data © OpenStreetMap contributors, SRTM | Map style © OpenTopoMap",
        name="Terrain", control=True,
    ).add_to(m)

    # Watershed polygons. Green = eligible, red = excluded at active threshold.
    if isinstance(basins, gpd.GeoDataFrame) and not basins.empty:
        basin_map = basins.merge(record_quality[["Station", "Valid_Years", "Eligible"]], on="Station", how="left")
        basin_map["Eligible"] = basin_map["Eligible"].fillna(False)
        basin_map["Valid_Years"] = basin_map["Valid_Years"].fillna(0)
        basin_map = basin_map.copy()
        basin_map["geometry"] = basin_map.geometry.simplify(0.0015, preserve_topology=True)

        if show_all_basins:
            geojson = json.loads(basin_map.to_json())
            folium.GeoJson(
                geojson,
                name="All watershed boundaries",
                style_function=lambda feat: {
                    "color": "#34d399" if feat["properties"].get("Eligible") else "#f87171",
                    "weight": 1.0,
                    "fillColor": "#34d399" if feat["properties"].get("Eligible") else "#f87171",
                    "fillOpacity": 0.035,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["Station", "area_km2", "Valid_Years", "Eligible"],
                    aliases=["Station", "Basin area (km²)", "Valid annual extremes", "Eligible"],
                    localize=True,
                ),
            ).add_to(m)

        selected_basin = basin_map[basin_map["Station"] == station_id]
        if not selected_basin.empty:
            folium.GeoJson(
                json.loads(selected_basin.to_json()),
                name=f"Selected watershed {station_id}",
                style_function=lambda feat: {
                    "color": "#facc15", "weight": 3.5, "fillColor": "#f59e0b", "fillOpacity": 0.16,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["Station", "area_km2", "Valid_Years"],
                    aliases=["Station", "Delineated area (km²)", "Valid annual extremes"],
                    localize=True,
                ),
            ).add_to(m)
            minx, miny, maxx, maxy = selected_basin.total_bounds
            if np.all(np.isfinite([minx, miny, maxx, maxy])):
                m.fit_bounds([[miny, minx], [maxy, maxx]], padding=(20, 20))

    # Station markers; by default only eligible stations are shown.
    map_df = eligible_stations.copy() if show_all_stations else eligible_stations[eligible_stations["Station"] == station_id].copy()
    cluster = MarkerCluster(name="Eligible DHM stations").add_to(m)
    rq_lookup = record_quality.set_index("Station") if not record_quality.empty else pd.DataFrame()

    for _, row in map_df.iterrows():
        if pd.isna(row.get("Latitude")) or pd.isna(row.get("Longitude")):
            continue
        sid = str(row["Station"])
        is_selected = sid == str(station_id)
        rr = rq_lookup.loc[sid] if isinstance(rq_lookup, pd.DataFrame) and sid in rq_lookup.index else None
        valid_years = int(rr["Valid_Years"]) if rr is not None else "NA"
        area = row.get("Delineated_Area_km2", np.nan)
        popup_html = f"""
        <div style="width:280px;font-family:Arial;">
          <h4 style="margin-bottom:8px;color:#0f3557;">DHM Station {sid}</h4>
          <b>River:</b> {row.get('River', 'NA')}<br>
          <b>Location:</b> {row.get('Location', 'NA')}<br>
          <b>Valid annual extremes:</b> {valid_years}<br>
          <b>Delineated basin:</b> {'NA' if pd.isna(area) else f'{float(area):,.1f} km²'}<br>
          <b>Latitude:</b> {row.get('Latitude', 'NA')}<br>
          <b>Longitude:</b> {row.get('Longitude', 'NA')}<br>
        </div>
        """
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=9 if is_selected else 5,
            color="red" if is_selected else "blue",
            weight=2, fill=True,
            fill_color="red" if is_selected else "blue",
            fill_opacity=0.85,
            tooltip=f"Station {sid} | {row.get('River', '')} | n={valid_years}",
            popup=folium.Popup(popup_html, max_width=340),
        ).add_to(cluster)

    folium.Marker(
        location=[lat, lon],
        tooltip=f"Selected Station {station_id}",
        popup=f"<b>Selected Station {station_id}</b><br>River: {meta.get('River', 'NA')}<br>Basin area: {safe_number(basin_area_km2, 1)} km²",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width=None, height=680, use_container_width=True)
    st.caption(
        f"🟢 watershed = meets ≥{min_valid_years}-year threshold · 🔴 watershed = excluded · "
        "🟡 selected watershed · station selector itself contains eligible stations only."
    )


# ============================================================
# EXPORT
# ============================================================
with tab_download:
    st.markdown("#### Export Selected Station Results")

    annual_export = annual_station.rename(columns={station_id: "Discharge"}) if not annual_station.empty else pd.DataFrame()
    basin_info = {
        "Station": station_id,
        "Delineated_Area_km2": basin_area_km2,
        "Basin_File": os.path.basename(BASIN_FILE) if BASIN_FILE else "NA",
        "CRS": "EPSG:4326" if isinstance(basins, gpd.GeoDataFrame) and not basins.empty else "NA",
    }

    export_bytes = build_download_excel(
        station_meta=meta,
        summary_row=summary_row,
        design_row=design_row,
        annual_df=annual_export,
        record_quality_row=rq_row,
        basin_info=basin_info,
        fit_df=fit_match,
        boot_df=boot_match,
        model_choice_df=choice_match,
    )

    filename = f"Station_{station_id}_{analysis_short}_HydroEVT_DSS.xlsx"
    st.download_button(
        label="⬇️ Download selected station analysis",
        data=export_bytes,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.markdown("#### Data Quality Check")
    qc = {
        "Station metadata available": True,
        "Annual series available": not annual_station.empty,
        f"Meets ≥{min_valid_years}-year screening threshold": int(rq_row.get("Valid_Years", 0)) >= min_valid_years,
        "Delineated watershed available": not basin_match.empty,
        "HydroEVT summary available": summary_row is not None,
        "Design-flow results available": design_row is not None,
        "Bootstrap uncertainty available": fit_match is not None and not fit_match.empty,
    }
    st.dataframe(pd.DataFrame({"Check": qc.keys(), "Available": qc.values()}), use_container_width=True, hide_index=True)


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "Nepal Hydrological Extreme Flow DSS | DHM station metadata + annual extreme-flow records + "
    "delineated station watersheds + HydroEVT frequency analysis + bootstrap/model-choice uncertainty."
)
