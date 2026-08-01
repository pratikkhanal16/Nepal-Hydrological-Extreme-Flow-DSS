import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px 
import pydeck as pdk
from scipy import stats as sstats

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Nepal Hydrological Extreme Flow DSS | Pratik & Saugat",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
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

# ============================================================
# THEME / CSS  (dark analytical dashboard)
# ============================================================
PLOTLY_TEMPLATE = "plotly_dark"

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
            color: #6fdc9a;
            margin-top: 0.15rem;
        }

        .kpi-sub.bad { color: #f87171; }
        .kpi-sub.neutral { color: #93a5b8; }

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
        .badge-gray { background: rgba(148,163,184,0.15); color: #94a3b8; }

        .section-card {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(148,197,255,0.10);
            border-radius: 14px;
            padding: 1.1rem 1.2rem;
        }

        .small-note { font-size: 0.82rem; color: #7d92a8; }

        div[data-testid="stMetricValue"] { color: #f1f8ff; }

        [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True
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
        unsafe_allow_html=True
    )


def badge(text, kind="blue"):
    return f'<span class="badge badge-{kind}">{text}</span>'


# ============================================================
# HELPERS
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
    mapping = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in mapping:
            return mapping[candidate.lower()]
    return None


def build_download_excel(station_meta, summary_row, design_row, ranking_df, annual_df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame([station_meta]).to_excel(writer, sheet_name="Station_Info", index=False)
        if summary_row is not None:
            pd.DataFrame([summary_row]).to_excel(writer, sheet_name="Summary", index=False)
        if design_row is not None:
            pd.DataFrame([design_row]).to_excel(writer, sheet_name="Design_Flows", index=False)
        if ranking_df is not None and not ranking_df.empty:
            ranking_df.to_excel(writer, sheet_name="Distribution_Ranking", index=False)
        if annual_df is not None and not annual_df.empty:
            annual_df.to_excel(writer, sheet_name="Annual_Series", index=False)
    buffer.seek(0)
    return buffer.getvalue()


def fit_overlay_curve(values, dist_name):
    """Fit a scipy distribution to annual extremes and return x, pdf-scaled-to-hist y."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 5:
        return None, None

    dist_map = {
        "Gumbel": sstats.gumbel_r,
        "GEV": sstats.genextreme,
        "LogNormal": sstats.lognorm,
        "Log-Normal": sstats.lognorm,
        "Pearson3": sstats.pearson3,
        "Log Pearson III": sstats.pearson3,
        "LogPearson3": sstats.pearson3,
        "Normal": sstats.norm,
        "Weibull": sstats.weibull_min,
        "Exponential": sstats.expon,
    }

    dist = None
    for key, d in dist_map.items():
        if key.lower() in str(dist_name).lower():
            dist = d
            break

    if dist is None:
        dist = sstats.gumbel_r

    try:
        params = dist.fit(values)
        x = np.linspace(values.min() * 0.85, values.max() * 1.15, 200)
        y = dist.pdf(x, *params)
        return x, y
    except Exception:
        return None, None


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

    max_df, min_df = pd.DataFrame(), pd.DataFrame()

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
missing = [p for p in [STATION_CSV, HYDROEVT_FILE, ANNUAL_EXCEL] if not os.path.exists(p)]
if missing:
    st.error("Required file(s) not found:")
    for p in missing:
        st.code(p)
    st.stop()

stations = load_station_metadata(STATION_CSV)
evt, evt_sheets = load_hydroevt(HYDROEVT_FILE)
max_raw, min_raw = load_annual_data(ANNUAL_EXCEL)

# ============================================================
# TITLE
# ============================================================
st.markdown('<div class="main-title">🌊 Nepal Hydrological Extreme Flow DSS</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Flood-frequency & low-flow decision-support system · '
    f'{len(stations)} DHM stations · trend, frequency, and distribution intelligence</div>',
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ DSS Controls")

river_values = stations["River"].dropna().astype(str).sort_values().unique().tolist()
selected_river = st.sidebar.selectbox("River", ["All Rivers"] + river_values)

station_subset = stations.copy() if selected_river == "All Rivers" else stations[
    stations["River"].astype(str) == selected_river
].copy()
station_subset = station_subset.sort_values(["River", "Station"])

station_options = {}
for _, row in station_subset.iterrows():
    sid = normalize_station_id(row["Station"])
    label = f"{sid} | {row.get('River', 'NA')} | {row.get('Location', 'NA')}"
    station_options[label] = sid

if not station_options:
    st.warning("No stations available for this river selection.")
    st.stop()

station_label = st.sidebar.selectbox("Station", list(station_options.keys()))
station_id = station_options[station_label]

analysis_type = st.sidebar.radio("Analysis", ["Maximum Flow", "Minimum Flow"], horizontal=False)

return_period = st.sidebar.selectbox("Return Period (years)", [2, 5, 10, 25, 50, 100, 200], index=5)

show_all_stations = st.sidebar.checkbox("Show all stations on map", value=True)
map_style_3d = st.sidebar.checkbox("3D elevation view on map", value=True)

st.sidebar.divider()
st.sidebar.caption(
    "Station IDs are normalized automatically, so 420, 420.0 and '420' are treated as the same station."
)

with st.sidebar.expander("📐 Basin-wide snapshot"):
    st.metric("Total stations", len(stations))
    st.metric("Rivers covered", stations["River"].nunique())
    if "Drainage_Area_km2" in stations.columns:
        st.metric("Median drainage area", f"{stations['Drainage_Area_km2'].median():,.0f} km²")

# ============================================================
# ACTIVE TABLES
# ============================================================
if analysis_type == "Maximum Flow":
    summary_df, design_df, ranking_df, all_models_df = (
        evt["max_summary"], evt["max_design"], evt["max_rank"], evt["max_all"]
    )
    annual_source = max_raw
    analysis_short = "MAX"
    accent = "#38bdf8"
else:
    summary_df, design_df, ranking_df, all_models_df = (
        evt["min_summary"], evt["min_design"], evt["min_rank"], evt["min_all"]
    )
    annual_source = min_raw
    analysis_short = "MIN"
    accent = "#f59e0b"

# ============================================================
# STATION META
# ============================================================
meta_match = stations[stations["Station"] == station_id].copy()
if meta_match.empty:
    st.error(f"Station metadata not found for station {station_id}.")
    st.stop()
meta = meta_match.iloc[0].to_dict()

# ============================================================
# HYDROEVT ROWS
# ============================================================
def match_station(df):
    if df is None or df.empty or "Station" not in df.columns:
        return pd.DataFrame()
    return df[df["Station"] == station_id].copy()

summary_match = match_station(summary_df)
design_match = match_station(design_df)
rank_match = match_station(ranking_df)
all_model_match = match_station(all_models_df)

summary_row = summary_match.iloc[0].to_dict() if not summary_match.empty else None
design_row = design_match.iloc[0].to_dict() if not design_match.empty else None

# ============================================================
# KPI ROW
# ============================================================
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    kpi_card("Station", station_id, str(meta.get("River", "NA")), "neutral")

with k2:
    drainage = meta.get("Drainage_Area_km2", np.nan)
    kpi_card("Drainage Area", "NA" if pd.isna(drainage) else f"{float(drainage):,.0f} km²", "catchment size", "neutral")

with k3:
    elev = meta.get("Elevation_m", np.nan)
    kpi_card("Elevation", "NA" if pd.isna(elev) else f"{float(elev):,.0f} m", str(meta.get("Location", "NA")), "neutral")

with k4:
    if summary_row is not None:
        n_col = get_col(summary_match, ["N"])
        mean_col = get_col(summary_match, ["Mean"])
        val = f"{float(summary_row[mean_col]):,.1f} m³/s" if mean_col else "NA"
        sub = f"n = {summary_row[n_col]} yrs" if n_col else ""
        kpi_card(f"Mean {analysis_short}", val, sub, "neutral")
    else:
        kpi_card(f"Mean {analysis_short}", "NA", "no HydroEVT match", "bad")

with k5:
    if summary_row is not None:
        trend_col = get_col(summary_match, ["MK_Trend"])
        p_col = get_col(summary_match, ["MK_p_value"])
        trend_val = str(summary_row[trend_col]) if trend_col else "NA"
        sig = ""
        cls = "neutral"
        if p_col:
            pv = summary_row[p_col]
            if pd.notna(pv):
                sig = f"p = {float(pv):.3f} · {'significant' if float(pv) < 0.05 else 'not significant'}"
                cls = "bad" if ("increas" in trend_val.lower() and float(pv) < 0.05 and analysis_type == "Maximum Flow") else (
                    "good" if float(pv) < 0.05 else "neutral"
                )
        kpi_card("Mann–Kendall Trend", trend_val, sig, cls if cls in ("bad", "neutral") else "neutral")
    else:
        kpi_card("Mann–Kendall Trend", "NA", "", "neutral")

st.write("")

# ============================================================
# TABS
# ============================================================
tab_overview, tab_frequency, tab_models, tab_compare, tab_map, tab_download = st.tabs(
    [
        "📊 Overview",
        "📈 Frequency Analysis",
        "🏆 Distribution Models",
        "🔀 Compare Stations",
        "🗺️ Station Map",
        "⬇️ Export"
    ]
)

# ============================================================
# OVERVIEW TAB
# ============================================================
with tab_overview:

    left, right = st.columns([1.0, 1.4])

    with left:
        st.markdown("#### Station Information")
        station_info = pd.DataFrame({
            "Attribute": ["Station", "River", "Location", "Latitude", "Longitude",
                          "Elevation (m)", "Drainage Area (km²)", "Published From", "Published To"],
            "Value": [
                station_id, meta.get("River", "NA"), meta.get("Location", "NA"),
                safe_number(meta.get("Latitude"), 5), safe_number(meta.get("Longitude"), 5),
                safe_number(meta.get("Elevation_m"), 0), safe_number(meta.get("Drainage_Area_km2"), 1),
                meta.get("Published_From", "NA"), meta.get("Published_To", "NA"),
            ]
        })
        st.dataframe(station_info, use_container_width=True, hide_index=True)

        st.markdown("#### HydroEVT Summary")
        if summary_row is None:
            st.warning(f"No {analysis_type.lower()} HydroEVT result found for station {station_id}.")
        else:
            best_col = get_col(summary_match, ["Best_Distribution"])
            slope_col = get_col(summary_match, ["Sen_Slope"])
            start_col = get_col(summary_match, ["Start_Year"])
            end_col = get_col(summary_match, ["End_Year"])

            if best_col:
                st.markdown(
                    f"Best-fit distribution: {badge(str(summary_row[best_col]), 'blue')}",
                    unsafe_allow_html=True
                )
            if start_col and end_col:
                st.caption(f"Record used: {int(summary_row[start_col])}–{int(summary_row[end_col])}")
            if slope_col and pd.notna(summary_row[slope_col]):
                direction = "rising" if float(summary_row[slope_col]) > 0 else "falling"
                st.caption(f"Sen's slope: {float(summary_row[slope_col]):.4f} m³/s/yr ({direction})")

    with right:
        st.markdown(f"#### Annual {analysis_type} Series")

        annual_station = pd.DataFrame()
        if annual_source is not None and not annual_source.empty and station_id in annual_source.columns:
            annual_station = annual_source[["Year", station_id]].copy()
            annual_station["Year"] = pd.to_numeric(annual_station["Year"], errors="coerce")
            annual_station[station_id] = pd.to_numeric(annual_station[station_id], errors="coerce")
            annual_station = annual_station.dropna().sort_values("Year")

            vals = annual_station[station_id].values
            years = annual_station["Year"].values

            # rolling mean for a "smart" analytical feel
            roll = pd.Series(vals).rolling(window=5, min_periods=2, center=True).mean()

            # flag top-3 extreme years
            top_idx = np.argsort(vals)[-3:] if analysis_type == "Maximum Flow" else np.argsort(vals)[:3]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=years, y=vals, name="Annual value",
                marker_color=accent, opacity=0.55,
                hovertemplate="Year=%{x}<br>Discharge=%{y:.2f} m³/s<extra></extra>"
            ))
            fig.add_trace(go.Scatter(
                x=years, y=roll, name="5-yr rolling mean",
                mode="lines", line=dict(color="#f1f8ff", width=2.5)
            ))
            if len(years) >= 2:
                z = np.polyfit(years, vals, 1)
                fig.add_trace(go.Scatter(
                    x=years, y=np.polyval(z, years), name="Linear trend",
                    mode="lines", line=dict(color="#f87171", width=2, dash="dash")
                ))
            fig.add_trace(go.Scatter(
                x=years[top_idx], y=vals[top_idx], mode="markers", name="Extreme years",
                marker=dict(size=13, color="#facc15", symbol="diamond", line=dict(width=1, color="#0b1622"))
            ))

            fig.update_layout(
                template=PLOTLY_TEMPLATE, height=460,
                xaxis_title="Year", yaxis_title="Discharge (m³/s)",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📌 Extreme-year detail"):
                extreme_df = annual_station.iloc[top_idx].sort_values(station_id, ascending=False)
                extreme_df = extreme_df.rename(columns={station_id: "Discharge (m³/s)"})
                st.dataframe(extreme_df, use_container_width=True, hide_index=True)
        else:
            st.warning(f"Station {station_id} was not found in the {analysis_type.lower()} annual-series sheet.")

# ============================================================
# FREQUENCY TAB
# ============================================================
with tab_frequency:

    st.markdown(f"#### {analysis_type} – Return Period Results")

    if design_row is None:
        st.warning("No design-flow table is available for this station.")
    else:
        if analysis_type == "Maximum Flow":
            return_map = {T: [f"Q{T}"] for T in [2, 5, 10, 25, 50, 100, 200]}
        else:
            return_map = {T: [f"Qmin{T}", f"Q{T}"] for T in [2, 5, 10, 25, 50, 100, 200]}

        rp_values = {}
        for T, candidates in return_map.items():
            col = get_col(design_match, candidates)
            rp_values[T] = design_row[col] if col is not None else np.nan

        selected_q = rp_values.get(return_period, np.nan)

        st.info(
            f"Selected return period: **{return_period} years**  |  Design discharge: "
            f"**{'NA' if pd.isna(selected_q) else f'{float(selected_q):,.2f} m³/s'}**"
        )

        cols = st.columns(7)
        for i, T in enumerate([2, 5, 10, 25, 50, 100, 200]):
            q = rp_values[T]
            cols[i].metric(f"{'Q' if analysis_type == 'Maximum Flow' else 'Qmin'}{T}",
                            "NA" if pd.isna(q) else f"{float(q):,.1f}")

        rp_df = pd.DataFrame({"Return Period": list(rp_values.keys()), "Design Discharge": list(rp_values.values())})

        c_curve, c_hist = st.columns([1.3, 1.0])

        with c_curve:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rp_df["Return Period"], y=rp_df["Design Discharge"],
                mode="lines+markers", name="Best-fit design flow",
                line=dict(color=accent, width=3),
                fill="tozeroy", fillcolor="rgba(56,189,248,0.08)",
                hovertemplate="T=%{x} yr<br>Q=%{y:.2f} m³/s<extra></extra>"
            ))
            if not pd.isna(selected_q):
                fig.add_trace(go.Scatter(
                    x=[return_period], y=[selected_q], mode="markers",
                    marker=dict(size=17, symbol="diamond", color="#facc15",
                                line=dict(width=1.5, color="#0b1622")),
                    name=f"Selected T={return_period}"
                ))
            fig.update_layout(
                template=PLOTLY_TEMPLATE, height=460, xaxis_type="log",
                xaxis_title="Return Period (years)", yaxis_title="Design Discharge (m³/s)",
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(rp_df, use_container_width=True, hide_index=True)

        with c_hist:
            st.markdown("###### Distribution fit vs. observed annual values")
            if annual_source is not None and not annual_source.empty and station_id in annual_source.columns:
                vals = pd.to_numeric(annual_source[station_id], errors="coerce").dropna().values
                best_col = get_col(summary_match, ["Best_Distribution"]) if summary_row is not None else None
                dist_label = summary_row[best_col] if best_col else "Gumbel"

                fig_h = go.Figure()
                fig_h.add_trace(go.Histogram(
                    x=vals, histnorm="probability density", name="Observed",
                    marker_color=accent, opacity=0.55, nbinsx=max(6, len(vals) // 3)
                ))
                x_fit, y_fit = fit_overlay_curve(vals, dist_label)
                if x_fit is not None:
                    fig_h.add_trace(go.Scatter(
                        x=x_fit, y=y_fit, mode="lines", name=f"{dist_label} fit",
                        line=dict(color="#facc15", width=2.5)
                    ))
                fig_h.update_layout(
                    template=PLOTLY_TEMPLATE, height=460,
                    xaxis_title="Discharge (m³/s)", yaxis_title="Density",
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_h, use_container_width=True)
            else:
                st.caption("No annual series available to overlay a fitted distribution.")

    if not all_model_match.empty:
        st.markdown("#### Return Levels — All Candidate Distributions")
        dist_col = get_col(all_model_match, ["Distribution"])
        rp_col = get_col(all_model_match, ["Return_Period", "Return Period"])
        q_col = get_col(all_model_match, ["Discharge", "Design Flood"])

        if dist_col and rp_col and q_col:
            plot_df = all_model_match[[dist_col, rp_col, q_col]].copy()
            plot_df[rp_col] = pd.to_numeric(plot_df[rp_col], errors="coerce")
            plot_df[q_col] = pd.to_numeric(plot_df[q_col], errors="coerce")
            plot_df = plot_df.dropna()

            fig2 = px.line(
                plot_df, x=rp_col, y=q_col, color=dist_col, markers=True, log_x=True,
                labels={rp_col: "Return Period (years)", q_col: "Discharge (m³/s)", dist_col: "Distribution"},
                template=PLOTLY_TEMPLATE
            )
            fig2.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10),
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# MODELS TAB
# ============================================================
with tab_models:

    st.markdown("#### Probability Distribution Ranking")

    if rank_match.empty:
        st.warning("No distribution ranking is available for this station.")
    else:
        total_rank_col = get_col(rank_match, ["Total_Rank"])
        distribution_col = get_col(rank_match, ["Distribution"])
        rmse_col = get_col(rank_match, ["RMSE"])
        mae_col = get_col(rank_match, ["MAE"])
        ks_col = get_col(rank_match, ["KS"])
        ksp_col = get_col(rank_match, ["KS_p_value"])

        c_bar, c_radar = st.columns([1.1, 1.0])

        if total_rank_col and distribution_col:
            rank_plot = rank_match.copy()
            rank_plot[total_rank_col] = pd.to_numeric(rank_plot[total_rank_col], errors="coerce")
            rank_plot = rank_plot.sort_values(total_rank_col)

            with c_bar:
                fig = px.bar(
                    rank_plot, x=distribution_col, y=total_rank_col,
                    color=total_rank_col, color_continuous_scale="Blues_r",
                    hover_data=[c for c in [rmse_col, mae_col, ks_col, ksp_col] if c],
                    template=PLOTLY_TEMPLATE
                )
                fig.update_layout(
                    height=440, xaxis_title="Distribution", yaxis_title="Total Rank (lower = better fit)",
                    margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)

            with c_radar:
                metric_cols = [c for c in [rmse_col, mae_col, ks_col] if c]
                if metric_cols and len(rank_plot) >= 1:
                    norm_df = rank_plot[[distribution_col] + metric_cols].copy()
                    for c in metric_cols:
                        norm_df[c] = pd.to_numeric(norm_df[c], errors="coerce")
                        rng = norm_df[c].max() - norm_df[c].min()
                        norm_df[c] = 1 - ((norm_df[c] - norm_df[c].min()) / rng) if rng > 0 else 1.0

                    fig_radar = go.Figure()
                    top_n = norm_df.head(4)
                    for _, r in top_n.iterrows():
                        fig_radar.add_trace(go.Scatterpolar(
                            r=[r[c] for c in metric_cols] + [r[metric_cols[0]]],
                            theta=metric_cols + [metric_cols[0]],
                            fill="toself", name=str(r[distribution_col]), opacity=0.55
                        ))
                    fig_radar.update_layout(
                        template=PLOTLY_TEMPLATE, height=440,
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        margin=dict(l=20, r=20, t=30, b=10),
                        title="Goodness-of-fit profile (higher = better, normalized)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

        st.dataframe(rank_match, use_container_width=True, hide_index=True)

# ============================================================
# COMPARE STATIONS TAB
# ============================================================
with tab_compare:

    st.markdown("#### Multi-Station Comparison")
    st.caption("Compare annual series and design-flow curves across any set of stations.")

    all_station_ids = stations["Station"].dropna().unique().tolist()
    default_sel = [station_id] if station_id in all_station_ids else all_station_ids[:1]

    compare_ids = st.multiselect(
        "Select stations to compare",
        options=all_station_ids,
        default=default_sel,
        max_selections=8
    )

    if compare_ids:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("###### Annual series overlay")
            fig_cmp = go.Figure()
            has_any = False
            if annual_source is not None and not annual_source.empty:
                for sid in compare_ids:
                    if sid in annual_source.columns:
                        s = pd.to_numeric(annual_source[sid], errors="coerce")
                        yrs = pd.to_numeric(annual_source["Year"], errors="coerce")
                        m = ~s.isna() & ~yrs.isna()
                        if m.any():
                            has_any = True
                            fig_cmp.add_trace(go.Scatter(
                                x=yrs[m], y=s[m], mode="lines+markers", name=f"Station {sid}"
                            ))
            if has_any:
                fig_cmp.update_layout(
                    template=PLOTLY_TEMPLATE, height=440,
                    xaxis_title="Year", yaxis_title="Discharge (m³/s)",
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_cmp, use_container_width=True)
            else:
                st.caption("None of the selected stations have annual-series data for this analysis type.")

        with c2:
            st.markdown("###### Design-flow curves (Q vs. Return Period)")
            if analysis_type == "Maximum Flow":
                rmap = {T: [f"Q{T}"] for T in [2, 5, 10, 25, 50, 100, 200]}
            else:
                rmap = {T: [f"Qmin{T}", f"Q{T}"] for T in [2, 5, 10, 25, 50, 100, 200]}

            fig_design = go.Figure()
            has_design = False
            for sid in compare_ids:
                dm = design_df[design_df["Station"] == sid] if not design_df.empty and "Station" in design_df.columns else pd.DataFrame()
                if dm.empty:
                    continue
                row = dm.iloc[0].to_dict()
                ys = []
                xs = []
                for T, cands in rmap.items():
                    col = get_col(dm, cands)
                    if col is not None and pd.notna(row[col]):
                        xs.append(T)
                        ys.append(row[col])
                if xs:
                    has_design = True
                    fig_design.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=f"Station {sid}"))
            if has_design:
                fig_design.update_layout(
                    template=PLOTLY_TEMPLATE, height=440, xaxis_type="log",
                    xaxis_title="Return Period (years)", yaxis_title="Design Discharge (m³/s)",
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_design, use_container_width=True)
            else:
                st.caption("None of the selected stations have design-flow results for this analysis type.")

        st.markdown("###### Side-by-side summary")
        rows = []
        for sid in compare_ids:
            sm = summary_df[summary_df["Station"] == sid] if not summary_df.empty and "Station" in summary_df.columns else pd.DataFrame()
            m2 = stations[stations["Station"] == sid]
            river_v = m2.iloc[0]["River"] if not m2.empty else "NA"
            if not sm.empty:
                row = sm.iloc[0].to_dict()
                mean_col = get_col(sm, ["Mean"])
                best_col = get_col(sm, ["Best_Distribution"])
                trend_col = get_col(sm, ["MK_Trend"])
                rows.append({
                    "Station": sid, "River": river_v,
                    "Mean": row.get(mean_col, np.nan) if mean_col else np.nan,
                    "Best Distribution": row.get(best_col, "NA") if best_col else "NA",
                    "MK Trend": row.get(trend_col, "NA") if trend_col else "NA",
                })
            else:
                rows.append({"Station": sid, "River": river_v, "Mean": np.nan, "Best Distribution": "NA", "MK Trend": "NA"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Select at least one station above to compare.")

# ============================================================
# MAP TAB — OPENSTREETMAP VERSION
# ============================================================

import folium
from folium.plugins import Fullscreen, MarkerCluster


with tab_map:

    st.markdown("#### DHM Streamflow Station Map")

    # --------------------------------------------------------
    # Select map data
    # --------------------------------------------------------

    map_df = stations.copy()

    if not show_all_stations:
        map_df = map_df[
            map_df["Station"] == station_id
        ].copy()

    # --------------------------------------------------------
    # Selected station coordinates
    # --------------------------------------------------------

    lat = (
        float(meta["Latitude"])
        if pd.notna(meta.get("Latitude"))
        else 28.0
    )

    lon = (
        float(meta["Longitude"])
        if pd.notna(meta.get("Longitude"))
        else 84.0
    )

    # --------------------------------------------------------
    # Create OpenStreetMap
    # --------------------------------------------------------

    m = folium.Map(
        location=[lat, lon],
        zoom_start=8,
        tiles="OpenStreetMap",
        control_scale=True
    )

    # Full screen button
    Fullscreen(
        position="topright",
        title="Full Screen",
        title_cancel="Exit Full Screen",
        force_separate_button=True
    ).add_to(m)

    # --------------------------------------------------------
    # Optional alternative basemaps
    # --------------------------------------------------------

    folium.TileLayer(
        tiles="CartoDB positron",
        name="Light Map",
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles="CartoDB dark_matter",
        name="Dark Map",
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles=(
            "https://{s}.tile.opentopomap.org/"
            "{z}/{x}/{y}.png"
        ),
        attr=(
            "Map data © OpenStreetMap contributors, "
            "SRTM | Map style © OpenTopoMap"
        ),
        name="Terrain",
        control=True
    ).add_to(m)

    # --------------------------------------------------------
    # Marker cluster for all stations
    # --------------------------------------------------------

    cluster = MarkerCluster(
        name="DHM Stations"
    ).add_to(m)

    # --------------------------------------------------------
    # Add station markers
    # --------------------------------------------------------

    for _, row in map_df.iterrows():

        sid = str(row["Station"])

        is_selected = (
            sid == str(station_id)
        )

        marker_color = (
            "red"
            if is_selected
            else "blue"
        )

        radius = (
            9
            if is_selected
            else 5
        )

        popup_html = f"""
        <div style="
            width:260px;
            font-family:Arial;
        ">

        <h4 style="
            margin-bottom:8px;
            color:#0f3557;
        ">
            DHM Station {sid}
        </h4>

        <b>River:</b>
        {row.get('River', 'NA')}<br>

        <b>Location:</b>
        {row.get('Location', 'NA')}<br>

        <b>Latitude:</b>
        {row.get('Latitude', 'NA')}<br>

        <b>Longitude:</b>
        {row.get('Longitude', 'NA')}<br>

        <b>Elevation:</b>
        {row.get('Elevation_m', 'NA')} m<br>

        <b>Drainage Area:</b>
        {row.get('Drainage_Area_km2', 'NA')} km²<br>

        <b>Record:</b>
        {row.get('Published_From', 'NA')}
        –
        {row.get('Published_To', 'NA')}

        </div>
        """

        folium.CircleMarker(

            location=[
                row["Latitude"],
                row["Longitude"]
            ],

            radius=radius,

            color=marker_color,

            weight=2,

            fill=True,

            fill_color=marker_color,

            fill_opacity=0.85,

            tooltip=(
                f"Station {sid} | "
                f"{row.get('River', '')} | "
                f"{row.get('Location', '')}"
            ),

            popup=folium.Popup(
                popup_html,
                max_width=320
            )

        ).add_to(cluster)

    # --------------------------------------------------------
    # Highlight selected station separately
    # --------------------------------------------------------

    selected_popup = f"""
    <b>Selected Station {station_id}</b><br>
    River: {meta.get('River', 'NA')}<br>
    Location: {meta.get('Location', 'NA')}
    """

    folium.Marker(

        location=[lat, lon],

        tooltip=(
            f"Selected Station {station_id}"
        ),

        popup=selected_popup,

        icon=folium.Icon(
            color="red",
            icon="info-sign"
        )

    ).add_to(m)

    # --------------------------------------------------------
    # Layer control
    # --------------------------------------------------------

    folium.LayerControl(
        collapsed=False
    ).add_to(m)

    # --------------------------------------------------------
    # Show map in Streamlit
    # --------------------------------------------------------

    from streamlit_folium import st_folium

    st_folium(
        m,
        width=None,
        height=650,
        use_container_width=True
    )

    st.caption(
        "🔴 Selected station · 🔵 Other DHM stations · "
        "Use the layer control to switch between OpenStreetMap, "
        "terrain, light, and dark basemaps."
    )

# ============================================================
# EXPORT TAB
# ============================================================
with tab_download:

    st.markdown("#### Export Selected Station Results")

    annual_station = pd.DataFrame()
    if annual_source is not None and not annual_source.empty and station_id in annual_source.columns:
        annual_station = annual_source[["Year", station_id]].copy().rename(columns={station_id: "Discharge"})
        annual_station["Year"] = pd.to_numeric(annual_station["Year"], errors="coerce")
        annual_station["Discharge"] = pd.to_numeric(annual_station["Discharge"], errors="coerce")
        annual_station = annual_station.dropna()

    export_bytes = build_download_excel(
        station_meta=meta, summary_row=summary_row, design_row=design_row,
        ranking_df=rank_match, annual_df=annual_station
    )

    filename = f"Station_{station_id}_{analysis_short}_HydroEVT_DSS.xlsx"

    st.download_button(
        label="⬇️ Download selected station analysis",
        data=export_bytes,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    st.markdown("#### Data Quality Check")
    qc = {
        "Station metadata available": True,
        "HydroEVT summary available": summary_row is not None,
        "Design-flow results available": design_row is not None,
        "Distribution ranking available": not rank_match.empty,
        "Annual series available": not annual_station.empty,
    }
    qc_df = pd.DataFrame({"Check": qc.keys(), "Available": qc.values()})
    st.dataframe(qc_df, use_container_width=True, hide_index=True)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "Nepal Hydrological Extreme Flow DSS | Station metadata from DHM Streamflow Summary | "
    "Frequency-analysis outputs from HydroEVT workflow."
)
