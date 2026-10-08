"""
Dashboard — Hourly Passenger Demand Forecasting for Mass Transit Stations.

Jalankan:
    streamlit run dashboard/app.py

Desain mengikuti kaidah data-viz (skill `dataviz`):
  - Filter satu baris di atas konten yang di-scope; semua KPI, chart, dan peta
    di tab "Operasional" dihitung dari slice yang sama (stasiun + rentang waktu).
  - Satu hero figure per tampilan, stat tile untuk angka pendukung; delta memakai
    ikon + label, bukan warna saja.
  - Bentuk "emphasis": seri model di warna aksen, konteks di abu-abu.
  - Peta memakai skala sekuensial satu hue (bentuk scatter maks. 3 kategori).
  - Grid hairline solid, mark tipis, tooltip crosshair, tabel data sebagai
    kembaran setiap chart.
Aset gratis: font Inter (Google Fonts, OFL), ikon Lucide (ISC, disimpan lokal di
dashboard/assets/icons), peta dasar OpenStreetMap (tanpa token, ditampilkan pudar).
"""
import html
import math
import re
from datetime import timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
ICON_DIR = Path(__file__).resolve().parent / "assets" / "icons"

st.set_page_config(
    page_title="MTA Ridership Forecasting",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Design tokens — palet referensi skill dataviz (mode terang), divalidasi dengan
# scripts/validate_palette.js: aksen #2a78d6 PASS, ramp ordinal peta PASS.
# ---------------------------------------------------------------------------
T = {
    "page": "#f9f9f7",
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "accent": "#2a78d6",
    "good": "#006300",
    "bad": "#d03b3b",
    "border": "rgba(11,11,11,0.10)",
    "navy": "#141B34",
    "amber": "#F5A623",
}
SEQ_RAMP = ["#86b6ef", "#5598e7", "#256abf", "#184f95", "#0d366b"]
FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

FEATURE_LABELS = {
    "lag_168h": "Lag 168 jam (minggu lalu)",
    "lag_24h": "Lag 24 jam (kemarin)",
    "lag_1h": "Lag 1 jam",
    "rolling_mean_3h": "Rata-rata bergulir 3 jam",
    "rolling_mean_24h": "Rata-rata bergulir 24 jam",
    "hour_of_day": "Jam",
    "day_of_week": "Hari",
    "is_weekend": "Akhir pekan",
    "month": "Bulan",
    "station_complex": "Stasiun",
    "borough": "Borough",
    "transit_mode": "Moda transit",
}
MODEL_LABELS = {
    "lgbm_tuned": "LightGBM (tuned)",
    "lgbm_untuned": "LightGBM (default)",
    "naive": "Baseline naif",
    "rolling_avg_24h": "Rata-rata 24 jam",
}
TIMESFM_LABELS = {
    "TimesFM 3.0": T["accent"],
    "Chronos-2": "#df6c36",
    "PatchTST": "#17846c",
    "LightGBM recursive": T["navy"],
    "Weekly seasonal naive": T["muted"],
}


# ---------------------------------------------------------------------------
# Format angka gaya Indonesia (ribuan ".", desimal ",")
# ---------------------------------------------------------------------------
def fmt_num(x: float, decimals: int = 0) -> str:
    s = f"{x:,.{decimals}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def fmt_compact(x: float) -> str:
    for div, unit in [(1e6, "jt"), (1e3, "rb")]:
        if abs(x) >= div:
            return f"{fmt_num(x / div, 1).removesuffix(',0')} {unit}"
    return fmt_num(x)


def fmt_date(d) -> str:
    bulan = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
    return f"{d.day} {bulan[d.month - 1]} {d.year}"


# ---------------------------------------------------------------------------
# Aset: ikon Lucide inline (warna ikut currentColor)
# ---------------------------------------------------------------------------
@st.cache_resource
def _load_icons() -> dict:
    icons = {}
    for f in ICON_DIR.glob("*.svg"):
        svg = f.read_text(encoding="utf-8")
        svg = re.sub(r"<!--.*?-->", "", svg, flags=re.S).strip()
        svg = re.sub(r'\s(width|height)="24"', "", svg)
        # SVG multi-baris berindentasi dipecah parser markdown Streamlit jadi teks.
        svg = re.sub(r"\s+", " ", svg).replace("> <", "><")
        icons[f.stem] = svg
    return icons


def icon(name: str, cls: str = "ic") -> str:
    svg = _load_icons().get(name, "")
    return svg.replace('class="lucide', f'aria-hidden="true" class="{cls} lucide', 1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data
def load_predictions() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "predictions.parquet")
    df["station_complex"] = df["station_complex"].astype(str)
    df["date"] = df["transit_timestamp"].dt.date
    df["hour"] = df["transit_timestamp"].dt.hour
    return df


@st.cache_data
def load_locations() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "station_locations.parquet")


@st.cache_data
def load_metrics_table() -> pd.DataFrame:
    return pd.read_csv(MODELS_DIR / "evaluation_metrics.csv")


def _benchmark_signature() -> tuple:
    """Include artifact mtimes so a live Streamlit process sees refreshed model outputs."""
    paths = (
        MODELS_DIR / "chronos_patchtst_benchmark_metrics.csv",
        PROCESSED_DIR / "chronos_patchtst_predictions.parquet",
        MODELS_DIR / "chronos_patchtst_horizon_metrics.csv",
        MODELS_DIR / "timesfm3_benchmark_metrics.csv",
        PROCESSED_DIR / "timesfm3_benchmark_predictions.parquet",
        MODELS_DIR / "timesfm3_horizon_metrics.csv",
    )
    return tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else (p.name, None, None) for p in paths)


@st.cache_data
def load_forecast_benchmark(signature: tuple) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    del signature  # cache key only; filesystem state is captured by _benchmark_signature().
    # Prefer the paired Chronos-2/PatchTST run when all of its outputs are present.
    expanded = (
        MODELS_DIR / "chronos_patchtst_benchmark_metrics.csv",
        PROCESSED_DIR / "chronos_patchtst_predictions.parquet",
        MODELS_DIR / "chronos_patchtst_horizon_metrics.csv",
    )
    legacy = (
        MODELS_DIR / "timesfm3_benchmark_metrics.csv",
        PROCESSED_DIR / "timesfm3_benchmark_predictions.parquet",
        MODELS_DIR / "timesfm3_horizon_metrics.csv",
    )
    expanded_complete = all(p.exists() for p in expanded)
    metrics_path, prediction_path, horizon_path = expanded if expanded_complete else legacy
    if not all(path.exists() for path in (metrics_path, prediction_path, horizon_path)):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), ""
    return (
        pd.read_csv(metrics_path),
        pd.read_parquet(prediction_path),
        pd.read_csv(horizon_path),
        "Chronos-2 + PatchTST" if expanded_complete else "TimesFM only (artefak benchmark terbaru belum lengkap)",
    )


@st.cache_resource
def load_booster() -> lgb.Booster:
    return lgb.Booster(model_file=str(MODELS_DIR / "lightgbm_final.txt"))


@st.cache_data
def station_totals(start, end) -> pd.DataFrame:
    df = load_predictions()
    sl = df[(df["date"] >= start) & (df["date"] <= end)]
    return sl.groupby("station_complex_id", observed=True)["entries"].sum().reset_index(name="total")


def error_metrics(y: pd.Series, yhat: pd.Series) -> dict:
    err = (y - yhat).abs()
    denom = y.sum()
    return {
        "mae": err.mean(),
        "rmse": float(np.sqrt(((y - yhat) ** 2).mean())),
        "wape": err.sum() / denom if denom > 0 else math.nan,
    }


# ---------------------------------------------------------------------------
# Styling global
# ---------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea {{
  font-family: {FONT};
}}
.stApp {{
  background: {T['page']}; color: {T['ink']}; color-scheme: light;
  --primary-color: {T['accent']}; --text-color: {T['ink']};
  --background-color: {T['page']}; --secondary-background-color: {T['surface']};
}}
[data-testid="stHeader"] {{ background: transparent; height: 0; }}
.block-container {{ max-width: 1320px; padding-top: 1.6rem; padding-bottom: 3rem; }}

/* Keep Streamlit controls legible even when the browser has a saved dark theme. */
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
[data-testid="stRadio"] label, [data-testid="stRadio"] label p,
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] {{
  color: {T['ink2']} !important; opacity: 1 !important;
}}
[data-testid="stDateInput"] input, [data-baseweb="select"] > div {{
  background: {T['surface']} !important; color: {T['ink']} !important;
  border-color: {T['axis']} !important; color-scheme: light;
}}
[data-testid="stDateInput"] input:disabled {{
  -webkit-text-fill-color: {T['ink']} !important; opacity: 1 !important;
}}
[data-testid="stDateInput"] svg, [data-baseweb="select"] svg {{ color: {T['ink2']} !important; }}
[data-baseweb="popover"] ul, [data-baseweb="popover"] li {{
  background: {T['surface']} !important; color: {T['ink']} !important;
}}
.stTabs [data-baseweb="tab"] {{ color: {T['ink2']} !important; }}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
  color: {T['accent']} !important; border-bottom-color: {T['accent']} !important;
}}

/* --- Banner brand (menyambung ke deck portfolio: navy + amber) --- */
.banner {{
  background: {T['navy']}; border-radius: 16px; padding: 26px 30px;
  display: flex; align-items: center; gap: 22px; margin-bottom: 8px;
  animation: rise .5s ease-out both;
}}
.banner .badge {{
  width: 54px; height: 54px; border-radius: 50%; background: {T['amber']};
  display: grid; place-items: center; flex: none; color: {T['navy']};
}}
.banner .badge svg {{ width: 28px; height: 28px; stroke-width: 2; }}
.banner .kicker {{
  color: {T['amber']}; font-size: 12px; font-weight: 600; letter-spacing: .14em;
  text-transform: uppercase; margin: 0 0 4px 0;
}}
.banner h1 {{
  color: #fff; font-size: 28px; font-weight: 700; margin: 0; padding: 0;
  line-height: 1.2; letter-spacing: -.01em;
}}
.banner p {{ color: #CADCFC; margin: 6px 0 0 0; font-size: 14.5px; }}

/* --- Tabs --- */
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {T['grid']}; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; font-size: 15px; padding: 10px 14px; }}

/* --- Kartu (st.container(border=True)) --- */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] > [data-testid="element-container"] > [data-testid="stMarkdown"] .card-head) {{
  background: {T['surface']}; border: 1px solid {T['border']} !important;
  border-radius: 14px; box-shadow: 0 1px 2px rgba(11,11,11,.04);
  animation: rise .45s ease-out both;
}}
.card-head {{ display: flex; gap: 12px; align-items: flex-start; margin-bottom: 2px; }}
.card-head .ic {{ width: 20px; height: 20px; color: {T['accent']}; margin-top: 3px; flex: none; }}
.card-head h3 {{ font-size: 17px; font-weight: 600; color: {T['ink']}; margin: 0; padding: 0; }}
.card-head p {{ font-size: 13.5px; color: {T['ink2']}; margin: 3px 0 0 0; }}

/* --- Baris scope filter --- */
.scope {{ font-size: 13.5px; color: {T['ink2']}; margin: -4px 0 14px 0; line-height: 1.5; }}
.scope b {{ color: {T['ink']}; font-weight: 600; }}
.scope .ic {{ width: 15px; height: 15px; color: {T['muted']}; vertical-align: -3px; margin-right: 6px; }}

/* Stack the profile and map panels on narrow screens so chart labels stay legible. */
@media (max-width: 900px) {{
  [data-testid="stHorizontalBlock"]:has(.card-head h3) {{
    flex-direction: column !important; align-items: stretch !important; gap: 1rem !important;
  }}
  [data-testid="stHorizontalBlock"]:has(.card-head h3) > [data-testid="column"] {{
    width: 100% !important; min-width: 100% !important; flex: 1 1 100% !important;
  }}
}}

/* --- KPI row --- */
.kpis {{ display: grid; grid-template-columns: 1.55fr repeat(4, 1fr); gap: 14px; margin: 4px 0 18px 0; }}
@media (max-width: 1000px) {{ .kpis {{ grid-template-columns: 1fr 1fr; }} }}
.tile {{
  background: {T['surface']}; border: 1px solid {T['border']}; border-radius: 14px;
  padding: 16px 18px; box-shadow: 0 1px 2px rgba(11,11,11,.04);
  transition: transform .18s ease, box-shadow .18s ease;
  animation: rise .45s ease-out both;
}}
.tile:hover {{ transform: translateY(-2px); box-shadow: 0 6px 18px rgba(11,11,11,.08); }}
.tile:nth-child(2) {{ animation-delay: .05s; }}
.tile:nth-child(3) {{ animation-delay: .10s; }}
.tile:nth-child(4) {{ animation-delay: .15s; }}
.tile:nth-child(5) {{ animation-delay: .20s; }}
.tile .lbl {{ display: flex; align-items: center; gap: 7px; font-size: 13px; color: {T['ink2']}; font-weight: 500; }}
.tile .lbl .ic {{ width: 16px; height: 16px; color: {T['muted']}; flex: none; }}
.tile .val {{ font-size: 28px; font-weight: 600; color: {T['ink']}; margin-top: 8px; line-height: 1.1; letter-spacing: -.01em; }}
.tile .val small {{ font-size: 13px; font-weight: 500; color: {T['ink2']}; margin-left: 4px; letter-spacing: 0; }}
.tile .sub {{ font-size: 12.5px; color: {T['ink2']}; margin-top: 6px; }}
.tile.hero .val {{ font-size: 52px; font-weight: 700; }}
.delta {{ display: inline-flex; align-items: center; gap: 5px; font-size: 12.5px; font-weight: 600; margin-top: 8px; }}
.delta .ic {{ width: 15px; height: 15px; }}
.delta.good {{ color: {T['good']}; }}
.delta.bad {{ color: {T['bad']}; }}
.delta span.note {{ color: {T['ink2']}; font-weight: 500; }}

/* --- Footer --- */
.foot {{ font-size: 12px; color: {T['muted']}; margin-top: 26px; line-height: 1.6; }}
.foot a {{ color: {T['ink2']}; }}

@keyframes rise {{
  from {{ opacity: 0; transform: translateY(8px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation: none !important; transition: none !important; }}
}}
</style>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Komponen HTML
# ---------------------------------------------------------------------------
def card_head(title: str, subtitle: str, icon_name: str) -> None:
    st.markdown(
        f'<div class="card-head">{icon(icon_name)}<div><h3>{html.escape(title)}</h3>'
        f"<p>{html.escape(subtitle)}</p></div></div>",
        unsafe_allow_html=True,
    )


def data_table(key: str, df: pd.DataFrame, **kw) -> None:
    """Kembaran tabel sebuah chart, dirender hanya saat diminta.

    st.dataframe di dalam expander tertutup (Streamlit 1.33) tidak memperbarui
    jumlah barisnya saat data menyusut antar filter, sehingga frontend melempar
    'Row index is out of range'.
    """
    if st.toggle("Tampilkan tabel data", key=key):
        st.dataframe(df, hide_index=True, use_container_width=True, **kw)


def tile(label: str, value: str, icon_name: str, *, unit: str = "", sub: str = "",
         delta: str = "", hero: bool = False) -> str:
    unit_html = f"<small>{html.escape(unit)}</small>" if unit else ""
    sub_html = f'<div class="sub">{html.escape(sub)}</div>' if sub else ""
    return (
        f'<div class="tile{" hero" if hero else ""}">'
        f'<div class="lbl">{icon(icon_name)}{html.escape(label)}</div>'
        f'<div class="val">{html.escape(value)}{unit_html}</div>{delta}{sub_html}</div>'
    )


def delta_vs(model_val: float, base_val: float, base_name: str) -> str:
    """Delta error model vs pembanding. Error lebih rendah = baik."""
    if not base_val or math.isnan(base_val):
        return ""
    change = (model_val - base_val) / base_val
    good = change <= 0
    arrow = "trending-down" if change <= 0 else "trending-up"
    word = "lebih baik" if good else "lebih buruk"
    return (
        f'<div class="delta {"good" if good else "bad"}">{icon(arrow)}'
        f'{fmt_num(abs(change) * 100, 1)}% {word} '
        f'<span class="note">vs {html.escape(base_name)}</span></div>'
    )


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------
def base_layout(height: int, **kw) -> dict:
    layout = dict(
        height=height,
        margin=dict(l=8, r=16, t=12, b=8),
        paper_bgcolor=T["surface"],
        plot_bgcolor=T["surface"],
        font=dict(family=FONT, size=12.5, color=T["ink2"]),
        separators=",.",
        hoverlabel=dict(bgcolor=T["surface"], bordercolor=T["axis"],
                        font=dict(family=FONT, size=13, color=T["ink"])),
        showlegend=False,
    )
    layout.update(kw)
    return layout


# Transisi Plotly hanya untuk chart yang nilainya berubah saat filter diganti.
# Di chart statis ia justru merugikan: saat tab dibuka/jendela di-resize, bar
# "tumbuh" sementara label teks langsung melompat ke posisi akhir, dan pada
# chart garis ribuan titik SVG-nya gagal ikut resize.
FILTER_TRANSITION = dict(duration=450, easing="cubic-in-out")


def axis(**kw) -> dict:
    a = dict(
        automargin=True,
        showgrid=True, gridcolor=T["grid"], gridwidth=1, griddash="solid",
        zeroline=False, showline=True, linecolor=T["axis"], linewidth=1,
        tickfont=dict(color=T["muted"], size=11.5), ticks="",
        title=dict(font=dict(color=T["muted"], size=12)),
    )
    a.update(kw)
    return a


def fig_actual_vs_pred(sl: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sl["transit_timestamp"], y=sl["pred_naive"], name="Baseline naif",
        mode="lines", line=dict(color=T["muted"], width=1.5, dash="dash"),
        legendrank=3, hovertemplate="%{y:,.0f}",
    ))
    fig.add_trace(go.Scatter(
        x=sl["transit_timestamp"], y=sl["entries"], name="Aktual",
        mode="lines", line=dict(color=T["ink2"], width=2),
        legendrank=1, hovertemplate="%{y:,.0f}",
    ))
    fig.add_trace(go.Scatter(
        x=sl["transit_timestamp"], y=sl["pred_lgbm_tuned"], name="Prediksi LightGBM",
        mode="lines", line=dict(color=T["accent"], width=2),
        legendrank=2, hovertemplate="%{y:,.0f}",
    ))
    fig.update_layout(**base_layout(
        400,
        showlegend=True,
        # Legenda di luar area plot (yanchor bottom di y>1): bila terbungkus di layar
        # sempit, Plotly menambah margin atas alih-alih menimpa garis data.
        legend=dict(orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom",
                    font=dict(size=12.5, color=T["ink2"]), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=8, r=16, t=12, b=8),
        hovermode="x unified",
        xaxis=axis(showgrid=False, tickformat="%d/%m", hoverformat="%d/%m/%Y · %H:00",
                   showspikes=True, spikemode="across", spikethickness=1,
                   spikecolor=T["axis"], spikedash="solid", spikesnap="cursor"),
        yaxis=axis(tickformat=",.0f", rangemode="tozero", title_text="Entries tercatat per jam"),
    ))
    return fig


def fig_timesfm_overlay(df: pd.DataFrame) -> go.Figure:
    columns = [
        "entries", "pred_timesfm3", "pred_chronos2", "pred_patchtst",
        "pred_lgbm_recursive", "pred_weekly_naive",
    ]
    columns = [column for column in columns if column in df.columns]
    hourly = df.groupby("transit_timestamp", as_index=False)[columns].sum()
    fig = go.Figure()
    series = [
        ("entries", "Aktual", T["ink2"], 2.2),
        ("pred_timesfm3", "TimesFM 3.0", T["accent"], 2.0),
        ("pred_chronos2", "Chronos-2", "#df6c36", 2.0),
        ("pred_patchtst", "PatchTST", "#17846c", 1.8),
        ("pred_lgbm_recursive", "LightGBM rekursif", T["navy"], 1.8),
        ("pred_weekly_naive", "Seasonal-naive", T["muted"], 1.5),
    ]
    for col, name, color, width in series:
        if col not in hourly.columns:
            continue
        fig.add_trace(go.Scatter(
            x=hourly["transit_timestamp"], y=hourly[col], mode="lines", name=name,
            line=dict(color=color, width=width),
            hovertemplate="%{y:,.0f}<extra>" + name + "</extra>",
        ))
    fig.update_layout(**base_layout(
        390, showlegend=True, hovermode="x unified",
        legend=dict(orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom",
                    font=dict(size=12, color=T["ink2"]), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=8, r=16, t=18, b=8),
        xaxis=axis(showgrid=False, tickformat="%d/%m", hoverformat="%d/%m/%Y · %H:00"),
        yaxis=axis(tickformat=",.0f", rangemode="tozero", title_text="Entries agregat per jam"),
    ))
    return fig


def fig_timesfm_horizon(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for model, color in TIMESFM_LABELS.items():
        sl = df[df["model"] == model].sort_values("horizon_step")
        if sl.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sl["horizon_step"], y=sl["mae"], mode="lines+markers", name=model,
            line=dict(color=color, width=2), marker=dict(size=5),
            hovertemplate="Jam ke-%{x}<br>MAE %{y:,.1f}<extra>" + model + "</extra>",
        ))
    fig.update_layout(**base_layout(
        330, showlegend=True,
        legend=dict(orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom",
                    font=dict(size=11.5, color=T["ink2"]), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=8, r=12, t=18, b=8),
        xaxis=axis(dtick=3, title_text="Langkah forecast (jam)"),
        yaxis=axis(tickformat=",.0f", rangemode="tozero", title_text="MAE per stasiun-jam"),
    ))
    return fig


def fig_station_forecast(df: pd.DataFrame) -> go.Figure:
    """Latest paired multi-step forecasts for the station selected in Operations."""
    series = [
        ("entries", "Aktual", T["ink2"], 2.2),
        ("pred_chronos2", "Chronos-2", "#df6c36", 2.0),
        ("pred_patchtst", "PatchTST", "#17846c", 1.8),
        ("pred_timesfm3", "TimesFM 3.0", T["accent"], 1.8),
        ("pred_lgbm_recursive", "LightGBM rekursif", T["navy"], 1.7),
        ("pred_weekly_naive", "Seasonal-naive", T["muted"], 1.5),
    ]
    fig = go.Figure()
    for column, name, color, width in series:
        if column not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df["transit_timestamp"], y=df[column], name=name, mode="lines",
            line=dict(color=color, width=width),
            hovertemplate="%{y:,.0f}<extra>" + name + "</extra>",
        ))
    fig.update_layout(**base_layout(
        390, showlegend=True, hovermode="x unified",
        legend=dict(orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom",
                    font=dict(size=11.5, color=T["ink2"]), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=8, r=16, t=18, b=8),
        xaxis=axis(showgrid=False, tickformat="%d/%m", hoverformat="%d/%m/%Y · %H:00"),
        yaxis=axis(tickformat=",.0f", rangemode="tozero", title_text="Entries tercatat per jam"),
    ))
    return fig


def fig_hour_profile(profile: pd.DataFrame, peak_hour: int, peak_val: float) -> go.Figure:
    colors = [T["accent"] if h == peak_hour else T["muted"] for h in profile["hour"]]
    fig = go.Figure(go.Bar(
        x=profile["hour"], y=profile["entries"], marker=dict(color=colors, cornerradius=4),
        width=0.62, customdata=[f"{h:02d}:00" for h in profile["hour"]],
        hovertemplate="<b>%{y:,.0f}</b> entries/jam<br>%{customdata}<extra></extra>",
    ))
    fig.add_annotation(
        x=peak_hour, y=peak_val, yshift=14, showarrow=False,
        text=f"<b>Puncak {peak_hour:02d}:00</b> · {fmt_num(peak_val)}/jam",
        font=dict(size=12.5, color=T["ink"]),
    )
    fig.update_layout(**base_layout(
        330,
        transition=FILTER_TRANSITION,
        bargap=0.3,
        xaxis=axis(showgrid=False, tickmode="array", tickvals=list(range(0, 24, 3)),
                   ticktext=[f"{h:02d}" for h in range(0, 24, 3)], title_text="Jam"),
        yaxis=axis(tickformat=",.0f", range=[0, max(peak_val, 1) * 1.2]),
    ))
    return fig


def fig_station_map(map_df: pd.DataFrame, selected: str) -> go.Figure:
    vals = map_df["total"].clip(lower=1)
    logv = np.log10(vals)
    lo, hi = float(logv.min()), float(logv.max())
    if hi <= lo:
        hi = lo + 1
    ticks = [k for k in range(math.floor(lo), math.ceil(hi) + 1) if lo - 0.01 <= k <= hi + 0.01]
    size = 8 + 14 * np.sqrt((vals - vals.min()) / max(vals.max() - vals.min(), 1))

    fig = go.Figure(go.Scattermapbox(
        lat=map_df["latitude"], lon=map_df["longitude"], mode="markers",
        marker=dict(size=size, color=logv, cmin=lo, cmax=hi, opacity=0.9,
                    colorscale=[[i / (len(SEQ_RAMP) - 1), c] for i, c in enumerate(SEQ_RAMP)],
                    colorbar=dict(title=dict(text="Total entries", side="top",
                                             font=dict(size=11.5, color=T["ink2"])),
                                  orientation="h", x=0.02, xanchor="left", y=0.03, yanchor="bottom",
                                  len=0.38, thickness=8, outlinewidth=0,
                                  tickvals=ticks, ticktext=[fmt_compact(10 ** k) for k in ticks],
                                  tickfont=dict(size=11, color=T["ink2"]),
                                  bgcolor="rgba(252,252,251,0.9)")),
        customdata=np.stack([map_df["station_complex"], map_df["borough"],
                             [fmt_num(v) for v in map_df["total"]]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
                      "%{customdata[2]} entries tercatat<extra></extra>",
    ))
    sel = map_df[map_df["station_complex"] == selected]
    if len(sel):
        for sz, col in [(26, T["ink"]), (16, T["surface"])]:
            fig.add_trace(go.Scattermapbox(
                lat=sel["latitude"], lon=sel["longitude"], mode="markers",
                marker=dict(size=sz, color=col), hoverinfo="skip",
            ))
        center = dict(lat=float(sel["latitude"].iloc[0]), lon=float(sel["longitude"].iloc[0]))
    else:
        center = dict(lat=float(map_df["latitude"].mean()), lon=float(map_df["longitude"].mean()))

    fig.update_layout(**base_layout(
        380,
        margin=dict(l=0, r=0, t=0, b=0),
        # Tile CARTO kini butuh API key; OSM gratis dipakai sebagai raster pudar
        # supaya peta dasar tetap tenang dan titik data yang menonjol.
        mapbox=dict(
            style="white-bg", center=center, zoom=10.2,
            layers=[dict(
                below="traces", sourcetype="raster", opacity=0.45,
                source=["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
                sourceattribution="© OpenStreetMap contributors",
            )],
        ),
    ))
    return fig


def fig_model_bars(df: pd.DataFrame, col: str) -> go.Figure:
    df = df.sort_values(col, ascending=False)
    colors = [T["accent"] if m == "lgbm_tuned" else T["muted"] for m in df["model"]]
    fig = go.Figure(go.Bar(
        y=[MODEL_LABELS[m] for m in df["model"]], x=df[col], orientation="h",
        marker=dict(color=colors, cornerradius=4), width=0.5,
        text=[fmt_num(v, 1) for v in df[col]], textposition="outside",
        textfont=dict(color=T["ink"], size=13), cliponaxis=False,
        hovertemplate="%{y}<br><b>%{x:,.2f}</b><extra></extra>",
    ))
    fig.update_layout(**base_layout(
        230,
        margin=dict(l=8, r=40, t=8, b=8),
        xaxis=axis(tickformat=",.0f", rangemode="tozero", range=[0, df[col].max() * 1.18]),
        yaxis=axis(showgrid=False, showline=False, tickfont=dict(color=T["ink2"], size=12.5)),
    ))
    return fig


def fig_feature_importance(imp: pd.DataFrame) -> go.Figure:
    imp = imp.sort_values("share")
    labels = [fmt_num(v, 1) + "%" if v >= 1 else "" for v in imp["share"]]
    fig = go.Figure(go.Bar(
        y=imp["label"], x=imp["share"], orientation="h",
        marker=dict(color=T["accent"], cornerradius=4), width=0.6,
        text=labels, textposition="outside", textfont=dict(color=T["ink"], size=12.5),
        cliponaxis=False,
        hovertemplate="%{y}<br><b>%{x:,.2f}%</b> dari total gain<extra></extra>",
    ))
    fig.update_layout(**base_layout(
        420,
        margin=dict(l=8, r=48, t=8, b=8),
        xaxis=axis(ticksuffix="%", range=[0, imp["share"].max() * 1.15]),
        yaxis=axis(showgrid=False, showline=False, tickfont=dict(color=T["ink2"], size=12.5)),
    ))
    return fig


# ---------------------------------------------------------------------------
# Halaman
# ---------------------------------------------------------------------------
inject_css()
preds = load_predictions()
locations = load_locations()
metrics_table = load_metrics_table()
benchmark_metrics, benchmark_predictions, benchmark_horizon_metrics, benchmark_source = load_forecast_benchmark(
    _benchmark_signature()
)
station_count = preds["station_complex_id"].nunique()

st.markdown(
    f"""
<div class="banner">
  <div class="badge">{icon("train-front")}</div>
  <div>
    <div class="kicker">MTA New York · {fmt_num(station_count)} stasiun · forecasting per jam</div>
    <h1>Hourly Passenger Demand Forecasting</h1>
    <p>Entries tercatat per stasiun per jam. Bandingkan forecast 24 jam Chronos-2, PatchTST, TimesFM, dan LightGBM.</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

tab_ops, tab_model = st.tabs(["Operasional stasiun", "Performa model"])

# ============================ TAB OPERASIONAL ===============================
with tab_ops:
    min_date, max_date = preds["date"].min(), preds["date"].max()
    station_lookup = preds[["station_complex_id", "station_complex"]].drop_duplicates()
    station_lookup = station_lookup.sort_values("station_complex", kind="stable")
    station_names = dict(zip(station_lookup["station_complex_id"].astype(str), station_lookup["station_complex"]))
    station_lookup["station_complex_id"] = station_lookup["station_complex_id"].astype(str)
    station_lookup["selector_label"] = station_lookup["station_complex"].astype(str)
    duplicate_names = station_lookup["station_complex"].duplicated(keep=False)
    station_lookup.loc[duplicate_names, "selector_label"] += (
        " · ID " + station_lookup.loc[duplicate_names, "station_complex_id"]
    )
    station_ids_by_label = dict(zip(station_lookup["selector_label"], station_lookup["station_complex_id"]))
    station_labels = station_lookup["selector_label"].tolist()
    busiest_id = str(preds.groupby("station_complex_id", observed=True)["entries"].sum().idxmax())
    busiest_label = station_lookup.loc[
        station_lookup["station_complex_id"] == busiest_id, "selector_label"
    ].iloc[0]
    latest_ts = preds["transit_timestamp"].max()
    latest_complete_date = latest_ts.date() if latest_ts.hour == 23 else latest_ts.date() - timedelta(days=1)

    preset = st.radio(
        "Rentang waktu",
        ["7 hari lengkap", "30 hari lengkap", "Seluruh periode test", "Kustom"],
        horizontal=True,
    )
    f_date, f_station = st.columns([1, 1.45], gap="medium")
    with f_date:
        if preset == "Kustom":
            picked = st.date_input(
                "Tanggal", value=(max_date - timedelta(days=13), max_date),
                min_value=min_date, max_value=max_date, format="DD/MM/YYYY",
            )
            if isinstance(picked, tuple) and len(picked) == 2:
                start, end = picked
            else:
                start, end = min_date, max_date
                st.caption("Pilih tanggal akhir untuk menerapkan rentang kustom.")
        else:
            days = {"30 hari lengkap": 30, "7 hari lengkap": 7}.get(preset)
            end = min(max_date, latest_complete_date) if days else max_date
            start = max(min_date, end - timedelta(days=days - 1)) if days else min_date
            st.date_input("Tanggal", value=(start, end), disabled=True, format="DD/MM/YYYY")
    with f_station:
        station_widget_key = "station_complex_id_select"
        if st.session_state.get(station_widget_key) not in station_labels:
            st.session_state.pop(station_widget_key, None)
        selected_station_label = st.selectbox(
            "Stasiun", station_labels, index=station_labels.index(busiest_label),
            key=station_widget_key,
        )
        station_id = station_ids_by_label[selected_station_label]
        station = station_names[station_id]

    sl = preds[
        (preds["station_complex_id"].astype(str) == station_id)
        & (preds["date"] >= start) & (preds["date"] <= end)
    ].sort_values("transit_timestamp")

    if sl.empty:
        st.info("Tidak ada data untuk kombinasi stasiun dan rentang waktu ini.")
        st.stop()

    actual_start = sl["transit_timestamp"].min()
    actual_end = sl["transit_timestamp"].max()
    st.markdown(
        f'<div class="scope">{icon("info")}Menampilkan <b>{html.escape(station)}</b> · '
        f"{fmt_date(actual_start.date())} {actual_start:%H:%M} – "
        f"{fmt_date(actual_end.date())} {actual_end:%H:%M} · "
        f"{fmt_num(len(sl))} observasi stasiun-jam</div>",
        unsafe_allow_html=True,
    )

    m_model = error_metrics(sl["entries"], sl["pred_lgbm_tuned"])
    m_naive = error_metrics(sl["entries"], sl["pred_naive"])
    profile = sl.groupby("hour")["entries"].mean().reindex(range(24), fill_value=0).reset_index()
    peak_row = profile.loc[profile["entries"].idxmax()]
    peak_hour, peak_val = int(peak_row["hour"]), float(peak_row["entries"])

    wape_txt = "–" if math.isnan(m_model["wape"]) else f"{fmt_num(m_model['wape'] * 100, 1)}%"
    st.markdown(
        '<div class="kpis">'
        + tile("Error LightGBM one-step (MAE)", fmt_num(m_model["mae"], 1), "target",
               unit="entries tercatat/jam", hero=True,
               delta=delta_vs(m_model["mae"], m_naive["mae"], "baseline naif"))
        + tile("RMSE model", fmt_num(m_model["rmse"], 1), "activity",
               delta=delta_vs(m_model["rmse"], m_naive["rmse"], "naif"))
        + tile("Error relatif (WAPE)", wape_txt, "gauge",
               sub="Jumlah absolut error ÷ jumlah entries aktual")
        + tile("Total entries aktual", fmt_compact(sl["entries"].sum()), "users",
               sub=f"{fmt_num(sl['entries'].mean())} rata-rata per jam")
        + tile("Jam puncak", f"{peak_hour:02d}:00", "clock-3",
               sub=f"{fmt_num(peak_val)} entries/jam rata-rata")
        + "</div>",
        unsafe_allow_html=True,
    )

    if not benchmark_predictions.empty:
        end_exclusive = pd.Timestamp(end) + pd.Timedelta(days=1)
        station_forecasts = benchmark_predictions[
            (benchmark_predictions["station_complex_id"].astype(str) == station_id)
            & (benchmark_predictions["transit_timestamp"] >= pd.Timestamp(start))
            & (benchmark_predictions["transit_timestamp"] < end_exclusive)
        ].sort_values("transit_timestamp")
        with st.container(border=True):
            card_head(
                "Forecast 24 jam · model terbaru",
                "Aktual dan prediksi Chronos-2, PatchTST, TimesFM, LightGBM rekursif, serta seasonal-naive "
                "untuk stasiun terpilih. Blok forecast harian tidak tumpang tindih.",
                "activity",
            )
            if station_forecasts.empty:
                coverage_start = benchmark_predictions["transit_timestamp"].min()
                coverage_end = benchmark_predictions["transit_timestamp"].max()
                st.info(
                    "Benchmark 24 jam tidak tersedia pada tanggal ini. Cakupan benchmark: "
                    f"{fmt_date(coverage_start.date())}–{fmt_date(coverage_end.date())}."
                )
            else:
                st.plotly_chart(fig_station_forecast(station_forecasts), use_container_width=True,
                                theme=None, config=PLOTLY_CONFIG)

    with st.container(border=True):
        card_head(
            "Evaluasi one-step · LightGBM",
            "Model dievaluasi per jam dengan forecast satu langkah. Baseline naif memakai nilai jam yang sama minggu lalu.",
            "activity",
        )
        st.plotly_chart(fig_actual_vs_pred(sl), use_container_width=True, theme=None,
                        config=PLOTLY_CONFIG)
        data_table(
            "tbl_hourly",
            sl[["transit_timestamp", "entries", "pred_lgbm_tuned", "pred_naive"]]
            .rename(columns={"transit_timestamp": "Waktu", "entries": "Aktual",
                             "pred_lgbm_tuned": "Prediksi LightGBM",
                             "pred_naive": "Baseline naif"}),
            column_config={
                "Waktu": st.column_config.DatetimeColumn(format="DD/MM/YYYY HH:00"),
                "Aktual": st.column_config.NumberColumn(format="%d"),
                "Prediksi LightGBM": st.column_config.NumberColumn(format="%.0f"),
                "Baseline naif": st.column_config.NumberColumn(format="%d"),
            },
        )

    c1, c2 = st.columns([5, 7], gap="medium")
    with c1:
        with st.container(border=True):
            card_head(
                "Profil jam sibuk",
                f"Rata-rata entries tercatat per jam di rentang terpilih. Puncak pukul {peak_hour:02d}:00.",
                "clock-3",
            )
            st.plotly_chart(fig_hour_profile(profile, peak_hour, peak_val),
                            use_container_width=True, theme=None, config=PLOTLY_CONFIG)
            data_table(
                "tbl_profile",
                profile.assign(Jam=profile["hour"].map(lambda h: f"{h:02d}:00"))
                [["Jam", "entries"]].rename(columns={"entries": "Rata-rata entries/jam"}),
                column_config={"Rata-rata entries/jam": st.column_config.NumberColumn(format="%.0f")},
            )

    with c2:
        with st.container(border=True):
            card_head(
                "Peta volume stasiun",
                "Warna & ukuran titik = total entries tercatat di rentang terpilih (skala log). "
                "Stasiun terpilih diberi cincin gelap.",
                "map",
            )
            totals = station_totals(start, end)
            map_df = locations.merge(totals, on="station_complex_id", how="inner", validate="one_to_one")
            if map_df.empty:
                st.info("Lokasi stasiun tidak tersedia untuk rentang yang dipilih.")
            else:
                st.plotly_chart(fig_station_map(map_df, station), use_container_width=True,
                                theme=None, config=PLOTLY_CONFIG)
            data_table(
                "tbl_map",
                map_df.sort_values("total", ascending=False)
                [["station_complex", "borough", "total"]]
                .rename(columns={"station_complex": "Stasiun", "borough": "Borough",
                                 "total": "Total entries"}),
                column_config={"Total entries": st.column_config.NumberColumn(format="%d")},
            )

# ============================== TAB MODEL ===================================
with tab_model:
    st.subheader("Benchmark utama · forecast 24 jam")
    if benchmark_metrics.empty:
        st.warning("Artefak forecast belum tersedia. Jalankan benchmark TimesFM dan Chronos-2/PatchTST setelah data bersih.")
    else:
        ranked = benchmark_metrics.sort_values("mae", kind="stable").reset_index(drop=True)
        best = ranked.iloc[0]
        runner_up = ranked.iloc[1] if len(ranked) > 1 else None
        gain_vs_runner_up = (1 - best["mae"] / runner_up["mae"]) * 100 if runner_up is not None else math.nan
        target_start = benchmark_predictions["transit_timestamp"].min()
        target_end = benchmark_predictions["transit_timestamp"].max()
        st.caption(
            f"Sumber model: {benchmark_source}. {int(benchmark_metrics['origins'].max()):,} origin harian · "
            f"{int(benchmark_metrics['stations'].max()):,} stasiun · "
            f"{int(benchmark_metrics['rows'].max()):,} observasi stasiun-jam · "
            f"horizon 24 jam · {fmt_date(target_start.date())}–{fmt_date(target_end.date())}. "
            "Angka MAE/RMSE adalah error per stasiun-jam; entries adalah catatan entry, bukan orang unik."
        )
        if not benchmark_source.startswith("Chronos-2"):
            st.warning(benchmark_source)
        st.markdown(
            '<div class="kpis">'
            + tile("MAE terendah · " + str(best["model"]), fmt_num(best["mae"], 2),
                   "target", hero=True, unit="entries/stasiun-jam")
            + tile("RMSE model terbaik", fmt_num(best["rmse"], 2), "activity",
                   unit="entries/stasiun-jam")
            + tile("Selisih MAE vs berikutnya", "–" if math.isnan(gain_vs_runner_up)
                   else f"{fmt_num(gain_vs_runner_up, 1)}%", "trending-down",
                   sub=f"{runner_up['model']} · MAE {fmt_num(runner_up['mae'], 2)}" if runner_up is not None else "")
            + tile("Jumlah origin", fmt_num(best["origins"]), "clock-3",
                   sub=f"{fmt_num(best['stations'])} stasiun")
            + "</div>",
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            card_head("Akurasi semua model pada test yang sama", "MAE dan RMSE agregat per observasi stasiun-jam; lebih rendah lebih baik.", "target")
            st.dataframe(
                benchmark_metrics.rename(columns={
                    "model": "Model", "mae": "MAE", "rmse": "RMSE",
                    "macro_station_mae": "Rata-rata MAE stasiun", "rows": "Observasi stasiun-jam",
                })[["Model", "MAE", "RMSE", "Rata-rata MAE stasiun", "Observasi stasiun-jam"]],
                hide_index=True, use_container_width=True,
                column_config={
                    "MAE": st.column_config.NumberColumn(format="%.2f"),
                    "RMSE": st.column_config.NumberColumn(format="%.2f"),
                    "Rata-rata MAE stasiun": st.column_config.NumberColumn(format="%.2f"),
                    "Observasi stasiun-jam": st.column_config.NumberColumn(format="%d"),
                },
            )
        with st.container(border=True):
            card_head("Aktual vs prediksi agregat", "Jumlah entries dari seluruh stasiun per jam pada blok test 24 jam.", "activity")
            st.plotly_chart(fig_timesfm_overlay(benchmark_predictions), use_container_width=True,
                            theme=None, config=PLOTLY_CONFIG)
        with st.container(border=True):
            card_head("MAE menurut jarak forecast", "Error tiap jam ke-1 sampai ke-24 setelah origin; unit entries/stasiun-jam.", "clock-3")
            st.plotly_chart(fig_timesfm_horizon(benchmark_horizon_metrics), use_container_width=True,
                            theme=None, config=PLOTLY_CONFIG)

    st.divider()
    st.subheader("Evaluasi one-step · LightGBM (pembaruan per jam)")
    mt = metrics_table.set_index("model")
    tuned, untuned, naive = (mt.loc[k] for k in ["lgbm_tuned", "lgbm_untuned", "naive"])
    gain_vs_naive = 1 - tuned["mae"] / naive["mae"]
    gain_vs_default = 1 - tuned["mae"] / untuned["mae"]
    test_start = preds["transit_timestamp"].min()
    test_end = preds["transit_timestamp"].max()
    st.caption(
        f"{station_count:,} stasiun · {fmt_date(test_start.date())} {test_start:%H:%M}–"
        f"{fmt_date(test_end.date())} {test_end:%H:%M} · {len(preds):,} observasi. "
        "Evaluasi satu langkah ini terpisah dari benchmark forecast 24 jam di atas."
    )
    st.markdown(
        '<div class="kpis">'
        + tile("Lebih akurat dari baseline naif", f"{fmt_num(gain_vs_naive * 100, 1)}%",
               "target", hero=True,
               sub=f"MAE {fmt_num(tuned['mae'], 2)} vs {fmt_num(naive['mae'], 2)}")
        + tile("Lebih akurat dari LightGBM default", f"{fmt_num(gain_vs_default * 100, 1)}%",
               "brain-circuit", sub="Kontribusi tuning Optuna")
        + tile("MAE model", fmt_num(tuned["mae"], 2), "activity", unit="entries/stasiun-jam")
        + tile("RMSE model", fmt_num(tuned["rmse"], 2), "activity", unit="entries/stasiun-jam")
        + tile("Observasi test", fmt_compact(len(preds)), "table-2",
               sub=f"{fmt_num(len(preds))} stasiun-jam")
        + "</div>", unsafe_allow_html=True,
    )

    comp = metrics_table[metrics_table["model"] != "rolling_avg_24h"]
    b1, b2 = st.columns(2, gap="medium")
    with b1:
        with st.container(border=True):
            card_head("MAE per model", "Evaluasi one-step; lebih rendah lebih baik.", "target")
            st.plotly_chart(fig_model_bars(comp, "mae"), use_container_width=True,
                            theme=None, config=PLOTLY_CONFIG)
    with b2:
        with st.container(border=True):
            card_head("RMSE per model", "Evaluasi one-step; error besar mendapat penalti lebih tinggi.", "activity")
            st.plotly_chart(fig_model_bars(comp, "rmse"), use_container_width=True,
                            theme=None, config=PLOTLY_CONFIG)

    with st.container(border=True):
        card_head("Tabel pembanding one-step", "Termasuk rolling average 24 jam; hasilnya tidak dibandingkan langsung dengan horizon 24 jam.", "table-2")
        st.dataframe(
            metrics_table.assign(Model=metrics_table["model"].map(MODEL_LABELS))
            [["Model", "mae", "rmse"]].rename(columns={"mae": "MAE", "rmse": "RMSE"}),
            hide_index=True, use_container_width=True,
            column_config={"MAE": st.column_config.NumberColumn(format="%.2f"),
                           "RMSE": st.column_config.NumberColumn(format="%.2f")},
        )

    booster = load_booster()
    gains = booster.feature_importance(importance_type="gain")
    imp = pd.DataFrame({"feature": booster.feature_name(), "gain": gains})
    imp["share"] = imp["gain"] / imp["gain"].sum() * 100
    imp["label"] = imp["feature"].map(FEATURE_LABELS).fillna(imp["feature"])

    with st.container(border=True):
        card_head(
            "Fitur paling berpengaruh",
            "Porsi total gain LightGBM per fitur. Pola mingguan (lag 168 jam) adalah sinyal terkuat.",
            "brain-circuit",
        )
        st.plotly_chart(fig_feature_importance(imp), use_container_width=True,
                        theme=None, config=PLOTLY_CONFIG)
        data_table(
            "tbl_features",
            imp.sort_values("share", ascending=False)[["label", "feature", "share"]]
            .rename(columns={"label": "Fitur", "feature": "Nama kolom", "share": "Porsi gain (%)"}),
            column_config={"Porsi gain (%)": st.column_config.NumberColumn(format="%.2f")},
        )

st.markdown(
    '<div class="foot">Data: MTA Subway Hourly Ridership 2022–2024 (Kaggle). '
    "Ikon: <a href='https://lucide.dev' target='_blank'>Lucide</a> (ISC) · "
    "Font: <a href='https://rsms.me/inter/' target='_blank'>Inter</a> (OFL) · "
    "Peta dasar: © <a href='https://www.openstreetmap.org/copyright' target='_blank'>OpenStreetMap</a> contributors.</div>",
    unsafe_allow_html=True,
)
