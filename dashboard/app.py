"""
Dashboard interaktif — Hourly Passenger Demand Forecasting for Mass Transit Stations.

Jalankan dengan:
    streamlit run dashboard/app.py

Menampilkan:
  - Peta seluruh stasiun (ukuran/warna = volume ridership, stasiun terpilih di-highlight)
  - Filter stasiun & rentang waktu
  - Grafik aktual vs prediksi per jam (LightGBM tuned vs naive)
  - Ringkasan metrik (MAE/RMSE, baseline vs model)
  - Feature importance
  - Insight jam sibuk per stasiun (berdasar seluruh histori, bukan cuma test set)
"""
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

st.set_page_config(page_title="MTA Ridership Forecasting", layout="wide")


@st.cache_data
def load_predictions() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "predictions.parquet")


@st.cache_data
def load_cleaned() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "cleaned.parquet", columns=["station_complex", "transit_timestamp", "entries"])
    df["hour_of_day"] = df["transit_timestamp"].dt.hour
    return df


@st.cache_data
def load_metrics_table() -> pd.DataFrame:
    return pd.read_csv(MODELS_DIR / "evaluation_metrics.csv")


@st.cache_data
def load_station_locations() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "station_locations.parquet")


@st.cache_resource
def load_booster() -> lgb.Booster:
    return lgb.Booster(model_file=str(MODELS_DIR / "lightgbm_final.txt"))


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, float]:
    return mean_absolute_error(y_true, y_pred), mean_squared_error(y_true, y_pred) ** 0.5


preds = load_predictions()
cleaned = load_cleaned()
metrics_table = load_metrics_table()
booster = load_booster()
station_locations = load_station_locations()

st.title("Hourly Passenger Demand Forecasting for Mass Transit Stations")
st.caption(
    "Prediksi jumlah penumpang (entries) per stasiun per jam — MTA Subway, "
    "Staten Island Railway, & Tram (2022-2024)."
)

# --- Sidebar filter ---
st.sidebar.header("Filter")
station_list = sorted(preds["station_complex"].unique())
default_station = preds.groupby("station_complex", observed=True)["entries"].sum().idxmax()
station = st.sidebar.selectbox(
    "Stasiun", station_list, index=station_list.index(default_station)
)

min_date = preds["transit_timestamp"].min().date()
max_date = preds["transit_timestamp"].max().date()
date_range = st.sidebar.date_input(
    "Rentang waktu (test set)", value=(min_date, max_date), min_value=min_date, max_value=max_date
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

st.sidebar.caption(
    "Grafik aktual vs prediksi hanya tersedia untuk periode test "
    f"({min_date} s/d {max_date}), karena prediksi hanya dihasilkan di luar data training."
)

mask = (
    (preds["station_complex"] == station)
    & (preds["transit_timestamp"].dt.date >= start_date)
    & (preds["transit_timestamp"].dt.date <= end_date)
)
filtered = preds[mask].sort_values("transit_timestamp")

# --- Peta stasiun ---
st.subheader("Peta Mass Transit Stations")
st.caption("Ukuran & warna titik = total volume ridership (2022-2024). Stasiun terpilih ditandai titik merah besar.")

station_totals = (
    cleaned.groupby("station_complex", observed=True)["entries"].sum().reset_index(name="total_entries")
)
map_df = station_locations.merge(station_totals, on="station_complex", how="left")

fig_map = px.scatter_mapbox(
    map_df, lat="latitude", lon="longitude",
    size="total_entries", color="borough",
    hover_name="station_complex",
    hover_data={"total_entries": ":,", "latitude": False, "longitude": False, "borough": False},
    size_max=28, zoom=9.3, height=550,
    mapbox_style="open-street-map",
)
selected_row = map_df[map_df["station_complex"] == station]
fig_map.add_trace(go.Scattermapbox(
    lat=selected_row["latitude"], lon=selected_row["longitude"],
    mode="markers", marker=dict(size=26, color="red"),
    name="Stasiun terpilih", hoverinfo="skip",
))
fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.01))
st.plotly_chart(fig_map, use_container_width=True)

# --- Ringkasan metrik ---
st.subheader("Ringkasan Metrik")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Test set keseluruhan (semua stasiun)**")
    st.dataframe(metrics_table.set_index("model").style.format({"mae": "{:.2f}", "rmse": "{:.2f}"}))

with col2:
    st.markdown(f"**Stasiun terpilih: {station}** ({start_date} s/d {end_date})")
    if len(filtered) > 0:
        rows = []
        for name, col in [
            ("lgbm_tuned", "pred_lgbm_tuned"),
            ("naive", "pred_naive"),
            ("rolling_avg_24h", "pred_rolling_avg_24h"),
        ]:
            mae, rmse = compute_metrics(filtered["entries"], filtered[col])
            rows.append({"model": name, "mae": mae, "rmse": rmse})
        st.dataframe(pd.DataFrame(rows).set_index("model").style.format({"mae": "{:.2f}", "rmse": "{:.2f}"}))
    else:
        st.info("Tidak ada data untuk filter ini.")

# --- Grafik aktual vs prediksi ---
st.subheader("Aktual vs Prediksi per Jam")
if len(filtered) > 0:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered["transit_timestamp"], y=filtered["entries"], name="Aktual", mode="lines"))
    fig.add_trace(go.Scatter(
        x=filtered["transit_timestamp"], y=filtered["pred_lgbm_tuned"], name="LightGBM (tuned)", mode="lines"
    ))
    fig.add_trace(go.Scatter(
        x=filtered["transit_timestamp"], y=filtered["pred_naive"], name="Naive (minggu lalu)", mode="lines",
        opacity=0.5,
    ))
    fig.update_layout(
        template="plotly_white", height=450, xaxis_title="Waktu", yaxis_title="Entries",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Tidak ada data untuk filter ini.")

# --- Feature importance & insight jam sibuk ---
col3, col4 = st.columns(2)

with col3:
    st.subheader("Feature Importance (LightGBM tuned)")
    importance = pd.DataFrame({
        "feature": booster.feature_name(),
        "importance": booster.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=True)
    fig_imp = px.bar(
        importance, x="importance", y="feature", orientation="h",
        labels={"importance": "Importance (gain)", "feature": ""},
        template="plotly_white",
    )
    fig_imp.update_layout(height=400)
    st.plotly_chart(fig_imp, use_container_width=True)

with col4:
    st.subheader(f"Insight Jam Sibuk — {station}")
    st.caption("Berdasarkan seluruh histori data (2022-2024), bukan hanya periode test.")
    station_hourly = (
        cleaned[cleaned["station_complex"] == station]
        .groupby("hour_of_day")["entries"].mean()
        .reset_index()
    )
    fig_hour = px.bar(
        station_hourly, x="hour_of_day", y="entries",
        labels={"hour_of_day": "Jam", "entries": "Rata-rata Entries"},
        template="plotly_white",
    )
    fig_hour.update_layout(height=400)
    st.plotly_chart(fig_hour, use_container_width=True)

    peak_row = station_hourly.loc[station_hourly["entries"].idxmax()]
    st.metric(
        "Jam Puncak", f"{int(peak_row['hour_of_day']):02d}:00",
        f"{peak_row['entries']:.0f} entries/jam rata-rata",
    )
