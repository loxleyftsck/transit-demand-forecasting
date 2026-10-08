"""Paired 24-hour rolling-origin benchmark for TimesFM 3.0 vs local baselines.

The benchmark uses the same forecast origins, 428 station series, and 24-hour
targets for TimesFM, recursive LightGBM, and weekly seasonal naive. At each
origin, both learned models only receive observations strictly before it.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from timesfm3 import ModelConfig, TimesFM3Evaluator

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODEL_PATH = MODELS / "lightgbm_final.txt"
PANEL_PATH = PROCESSED / "cleaned.parquet"
FEATURES_PATH = PROCESSED / "features.parquet"
PREDICTIONS_PATH = PROCESSED / "timesfm3_benchmark_predictions.parquet"
METRICS_PATH = MODELS / "timesfm3_benchmark_metrics.csv"
HORIZON_METRICS_PATH = MODELS / "timesfm3_horizon_metrics.csv"
PLOT_PATH = MODELS / "timesfm3_benchmark_overlay.html"
METADATA_PATH = MODELS / "timesfm3_benchmark_metadata.json"

HORIZON = 24
CONTEXT_LENGTH = 512
TEST_START = pd.Timestamp("2024-03-15 00:00:00")
CHECKPOINT_REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"
FEATURE_COLS = [
    "lag_1h", "lag_24h", "lag_168h", "rolling_mean_3h", "rolling_mean_24h",
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "station_complex", "borough", "transit_mode",
]
CAT_COLS = ["station_complex", "borough", "transit_mode"]
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def get_forecaster() -> TimesFM3Evaluator:
    snapshots = get_checkpoint_snapshots()
    if not snapshots:
        raise FileNotFoundError(
            f"TimesFM checkpoint {CHECKPOINT_REVISION} not found under .hf-cache-timesfm3; "
            "run src/download_timesfm3.py first"
        )
    cfg = ModelConfig(
        checkpoint_path=str(snapshots[-1]),
        per_core_batch_size=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        local_files_only=True,
    )
    logger.info("Loading TimesFM 3.0 on %s", cfg.device)
    return TimesFM3Evaluator(cfg)


def get_checkpoint_snapshots() -> list[Path]:
    pinned = ROOT / ".hf-cache-timesfm3" / "models--google--timesfm-3.0-pytorch" / "snapshots" / CHECKPOINT_REVISION
    return [pinned] if pinned.is_dir() else []


def write_metadata(predictions: pd.DataFrame) -> None:
    snapshots = get_checkpoint_snapshots()
    if not snapshots:
        raise FileNotFoundError("TimesFM checkpoint revision could not be determined")
    origin_times = pd.to_datetime(predictions["forecast_origin"])
    target_times = pd.to_datetime(predictions["transit_timestamp"])
    metadata = {
        "model": "google/timesfm-3.0-pytorch",
        "package_version": "3.0.2",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "context_length": CONTEXT_LENGTH,
        "forecast_horizon_hours": HORIZON,
        "test_start": str(origin_times.min()),
        "test_end_evaluated": str(target_times.max()),
        "origins": int(origin_times.nunique()),
        "stations": int(predictions["station_complex_id"].nunique()),
        "scored_rows": len(predictions),
        "notes": "Non-overlapping 24-hour forecasts; actual values strictly before each forecast origin only. Last incomplete test hours are omitted. Existing one-step LightGBM metrics are not directly comparable to these 24-hour recursive forecasts.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def get_lightgbm_rows(station_ids: list[str], features: pd.DataFrame) -> pd.DataFrame:
    meta = (
        features.sort_values("transit_timestamp")
        .drop_duplicates("station_complex_id")
        .set_index("station_complex_id")
        .reindex(station_ids)
    )
    if meta[CAT_COLS].isna().any().any():
        raise ValueError("Missing station categorical metadata for LightGBM recursion")
    for col in CAT_COLS:
        meta[col] = pd.Categorical(meta[col], categories=features[col].cat.categories)
    return meta.reset_index()


def make_recursive_features(
    station_meta: pd.DataFrame,
    values: np.ndarray,
    target_time: pd.Timestamp,
) -> pd.DataFrame:
    """Create one-step LightGBM features from history available before target."""
    n_stations, n_history = values.shape
    if n_history < 168:
        raise ValueError("Need at least 168 observed/predicted hours in context")
    dow = target_time.dayofweek
    out = station_meta[["station_complex_id", "station_complex", "borough", "transit_mode"]].copy()
    out["lag_1h"] = values[:, -1]
    out["lag_24h"] = values[:, -24]
    out["lag_168h"] = values[:, -168]
    out["rolling_mean_3h"] = values[:, -3:].mean(axis=1)
    out["rolling_mean_24h"] = values[:, -24:].mean(axis=1)
    out["hour_of_day"] = target_time.hour
    out["day_of_week"] = dow
    out["month"] = target_time.month
    out["is_weekend"] = dow >= 5
    for col in CAT_COLS:
        out[col] = pd.Categorical(out[col], categories=station_meta[col].cat.categories)
    if len(out) != n_stations:
        raise AssertionError("Station metadata and panel do not align")
    return out[FEATURE_COLS]


def get_metrics(frame: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for model, col in [
        ("TimesFM 3.0", "pred_timesfm3"),
        ("LightGBM recursive", "pred_lgbm_recursive"),
        ("Weekly seasonal naive", "pred_weekly_naive"),
    ]:
        y, pred = frame["entries"].to_numpy(), frame[col].to_numpy()
        rows.append({
            "model": model,
            "mae": mean_absolute_error(y, pred),
            "rmse": mean_squared_error(y, pred) ** 0.5,
            "macro_station_mae": frame.assign(_ae=np.abs(y - pred)).groupby("station_complex_id", observed=True)["_ae"].mean().mean(),
            "rows": len(frame),
            "stations": frame["station_complex_id"].nunique(),
            "horizon_hours": HORIZON,
        })
    return rows


def run(max_origins: int | None = None) -> None:
    logger.info("Read complete cleaned hourly panel: %s", PANEL_PATH)
    panel_df = pd.read_parquet(PANEL_PATH, columns=["transit_timestamp", "station_complex_id", "station_complex", "borough", "transit_mode", "entries"])
    panel_df["transit_timestamp"] = pd.to_datetime(panel_df["transit_timestamp"])
    panel_df["station_complex_id"] = panel_df["station_complex_id"].astype(str)
    features = pd.read_parquet(FEATURES_PATH, columns=["station_complex_id", *CAT_COLS, "transit_timestamp"])
    features["station_complex_id"] = features["station_complex_id"].astype(str)
    panel = panel_df.pivot(index="station_complex_id", columns="transit_timestamp", values="entries").sort_index(axis=1)
    station_ids = panel.index.astype(str).tolist()
    timestamps = pd.DatetimeIndex(panel.columns)
    values = panel.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Panel has missing/non-finite station-hour values; do not silently impute in benchmark")
    if len(timestamps) < CONTEXT_LENGTH + HORIZON:
        raise ValueError("Dataset is shorter than required history + forecast horizon")
    if not (timestamps.to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("Global panel is not a continuous hourly timeline")

    start_idx = int(timestamps.searchsorted(TEST_START))
    origins = list(range(start_idx, len(timestamps) - HORIZON + 1, HORIZON))
    if max_origins is not None:
        origins = origins[:max_origins]
    if not origins:
        raise ValueError("No complete 24-hour test windows available")
    station_meta = get_lightgbm_rows(station_ids, features)
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    forecaster = get_forecaster()
    logger.info("Panel: %s hours x %s stations; test windows: %s; forecast horizon: %sh", len(timestamps), len(station_ids), len(origins), HORIZON)

    output: list[pd.DataFrame] = []
    for origin_idx in origins:
        origin = timestamps[origin_idx]
        context = values[:, max(0, origin_idx - CONTEXT_LENGTH):origin_idx]
        if context.shape[1] != CONTEXT_LENGTH:
            continue
        fm_result = next(forecaster.predict_batch(
            [context], horizon=HORIZON, return_quantiles=False,
            use_symmetric_averaging=False, univariate=False,
        ))
        fm_prediction = np.maximum(fm_result.forecast, 0)
        lgb_history = context.copy()
        lgb_predictions = np.empty((len(station_ids), HORIZON), dtype=np.float32)
        for step in range(HORIZON):
            target_time = timestamps[origin_idx + step]
            X = make_recursive_features(station_meta, lgb_history, target_time)
            one_step = np.maximum(booster.predict(X), 0).astype(np.float32)
            lgb_predictions[:, step] = one_step
            lgb_history = np.concatenate([lgb_history, one_step[:, None]], axis=1)
        target_values = values[:, origin_idx:origin_idx + HORIZON]
        weekly_naive = values[:, origin_idx - 168:origin_idx - 168 + HORIZON]
        index = pd.MultiIndex.from_product([station_ids, timestamps[origin_idx:origin_idx + HORIZON]], names=["station_complex_id", "transit_timestamp"])
        block = pd.DataFrame({
            "station_complex_id": index.get_level_values(0),
            "transit_timestamp": index.get_level_values(1),
            "entries": target_values.reshape(-1),
            "forecast_origin": origin,
            "horizon_step": np.tile(np.arange(1, HORIZON + 1), len(station_ids)),
            "pred_timesfm3": fm_prediction.reshape(-1),
            "pred_lgbm_recursive": lgb_predictions.reshape(-1),
            "pred_weekly_naive": weekly_naive.reshape(-1),
        })
        output.append(block)
        logger.info("Origin %s done (%s/%s)", origin, len(output), len(origins))

    predictions = pd.concat(output, ignore_index=True)
    name_map = panel_df[["station_complex_id", "station_complex", "borough"]].drop_duplicates("station_complex_id")
    predictions = predictions.merge(name_map, on="station_complex_id", how="left", validate="many_to_one")
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(PREDICTIONS_PATH, index=False)
    metrics = pd.DataFrame(get_metrics(predictions))
    metrics.to_csv(METRICS_PATH, index=False)
    horizon_rows = []
    for horizon_step, part in predictions.groupby("horizon_step", sort=True):
        for model_name, prediction_col in [
            ("TimesFM 3.0", "pred_timesfm3"),
            ("LightGBM recursive", "pred_lgbm_recursive"),
            ("Weekly seasonal naive", "pred_weekly_naive"),
        ]:
            y, pred = part["entries"].to_numpy(), part[prediction_col].to_numpy()
            horizon_rows.append({
                "horizon_step": int(horizon_step),
                "model": model_name,
                "mae": mean_absolute_error(y, pred),
                "rmse": mean_squared_error(y, pred) ** 0.5,
                "rows": len(part),
            })
    pd.DataFrame(horizon_rows).to_csv(HORIZON_METRICS_PATH, index=False)

    fig = go.Figure()
    hourly = predictions.groupby("transit_timestamp", as_index=False)[["entries", "pred_timesfm3", "pred_lgbm_recursive", "pred_weekly_naive"]].sum()
    for col, name in [("entries", "Actual"), ("pred_timesfm3", "TimesFM 3.0"), ("pred_lgbm_recursive", "LightGBM recursive"), ("pred_weekly_naive", "Weekly seasonal naive")]:
        fig.add_trace(go.Scatter(x=hourly["transit_timestamp"], y=hourly[col], mode="lines", name=name))
    fig.update_layout(title="MTA Ridership: paired 24-hour rolling-origin benchmark", xaxis_title="Target hour", yaxis_title="Entries (all stations)", hovermode="x unified")
    fig.write_html(PLOT_PATH, include_plotlyjs="cdn")

    write_metadata(predictions)
    logger.info("Metrics:\n%s", metrics.to_string(index=False))
    logger.info("Saved predictions %s; metrics %s; horizon metrics %s; chart %s", PREDICTIONS_PATH, METRICS_PATH, HORIZON_METRICS_PATH, PLOT_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-origins", type=int, default=None, help="Limit test origins for a smoke run")
    parser.add_argument("--finalize-existing", action="store_true", help="Write metadata from already-generated predictions without rerunning forecasts")
    args = parser.parse_args()
    if args.finalize_existing:
        write_metadata(pd.read_parquet(PREDICTIONS_PATH))
        logger.info("Metadata finalized: %s", METADATA_PATH)
    else:
        run(args.max_origins)
