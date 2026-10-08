"""Paired 24-hour benchmark for Chronos-2 and supervised PatchTST.

Chronos-2 forecasts the full profiled station panel zero-shot. PatchTST is trained on
pre-test data with a trailing 168-hour validation window. Both receive the
same 512 hours of context at each of the existing 66 daily test origins.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from chronos import Chronos2Pipeline
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
CHRONOS_CACHE = ROOT / ".hf-cache-chronos2"
PATCHTST_CACHE = ROOT / ".patchtst-checkpoints"
PANEL_PATH = PROCESSED / "cleaned.parquet"
BASELINE_PATH = PROCESSED / "timesfm3_benchmark_predictions.parquet"
PREDICTIONS_PATH = PROCESSED / "chronos_patchtst_predictions.parquet"
METRICS_PATH = MODELS / "chronos_patchtst_benchmark_metrics.csv"
HORIZON_METRICS_PATH = MODELS / "chronos_patchtst_horizon_metrics.csv"
PLOT_PATH = MODELS / "chronos_patchtst_benchmark_overlay.html"
METADATA_PATH = MODELS / "chronos_patchtst_benchmark_metadata.json"

CHRONOS_REPO = "amazon/chronos-2"
CHRONOS_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
HORIZON = 24
CONTEXT_LENGTH = 512
VALIDATION_HOURS = 168
MODEL_COLUMNS = {
    "TimesFM 3.0": "pred_timesfm3",
    "Chronos-2": "pred_chronos2",
    "PatchTST": "pred_patchtst",
    "LightGBM recursive": "pred_lgbm_recursive",
    "Weekly seasonal naive": "pred_weekly_naive",
}
COLORS = {
    "TimesFM 3.0": "#2a78d6",
    "Chronos-2": "#df6c36",
    "PatchTST": "#17846c",
    "LightGBM recursive": "#141B34",
    "Weekly seasonal naive": "#898781",
}
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def metrics_for(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, column in MODEL_COLUMNS.items():
        err = frame["entries"].to_numpy() - frame[column].to_numpy()
        frame_err = frame[["station_complex_id"]].copy()
        frame_err["absolute_error"] = np.abs(err)
        rows.append({
            "model": model,
            "mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt(np.square(err).mean())),
            "macro_station_mae": float(frame_err.groupby("station_complex_id")["absolute_error"].mean().mean()),
            "rows": len(frame),
            "stations": frame["station_complex_id"].nunique(),
            "origins": frame["forecast_origin"].nunique(),
            "horizon_hours": HORIZON,
        })
    return pd.DataFrame(rows)


def make_plot(frame: pd.DataFrame) -> None:
    hourly = frame.groupby("transit_timestamp", as_index=False)[list(MODEL_COLUMNS.values())].sum()
    fig = go.Figure()
    for model, column in MODEL_COLUMNS.items():
        fig.add_trace(go.Scatter(
            x=hourly["transit_timestamp"], y=hourly[column], mode="lines", name=model,
            line=dict(color=COLORS[model], width=2 if model in ("Chronos-2", "PatchTST") else 1.5),
        ))
    fig.update_layout(
        title="MTA: paired 24-hour rolling-origin model benchmark",
        xaxis_title="Target hour", yaxis_title="Entries (all stations)", hovermode="x unified",
    )
    fig.write_html(PLOT_PATH, include_plotlyjs="cdn")


def run(max_origins: int | None = None, max_steps: int = 500) -> None:
    if not BASELINE_PATH.exists():
        raise FileNotFoundError("Run src/benchmark_timesfm3.py first to create the common origins/baselines")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device=%s torch=%s CUDA=%s", device, torch.__version__, torch.version.cuda)
    base = pd.read_parquet(BASELINE_PATH)
    base["station_complex_id"] = base["station_complex_id"].astype(str)
    base["transit_timestamp"] = pd.to_datetime(base["transit_timestamp"])
    base["forecast_origin"] = pd.to_datetime(base["forecast_origin"])
    origins = sorted(base["forecast_origin"].unique())
    if max_origins is not None:
        origins = origins[:max_origins]

    logger.info("Load complete hourly panel for %s forecast origins", len(origins))
    raw = pd.read_parquet(PANEL_PATH, columns=["transit_timestamp", "station_complex_id", "entries"])
    raw["transit_timestamp"] = pd.to_datetime(raw["transit_timestamp"])
    raw["station_complex_id"] = raw["station_complex_id"].astype(str)
    panel = raw.pivot(index="station_complex_id", columns="transit_timestamp", values="entries").sort_index(axis=1)
    station_ids = panel.index.astype(str).tolist()
    timestamps = pd.DatetimeIndex(panel.columns)
    values = panel.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Hourly station panel has missing or non-finite values")
    logger.info("Using all %s profiled stations; no station filter is applied", len(station_ids))
    if not (timestamps.to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("Input panel must be a complete, continuous hourly timeline")

    # Train PatchTST only before the held-out test period. The final 168 hours
    # are reserved for early stopping; no test target is used for fitting.
    test_start = pd.Timestamp(origins[0])
    train_end_idx = int(timestamps.searchsorted(test_start))
    train_ids = np.repeat(np.asarray(station_ids, dtype=object), train_end_idx)
    train_times = np.tile(timestamps[:train_end_idx].to_numpy(), len(station_ids))
    train_values = values[:, :train_end_idx].reshape(-1)
    train_df = pd.DataFrame({"unique_id": train_ids, "ds": train_times, "y": train_values})
    logger.info("PatchTST fit rows=%s, validation tail=%sh, max_steps=%s", len(train_df), VALIDATION_HOURS, max_steps)

    patch_model = PatchTST(
        h=HORIZON,
        input_size=CONTEXT_LENGTH,
        patch_len=32,
        stride=16,
        max_steps=max_steps,
        learning_rate=1e-3,
        early_stop_patience_steps=4,
        val_check_steps=50,
        batch_size=32,
        valid_batch_size=32,
        windows_batch_size=512,
        inference_windows_batch_size=512,
        scaler_type="robust",
        random_seed=42,
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
    )
    patch_nf = NeuralForecast(models=[patch_model], freq="h")
    patch_nf.fit(df=train_df, val_size=VALIDATION_HOURS, verbose=False)
    PATCHTST_CACHE.mkdir(parents=True, exist_ok=True)
    patch_nf.save(str(PATCHTST_CACHE / "patchtst_24h"), save_dataset=False, overwrite=True)

    logger.info("Load Chronos-2 %s", CHRONOS_REVISION)
    chronos = Chronos2Pipeline.from_pretrained(
        CHRONOS_REPO,
        revision=CHRONOS_REVISION,
        device_map=device,
        cache_dir=str(CHRONOS_CACHE),
    )

    results = []
    for origin_num, origin in enumerate(origins, start=1):
        origin = pd.Timestamp(origin)
        idx = int(timestamps.searchsorted(origin))
        context = values[:, idx - CONTEXT_LENGTH:idx]
        if context.shape != (len(station_ids), CONTEXT_LENGTH):
            raise ValueError(f"Insufficient common context before {origin}")

        chronos_quantiles, chronos_median = chronos.predict_quantiles(
            inputs=[torch.from_numpy(context)],
            prediction_length=HORIZON,
            quantile_levels=[0.5],
            batch_size=512,
            context_length=CONTEXT_LENGTH,
            cross_learning=True,
        )
        pred_chronos = np.maximum(chronos_median[0].detach().cpu().numpy(), 0)

        context_df = pd.DataFrame({
            "unique_id": np.repeat(np.asarray(station_ids, dtype=object), CONTEXT_LENGTH),
            "ds": np.tile(timestamps[idx - CONTEXT_LENGTH:idx].to_numpy(), len(station_ids)),
            "y": context.reshape(-1),
        })
        patch_frame = patch_nf.predict(df=context_df, verbose=False)
        patch_frame["unique_id"] = patch_frame["unique_id"].astype(str)
        patch_frame["ds"] = pd.to_datetime(patch_frame["ds"])
        pred_patch = (
            patch_frame.pivot(index="unique_id", columns="ds", values="PatchTST")
            .reindex(index=station_ids, columns=timestamps[idx:idx + HORIZON])
            .to_numpy(dtype=np.float32)
        )
        if pred_chronos.shape != pred_patch.shape or pred_patch.shape != (len(station_ids), HORIZON):
            raise ValueError(f"Model output shape mismatch at {origin}: {pred_chronos.shape}, {pred_patch.shape}")

        block = base[base["forecast_origin"] == origin].copy()
        index = pd.MultiIndex.from_product(
            [station_ids, timestamps[idx:idx + HORIZON]], names=["station_complex_id", "transit_timestamp"]
        )
        generated = pd.DataFrame({
            "station_complex_id": index.get_level_values("station_complex_id"),
            "transit_timestamp": index.get_level_values("transit_timestamp"),
            "pred_chronos2": pred_chronos.reshape(-1),
            "pred_patchtst": np.maximum(pred_patch, 0).reshape(-1),
        })
        block = block.merge(generated, on=["station_complex_id", "transit_timestamp"], how="left", validate="one_to_one")
        if block[["pred_chronos2", "pred_patchtst"]].isna().any().any():
            raise ValueError(f"Missing predictions at {origin}")
        results.append(block)
        logger.info("Origin %s completed (%s/%s)", origin, origin_num, len(origins))

    predictions = pd.concat(results, ignore_index=True)
    predictions.to_parquet(PREDICTIONS_PATH, index=False)
    metrics = metrics_for(predictions)
    metrics.to_csv(METRICS_PATH, index=False)

    horizon_rows = []
    for step, part in predictions.groupby("horizon_step", sort=True):
        for model, column in MODEL_COLUMNS.items():
            error = part["entries"].to_numpy() - part[column].to_numpy()
            horizon_rows.append({
                "horizon_step": int(step), "model": model,
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt(np.square(error).mean())),
                "rows": len(part),
            })
    pd.DataFrame(horizon_rows).to_csv(HORIZON_METRICS_PATH, index=False)
    make_plot(predictions)
    metadata = {
        "chronos_model": CHRONOS_REPO,
        "chronos_revision": CHRONOS_REVISION,
        "chronos_package_version": "2.3.2",
        "neuralforecast_version": "3.2.2",
        "patchtst_training": {
            "input_size_hours": CONTEXT_LENGTH,
            "horizon_hours": HORIZON,
            "max_steps": max_steps,
            "validation_hours": VALIDATION_HOURS,
            "training_cutoff_exclusive": str(test_start),
            "note": "PatchTST trained only on pre-test data; last 168 pre-test hours held out for early stopping.",
        },
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": device,
        "context_length": CONTEXT_LENGTH,
        "forecast_horizon_hours": HORIZON,
        "test_start": str(predictions["transit_timestamp"].min()),
        "test_end_evaluated": str(predictions["transit_timestamp"].max()),
        "origins": int(predictions["forecast_origin"].nunique()),
        "stations": int(predictions["station_complex_id"].nunique()),
        "scored_rows": len(predictions),
        "notes": "Same non-overlapping 24-hour origins, station set, context length, target timestamps, and weekly-naive/recursive-LightGBM/TimesFM predictions as the prior benchmark. Chronos-2 predicts all station variates jointly; PatchTST is a globally trained channel-independent model. Existing test period was already examined in prior project iterations; this is exploratory, not a new untouched holdout.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Metrics:\n%s", metrics.to_string(index=False))
    logger.info("Saved %s", PREDICTIONS_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-origins", type=int, default=None, help="Limit forecast origins for a short smoke run")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum PatchTST optimizer steps; validation early stopping applies")
    args = parser.parse_args()
    run(max_origins=args.max_origins, max_steps=args.max_steps)
