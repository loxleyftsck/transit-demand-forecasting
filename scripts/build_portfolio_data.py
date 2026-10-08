"""Build a compact, static-only public results payload from local artifacts.

This deliberately exports aggregates and one final 24-hour example forecast,
not the row-level MTA history or the full 677,952-row benchmark table.
Run from the project root: python scripts/build_portfolio_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "portfolio-site" / "data" / "results.json"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso", double_precision=3))


def main() -> None:
    metrics = pd.read_csv(MODELS / "chronos_patchtst_benchmark_metrics.csv")
    horizon = pd.read_csv(MODELS / "chronos_patchtst_horizon_metrics.csv")
    forecast = pd.read_parquet(PROCESSED / "chronos_patchtst_predictions.parquet")
    hourly_eval = pd.read_parquet(PROCESSED / "predictions.parquet")
    locations = pd.read_parquet(PROCESSED / "station_locations.parquet")
    one_step = pd.read_csv(MODELS / "evaluation_metrics.csv")

    forecast["transit_timestamp"] = pd.to_datetime(forecast["transit_timestamp"])
    forecast["forecast_origin"] = pd.to_datetime(forecast["forecast_origin"])
    last_origin = forecast["forecast_origin"].max()
    example = forecast.loc[forecast["forecast_origin"].eq(last_origin)].copy()
    series_cols = [
        "entries", "pred_chronos2", "pred_patchtst", "pred_timesfm3",
        "pred_lgbm_recursive", "pred_weekly_naive",
    ]
    timeline = example.groupby("transit_timestamp", as_index=False)[series_cols].sum()
    timeline = timeline.sort_values("transit_timestamp")
    forecast_times = timeline["transit_timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist()
    forecast_fields = {
        "entries": "actual", "pred_chronos2": "chronos2", "pred_patchtst": "patchtst",
        "pred_timesfm3": "timesfm3", "pred_lgbm_recursive": "lightgbm", "pred_weekly_naive": "weekly_naive",
    }
    station_forecasts = []
    for station_id, group in example.sort_values("transit_timestamp").groupby("station_complex_id", sort=False):
        if len(group) != len(forecast_times):
            continue
        first = group.iloc[0]
        station_forecasts.append({
            "id": str(station_id), "name": str(first["station_complex"]), "borough": str(first["borough"]),
            **{public: group[column].astype(float).round(1).tolist() for column, public in forecast_fields.items()},
        })

    hourly_eval["transit_timestamp"] = pd.to_datetime(hourly_eval["transit_timestamp"])
    hourly_eval["hour"] = hourly_eval["transit_timestamp"].dt.hour
    hourly_profile = (
        hourly_eval.groupby("hour", as_index=False)["entries"].mean()
        .rename(columns={"entries": "mean_entries"})
    )

    station_totals = (
        forecast.groupby(["station_complex_id", "station_complex", "borough"], as_index=False)["entries"]
        .sum().rename(columns={"entries": "total_entries"})
    )
    stations = station_totals.merge(
        locations[["station_complex_id", "latitude", "longitude"]],
        on="station_complex_id", how="left", validate="one_to_one",
    )
    stations = stations.dropna(subset=["latitude", "longitude"])
    stations["total_entries"] = stations["total_entries"].round(0).astype("int64")
    station_list = records(stations.sort_values("total_entries", ascending=False))

    booster_path = MODELS / "lightgbm_final.txt"
    importance = []
    if booster_path.exists():
        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(booster_path))
        gains = pd.DataFrame({"feature": booster.feature_name(), "gain": booster.feature_importance(importance_type="gain")})
        gains = gains.loc[gains["gain"].gt(0)].sort_values("gain", ascending=False).head(10)
        gains["share_pct"] = gains["gain"] / booster.feature_importance(importance_type="gain").sum() * 100
        importance = records(gains[["feature", "share_pct"]])

    result = {
        "project": "Hourly Passenger Demand Forecasting",
        "dataset": "MTA Subway Hourly Ridership 2022–2024",
        "units_note": "Entries are recorded fare-gate entries, not unique people.",
        "evaluation": {
            "period_start": str(forecast["transit_timestamp"].min()),
            "period_end": str(forecast["transit_timestamp"].max()),
            "stations": int(forecast["station_complex_id"].nunique()),
            "scored_rows": int(len(forecast)),
            "origins": int(forecast["forecast_origin"].nunique()),
            "horizon_hours": int(forecast["horizon_step"].max()),
            "example_origin": str(last_origin),
            "model_eval_caveat": (
                "This is an exploratory benchmark on the project's previously examined test window, "
                "not a newly untouched holdout."
            ),
        },
        "benchmark_metrics": records(metrics),
        "horizon_metrics": records(horizon),
        "one_step_metrics": records(one_step),
        "example_forecast": records(timeline),
        "forecast_times": forecast_times,
        "station_forecasts": station_forecasts,
        "hourly_profile": records(hourly_profile),
        "stations": station_list,
        "feature_importance": importance,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size:,} bytes)")
    print(
        f"{result['evaluation']['stations']} stations · {result['evaluation']['scored_rows']:,} scored rows · "
        f"{result['evaluation']['origins']} origins · {len(timeline)} aggregate hourly points · "
        f"{len(station_list)} mapped stations"
    )


if __name__ == "__main__":
    main()
