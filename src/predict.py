"""
Tahap 9: Export model final via MLflow + generate prediksi untuk test set.

- Model final (models/lightgbm_final.txt, hasil Tahap 7) diformalkan sebagai
  MLflow model artifact (mlflow.lightgbm.log_model) supaya versinya tercatat
  rapi di tracking lokal yang sama dengan Tahap 6-7.
- Prediksi untuk seluruh periode test disimpan ke
  data/processed/predictions.parquet, dipakai langsung oleh dashboard
  (Tahap 10) supaya dashboard tidak perlu load model & re-infer setiap kali
  dibuka.
"""
import logging
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd

from baseline import naive_forecast, rolling_average_forecast
from train import FEATURE_COLS, TARGET_COL

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FINAL_MODEL_PATH = MODELS_DIR / "lightgbm_final.txt"
PREDICTIONS_PARQUET = PROCESSED_DIR / "predictions.parquet"

ID_COLS = ["station_complex_id", "station_complex", "borough", "transit_mode", "transit_timestamp"]


def export_model_to_mlflow(booster: lgb.Booster) -> str:
    mlflow.set_tracking_uri(f"file:{PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment("mta-ridership-forecasting")

    with mlflow.start_run(run_name="final_model_export"):
        model_info = mlflow.lightgbm.log_model(
            lgb_model=booster,
            artifact_path="model",
            registered_model_name=None,  # cukup tracking lokal, tanpa Model Registry penuh
        )
        mlflow.log_param("n_estimators", booster.num_trees())
        mlflow.log_param("feature_names", booster.feature_name())
    logger.info("Model diekspor ke MLflow: %s", model_info.model_uri)
    return model_info.model_uri


def generate_predictions(booster: lgb.Booster) -> pd.DataFrame:
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    out = test[ID_COLS + [TARGET_COL]].copy()
    out["pred_lgbm_tuned"] = booster.predict(test[FEATURE_COLS])
    out["pred_naive"] = naive_forecast(test)
    out["pred_rolling_avg_24h"] = rolling_average_forecast(test)

    return out


def run() -> None:
    logger.info("Load model final: %s", FINAL_MODEL_PATH)
    booster = lgb.Booster(model_file=str(FINAL_MODEL_PATH))

    export_model_to_mlflow(booster)

    logger.info("Generate prediksi untuk test set...")
    preds = generate_predictions(booster)
    preds.to_parquet(PREDICTIONS_PARQUET, index=False)

    logger.info("=== RINGKASAN EXPORT & PREDIKSI ===")
    logger.info("Baris prediksi   : %s", len(preds))
    logger.info("Rentang waktu    : %s s/d %s", preds["transit_timestamp"].min(), preds["transit_timestamp"].max())
    logger.info("Kolom            : %s", list(preds.columns))
    logger.info("Disimpan ke      : %s", PREDICTIONS_PARQUET)


if __name__ == "__main__":
    run()
