"""
Evaluasi model (Tahap 8): bandingkan baseline naive, LightGBM sebelum tuning
(default params), dan LightGBM sesudah tuning (Tahap 6-7) di test set.

Output:
  - models/evaluation_metrics.csv  (tabel MAE/RMSE semua model)
  - models/actual_vs_predicted.png (time-series overlay)
  - models/feature_importance.png  (feature importance LightGBM tuned)

Catatan: chart statis di sini dibuat dengan matplotlib, BUKAN Plotly, karena
`fig.write_image()` Plotly butuh subprocess Chromium (via kaleido) yang macet
tanpa batas waktu di environment ini (sandbox tidak mengizinkan Chromium
headless jalan penuh). Dashboard interaktif (Tahap 10) tetap memakai Plotly
seperti biasa karena Streamlit merender langsung di browser tanpa perlu
export gambar statis.
"""
import json
import logging
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from baseline import naive_forecast, rolling_average_forecast
from train import CATEGORICAL_COLS, FEATURE_COLS, TARGET_COL

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_PARAMS_PATH = MODELS_DIR / "best_params.json"
FINAL_MODEL_PATH = MODELS_DIR / "lightgbm_final.txt"

METRICS_CSV = MODELS_DIR / "evaluation_metrics.csv"
ACTUAL_VS_PRED_PNG = MODELS_DIR / "actual_vs_predicted.png"
FEATURE_IMPORTANCE_PNG = MODELS_DIR / "feature_importance.png"


def evaluate(y_true, y_pred) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
    }


def train_untuned_baseline_model() -> lgb.LGBMRegressor:
    """LightGBM dengan hyperparameter DEFAULT (tanpa Optuna) sebagai pembanding
    'sebelum tuning'. Dilatih di atas train+val yang sama dengan model final
    supaya perbandingan adil (hanya hyperparameter yang berbeda)."""
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    train_val = pd.concat([train, val], ignore_index=True)

    model = lgb.LGBMRegressor(n_jobs=-1, verbose=-1, random_state=42)
    model.fit(
        train_val[FEATURE_COLS], train_val[TARGET_COL],
        categorical_feature=CATEGORICAL_COLS,
    )
    return model


def plot_actual_vs_predicted(test: pd.DataFrame, preds: dict) -> None:
    top_station = test.groupby("station_complex", observed=True)["entries"].sum().idxmax()
    sub = test[test["station_complex"] == top_station].sort_values("transit_timestamp")
    idx = sub.index

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sub["transit_timestamp"], sub["entries"], label="Aktual", linewidth=1.5)
    ax.plot(sub["transit_timestamp"], preds["lgbm_tuned"].loc[idx], label="LightGBM (tuned)", linewidth=1.2)
    ax.plot(sub["transit_timestamp"], preds["naive"].loc[idx], label="Naive (minggu lalu)", linewidth=1, alpha=0.6)
    ax.set_title(f"Aktual vs Prediksi — {top_station} (Test Set)")
    ax.set_xlabel("Waktu")
    ax.set_ylabel("Entries")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(ACTUAL_VS_PRED_PNG, dpi=150)
    plt.close(fig)
    logger.info("Chart aktual vs prediksi disimpan ke: %s", ACTUAL_VS_PRED_PNG)


def plot_feature_importance(model: lgb.Booster) -> None:
    importance = pd.DataFrame({
        "feature": model.feature_name(),
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(importance["feature"], importance["importance"])
    ax.set_title("Feature Importance — LightGBM (tuned), by gain")
    ax.set_xlabel("Importance (gain)")
    fig.tight_layout()
    fig.savefig(FEATURE_IMPORTANCE_PNG, dpi=150)
    plt.close(fig)
    logger.info("Chart feature importance disimpan ke: %s", FEATURE_IMPORTANCE_PNG)


def run() -> None:
    logger.info("Load test set...")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    y_true = test[TARGET_COL]

    logger.info("Load model tuned final: %s", FINAL_MODEL_PATH)
    tuned_booster = lgb.Booster(model_file=str(FINAL_MODEL_PATH))

    logger.info("Melatih LightGBM default (sebelum tuning) sebagai pembanding...")
    untuned_model = train_untuned_baseline_model()

    preds = {
        "naive": naive_forecast(test),
        "rolling_avg_24h": rolling_average_forecast(test),
        "lgbm_untuned": pd.Series(
            untuned_model.predict(test[FEATURE_COLS]), index=test.index
        ),
        "lgbm_tuned": pd.Series(
            tuned_booster.predict(test[FEATURE_COLS]), index=test.index
        ),
    }

    results = []
    for name, pred in preds.items():
        m = evaluate(y_true, pred)
        m["model"] = name
        results.append(m)
        logger.info("%-16s -> MAE=%.3f | RMSE=%.3f", name, m["mae"], m["rmse"])

    results_df = pd.DataFrame(results)[["model", "mae", "rmse"]].sort_values("mae")
    results_df.to_csv(METRICS_CSV, index=False)
    logger.info("Tabel evaluasi disimpan ke: %s", METRICS_CSV)

    plot_actual_vs_predicted(test, preds)
    plot_feature_importance(tuned_booster)

    with open(BEST_PARAMS_PATH) as f:
        best_params = json.load(f)

    logger.info("=== RINGKASAN EVALUASI (test set) ===")
    logger.info("\n%s", results_df.to_string(index=False))
    improvement_vs_naive = 100 * (1 - results_df.set_index("model").loc["lgbm_tuned", "mae"] / results_df.set_index("model").loc["naive", "mae"])
    logger.info("LightGBM (tuned) lebih baik %.1f%% dari baseline naive (MAE).", improvement_vs_naive)
    logger.info("Params model tuned: %s", best_params)


if __name__ == "__main__":
    run()
