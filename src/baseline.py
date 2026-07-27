"""
Baseline forecast sederhana sebagai pembanding LightGBM (Tahap 7-8).

1. Naive forecast: prediksi entries pada jam t = entries aktual 168 jam
   sebelumnya (jam yang sama, minggu lalu). Sudah tersedia sebagai kolom
   `lag_168h` di features/train/val/test parquet.
2. Rolling average forecast: prediksi entries pada jam t = rata-rata entries
   24 jam sebelum t. Sudah tersedia sebagai kolom `rolling_mean_24h`.

Kedua baseline ini tidak butuh training -- tinggal dibaca dari kolom yang
sudah dihitung di src/features.py (dan sudah dipastikan bebas leakage, karena
memakai .shift() sebelum rolling/lag).
"""
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_COL = "entries"


def naive_forecast(df: pd.DataFrame) -> pd.Series:
    """Prediksi = entries jam yang sama, minggu lalu."""
    return df["lag_168h"]


def rolling_average_forecast(df: pd.DataFrame) -> pd.Series:
    """Prediksi = rata-rata entries 24 jam sebelumnya."""
    return df["rolling_mean_24h"]


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
    }


if __name__ == "__main__":
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    naive_metrics = evaluate(test[TARGET_COL], naive_forecast(test))
    rolling_metrics = evaluate(test[TARGET_COL], rolling_average_forecast(test))

    print("Baseline - Naive forecast (lag_168h)   :", naive_metrics)
    print("Baseline - Rolling average (24h)        :", rolling_metrics)
