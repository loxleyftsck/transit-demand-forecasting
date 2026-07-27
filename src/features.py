"""
Feature engineering untuk forecasting entries per stasiun per jam.

Pipeline:
  1. Load data/processed/cleaned.parquet (Polars).
  2. Fitur lag: lag_1h, lag_24h, lag_168h (per stasiun, urut waktu).
  3. Fitur rolling: rolling_mean_3h, rolling_mean_24h -- dihitung HANYA dari jam
     sebelum t (shift dulu baru rolling) supaya tidak ada leakage dari nilai
     entries di jam yang sedang diprediksi.
  4. Fitur kalender: hour_of_day, day_of_week (0=Senin..6=Minggu), is_weekend, month.
  5. Fitur kategorikal: station_complex, borough, transit_mode.
  6. Drop baris dengan lag NaN (jam-jam awal per stasiun, sebanyak lag terpanjang
     yaitu 168 jam, karena belum ada histori 168 jam ke belakang).
  7. Convert ke pandas (LightGBM/scikit-learn tidak native support Polars),
     set kolom kategorikal ke dtype `category`.
  8. Simpan ke data/processed/features.parquet.

CATATAN desain:
  - `transfers` (kolom mentah di cleaned.parquet) TIDAK dipakai sebagai fitur
    di sini karena nilainya tercatat pada jam yang sama dengan target `entries`
    -- memakainya sebagai fitur akan menjadi data leakage (informasi masa
    depan relatif terhadap titik waktu prediksi).
"""
import logging
from pathlib import Path

import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_PARQUET = PROJECT_ROOT / "data" / "processed" / "cleaned.parquet"
OUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "features.parquet"

LAG_HOURS = [1, 24, 168]
ROLLING_WINDOWS = [3, 24]
CATEGORICAL_COLS = ["station_complex", "borough", "transit_mode"]


def add_lag_and_rolling_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(["station_complex_id", "transit_timestamp"])

    lag_exprs = [
        pl.col("entries").shift(h).over("station_complex_id").alias(f"lag_{h}h")
        for h in LAG_HOURS
    ]
    df = df.with_columns(lag_exprs)

    rolling_exprs = [
        pl.col("entries")
        .shift(1)
        .rolling_mean(window_size=w)
        .over("station_complex_id")
        .alias(f"rolling_mean_{w}h")
        for w in ROLLING_WINDOWS
    ]
    df = df.with_columns(rolling_exprs)

    return df


def add_calendar_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        [
            pl.col("transit_timestamp").dt.hour().alias("hour_of_day"),
            (pl.col("transit_timestamp").dt.weekday() - 1).alias("day_of_week"),
            pl.col("transit_timestamp").dt.month().alias("month"),
        ]
    )
    df = df.with_columns((pl.col("day_of_week") >= 5).alias("is_weekend"))
    return df


def run() -> None:
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Load %s ...", CLEANED_PARQUET)
    df = pl.read_parquet(CLEANED_PARQUET)
    n_rows_in = df.height

    df = add_lag_and_rolling_features(df)
    df = add_calendar_features(df)
    df = df.drop("transfers")

    max_lag = max(LAG_HOURS)
    lag_cols = [f"lag_{h}h" for h in LAG_HOURS]
    before_dropna = df.height
    df = df.drop_nulls(subset=lag_cols)
    n_dropped = before_dropna - df.height
    logger.info(
        "Drop %s baris dengan lag NaN (jam awal per stasiun, maks lag=%sh x %s stasiun).",
        n_dropped, max_lag, df["station_complex_id"].n_unique(),
    )

    pdf = df.to_pandas()
    for col in CATEGORICAL_COLS:
        pdf[col] = pdf[col].astype("category")

    pdf.to_parquet(OUT_PARQUET, index=False)

    logger.info("=== RINGKASAN FEATURE ENGINEERING ===")
    logger.info("Baris input (cleaned.parquet)   : %s", n_rows_in)
    logger.info("Baris dibuang (lag NaN)          : %s", n_dropped)
    logger.info("Baris output (features.parquet)  : %s", len(pdf))
    logger.info("Kolom akhir: %s", list(pdf.columns))
    logger.info("Disimpan ke: %s", OUT_PARQUET)


if __name__ == "__main__":
    run()
