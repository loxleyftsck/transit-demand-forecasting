"""
Time-based train/validation/test split untuk data/processed/features.parquet.

Split HARUS berbasis waktu (bukan random) karena ini adalah forecasting
time-series -- data test/validation harus selalu berada setelah data train
secara kronologis, supaya evaluasi mencerminkan skenario nyata (prediksi ke
masa depan berdasarkan histori).

Target proporsi: train ~80%, validation ~10%, test ~10% dari rentang waktu
AKTUAL di features.parquet (bukan angka tetap seperti "2 minggu").

CATATAN pengecekan anomali (wajib sebelum finalisasi, lihat brief Tahap 5):
  Titik potong 80% murni jatuh di 2023-12-26 13:00, yaitu SEHARI SETELAH Natal.
  Saat dicek agregat harian, terlihat penurunan tajam ridership akibat libur:
      2023-12-24 : 1.682.459   (Christmas Eve)
      2023-12-25 : 1.304.197   (Christmas Day)
      2023-12-31 : 1.961.017   (New Year's Eve)
      2024-01-01 : 1.672.635   (New Year's Day)
  vs hari kerja normal di sekitarnya (~3.3-3.9 juta/hari). Kalau batas
  train/validation dibiarkan di 2023-12-26, beberapa hari pertama validation
  akan berada di periode transisi pasca-libur yang belum sepenuhnya normal,
  membuat metrik tuning bias terhadap periode anomali ini.

  KEPUTUSAN: batas train/validation DIGESER ke 2024-01-02 00:00:00 (Selasa
  pertama setelah libur Tahun Baru, ridership sudah kembali normal ~3,3 juta/
  hari) supaya seluruh periode libur Natal & Tahun Baru masuk ke training
  (model tetap belajar pola libur ini), dan validation set mulai dari periode
  yang representatif/normal. Akibatnya proporsi bergeser sedikit dari 80/10/10
  murni menjadi ~80.9% / 10.0% / 9.1% -- deviasi kecil ini disengaja.

  Batas validation/test (~2024-03-14/15) sudah dicek dan TIDAK menunjukkan
  anomali (hanya pola mingguan biasa: dip di Sabtu-Minggu, weekday normal),
  sehingga tidak digeser.

  Ujung test set (2024-05-20 08:00) adalah batas akhir dataset itu sendiri
  (hari terakhir hanya terekam 9 jam pertama) -- bukan anomali, melainkan
  cutoff alami dataset, sehingga dibiarkan apa adanya.

Validasi no-leakage: fitur lag/rolling di features.parquet sudah dihitung di
atas seluruh deret waktu yang kontinu SEBELUM displit (lihat src/features.py,
memakai `.shift()` per stasiun yang hanya menghitung dari histori sebelum t).
Split ini hanya mempartisi baris berdasarkan `transit_timestamp` target,
sehingga tidak menambah leakage baru: setiap baris test/validation memakai
lag yang berasal dari jam-jam sebelumnya (train ataupun validation), sama
seperti kondisi nyata saat model dipakai untuk forecasting ke depan.
"""
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PARQUET = PROJECT_ROOT / "data" / "processed" / "features.parquet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_END = pd.Timestamp("2024-01-02 00:00:00")
VAL_END = pd.Timestamp("2024-03-15 00:00:00")


def time_based_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["transit_timestamp"] < TRAIN_END]
    val = df[(df["transit_timestamp"] >= TRAIN_END) & (df["transit_timestamp"] < VAL_END)]
    test = df[df["transit_timestamp"] >= VAL_END]
    return train, val, test


def validate_no_leakage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    assert train["transit_timestamp"].max() < val["transit_timestamp"].min(), "Train tumpang tindih dengan val!"
    assert val["transit_timestamp"].max() < test["transit_timestamp"].min(), "Val tumpang tindih dengan test!"
    logger.info("Validasi no-leakage OK: train < val < test secara kronologis, tanpa overlap.")


def run() -> None:
    logger.info("Load %s ...", FEATURES_PARQUET)
    df = pd.read_parquet(FEATURES_PARQUET)

    tmin, tmax = df["transit_timestamp"].min(), df["transit_timestamp"].max()
    total_hours = (tmax - tmin).total_seconds() / 3600 + 1

    train, val, test = time_based_split(df)
    validate_no_leakage(train, val, test)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)

    logger.info("=== RINGKASAN SPLIT (time-based) ===")
    logger.info("Rentang total   : %s s/d %s (%.0f jam)", tmin, tmax, total_hours)
    logger.info(
        "Train : %s s/d %s | %s baris (%.1f%%)",
        tmin, TRAIN_END, len(train), 100 * len(train) / len(df),
    )
    logger.info(
        "Val   : %s s/d %s | %s baris (%.1f%%)",
        TRAIN_END, VAL_END, len(val), 100 * len(val) / len(df),
    )
    logger.info(
        "Test  : %s s/d %s | %s baris (%.1f%%)",
        VAL_END, tmax, len(test), 100 * len(test) / len(df),
    )
    logger.info("Disimpan ke: %s/{train,val,test}.parquet", PROCESSED_DIR)


if __name__ == "__main__":
    run()
