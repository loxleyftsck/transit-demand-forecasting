"""
Data ingestion & cleaning untuk MTA Subway Hourly Ridership 2022-2024.

Pipeline:
  1. Load CSV mentah dengan Polars (lazy + streaming, karena file ~7.5 GB).
  2. Profiling wajib: rentang tanggal aktual & jumlah station_complex unik.
     Scope waktu & stasiun mengikuti hasil profiling ini, tidak di-hardcode.
  3. Perbaikan data error yang ditemukan saat profiling (lihat CATATAN di bawah).
  4. Agregasi dari granularitas (stasiun, jam, payment_method, fare_class_category)
     menjadi (stasiun, jam) dengan menjumlahkan ridership & transfers.
  5. Reindex per stasiun ke rentang jam penuh (mengisi gap dengan 0).
  6. Deteksi & penanganan outlier ekstrem (ridership negatif, lonjakan tak wajar).
  7. Simpan ke data/processed/cleaned.parquet.

CATATAN skema data aktual (berbeda dari asumsi awal brief):
  - Kolom volume penumpang bernama `ridership` (bukan `entries`). Kita rename
    menjadi `entries` di output supaya konsisten dengan istilah domain di seluruh
    proyek (jumlah penumpang per stasiun per jam).
  - Kolom kategori tarif bernama `fare_class_category` (bukan `fare_type`).
  - Ada kolom tambahan `latitude`, `longitude`, `Georeference` — tidak dipakai
    untuk forecasting per-jam, jadi di-drop setelah agregasi (nilainya konstan
    per stasiun sehingga tidak hilang informasi).
  - `station_complex_id` bukan selalu integer (ada "TRAM1", "TRAM2" untuk
    Roosevelt Island Tramway), jadi dibaca sebagai string.
  - `transit_mode` punya 3 nilai: subway, staten_island_railway, tram. Ketiganya
    tetap termasuk "mass transit station" sesuai judul proyek, jadi tidak
    difilter hanya ke subway.

CATATAN data error yang ditemukan & dikecualikan (alasan teknis jelas):
  - station_complex_id "502" (Tompkinsville) muncul dengan transit_mode="subway"
    untuk HANYA 2 baris jam (2023-02-24 & 2023-03-10) dengan total 3 penumpang.
    Padahal Tompkinsville secara faktual adalah stasiun Staten Island Railway,
    dan sudah ada entry terpisah untuk id "502" dengan transit_mode
    "staten_island_railway" yang punya coverage penuh (>88% dari rentang waktu).
    Ini jelas salah label input data (bukan pola nyata), sehingga 2 baris
    tersebut di-drop sebelum agregasi.
"""
import logging
from pathlib import Path

import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "MTA_Subway_Hourly_Ridership.csv"
OUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "cleaned.parquet"

TIMESTAMP_FORMAT = "%m/%d/%Y %I:%M:%S %p"

# Data error yang ditemukan lewat profiling manual (lihat docstring modul).
KNOWN_BAD_ROWS = [
    {"station_complex_id": "502", "transit_mode": "subway"},
]

# Outlier: kalikan IQR sebesar ini untuk batas atas capping per stasiun.
# 3x IQR dipilih (bukan 1.5x standar boxplot) supaya lonjakan wajar akibat
# jam sibuk/event besar tidak ikut terpotong, hanya nilai yang benar-benar
# ekstrem (mis. salah input / duplikasi) yang di-cap.
IQR_MULTIPLIER = 3.0


def scan_raw() -> pl.LazyFrame:
    lf = pl.scan_csv(
        RAW_CSV,
        try_parse_dates=False,
        schema_overrides={"station_complex_id": pl.Utf8},
    )
    lf = lf.with_columns(
        pl.col("transit_timestamp")
        .str.strptime(pl.Datetime, TIMESTAMP_FORMAT, strict=False)
        .alias("transit_timestamp")
    )
    return lf


def profile_raw(lf: pl.LazyFrame) -> dict:
    logger.info("Profiling data mentah (streaming scan atas file ~7.5GB)...")
    stats = lf.select(
        [
            pl.len().alias("n_rows"),
            pl.col("transit_timestamp").min().alias("min_ts"),
            pl.col("transit_timestamp").max().alias("max_ts"),
            pl.col("transit_timestamp").is_null().sum().alias("n_ts_parse_fail"),
            pl.col("station_complex_id").n_unique().alias("n_stations"),
        ]
    ).collect(streaming=True)
    row = stats.row(0, named=True)
    logger.info(
        "Profiling mentah -> n_rows=%s | rentang=%s s/d %s | n_stations=%s | ts_parse_fail=%s",
        row["n_rows"], row["min_ts"], row["max_ts"], row["n_stations"], row["n_ts_parse_fail"],
    )
    return row


GROUP_KEYS = ["station_complex_id", "station_complex", "borough", "transit_mode", "transit_timestamp"]

INTERMEDIATE_PARQUET = PROJECT_ROOT / "data" / "processed" / "_raw_reduced.parquet"


def sink_reduced_columns(lf: pl.LazyFrame) -> None:
    """Buang kolom tak terpakai (payment_method, fare_class_category, lat/long,
    Georeference) & baris data error lewat lazy sink ke parquet perantara.

    `pl.read_csv_batched` gagal ("found more fields than defined in Schema")
    pada file ini karena ada baris CSV yang ragged (jumlah kolom tidak
    konsisten, kemungkinan koma tak ter-quote di suatu tempat pada 51 juta
    baris). `scan_csv` lazy (dipakai juga saat profiling) menoleransi ini
    dengan baik, jadi jalur ingestion dipindah sepenuhnya ke lazy scan +
    `sink_parquet` (streaming write, tanpa group_by) alih-alih batched reader.
    """
    cond = pl.lit(True)
    for bad in KNOWN_BAD_ROWS:
        row_cond = pl.lit(True)
        for k, v in bad.items():
            row_cond = row_cond & (pl.col(k) == v)
        cond = cond & ~row_cond

    reduced = lf.filter(cond).select(GROUP_KEYS + ["ridership", "transfers"])
    logger.info("Menulis parquet perantara (kolom relevan saja) via streaming sink...")
    reduced.sink_parquet(INTERMEDIATE_PARQUET)
    logger.info("Parquet perantara selesai: %s", INTERMEDIATE_PARQUET)


def aggregate_to_station_hour() -> pl.DataFrame:
    """Jumlahkan ridership & transfers lintas payment_method/fare_class_category.

    Dibaca dari parquet perantara (kolom sudah dipangkas dari 12 -> 7 kolom)
    lalu diagregasi secara eager. Parquet columnar + kolom yang lebih sedikit
    membuat operasi ini jauh lebih ringan dibanding group_by langsung di atas
    CSV mentah 7.5GB, yang sempat gagal alokasi memori di lingkungan ini
    (RAM tersedia ~10.5GB).
    """
    df = pl.read_parquet(INTERMEDIATE_PARQUET)
    logger.info("Parquet perantara dimuat ke memori: %s baris.", df.height)
    agg = df.group_by(GROUP_KEYS).agg(
        [
            pl.col("ridership").sum().alias("entries"),
            pl.col("transfers").sum().alias("transfers"),
        ]
    )
    return agg


def reindex_full_hourly_range(df: pl.DataFrame, min_ts, max_ts) -> pl.DataFrame:
    """Reindex tiap stasiun ke rentang jam penuh [min_ts, max_ts], isi gap dengan 0.

    Keputusan desain: semua 428 stasiun (setelah drop data error) punya rekaman
    yang membentang dari min_ts s/d max_ts global (diverifikasi lewat profiling
    per-stasiun), jadi tidak ada indikasi stasiun baru dibuka/ditutup di tengah
    periode. Gap jam yang hilang paling mungkin merepresentasikan jam dengan nol
    tap tercatat (bukan data hilang secara acak), sehingga diisi 0 -- bukan
    interpolasi -- supaya sifat count/volume dari `entries`/`transfers` tetap
    terjaga (interpolasi bisa menciptakan nilai pecahan/tren palsu).
    """
    full_hours = pl.datetime_range(min_ts, max_ts, interval="1h", eager=True).alias("transit_timestamp")
    full_hours_df = pl.DataFrame({"transit_timestamp": full_hours})

    stations = df.select(["station_complex_id", "station_complex", "borough", "transit_mode"]).unique()

    scaffold = stations.join(full_hours_df, how="cross")
    reindexed = scaffold.join(
        df,
        on=["station_complex_id", "station_complex", "borough", "transit_mode", "transit_timestamp"],
        how="left",
        coalesce=True,
    )
    reindexed = reindexed.with_columns(
        [
            pl.col("entries").fill_null(0),
            pl.col("transfers").fill_null(0),
        ]
    )
    return reindexed


def handle_outliers(df: pl.DataFrame) -> pl.DataFrame:
    """Clip nilai negatif ke 0, dan cap lonjakan ekstrem per stasiun via IQR."""
    n_negative_entries = df.filter(pl.col("entries") < 0).height
    n_negative_transfers = df.filter(pl.col("transfers") < 0).height
    if n_negative_entries or n_negative_transfers:
        logger.warning(
            "Ditemukan %s entries negatif dan %s transfers negatif -> di-clip ke 0.",
            n_negative_entries, n_negative_transfers,
        )
    df = df.with_columns(
        [
            pl.col("entries").clip(lower_bound=0),
            pl.col("transfers").clip(lower_bound=0),
        ]
    )

    q = df.group_by("station_complex_id").agg(
        [
            pl.col("entries").quantile(0.25).alias("q1"),
            pl.col("entries").quantile(0.75).alias("q3"),
        ]
    ).with_columns(
        (pl.col("q3") - pl.col("q1")).alias("iqr")
    ).with_columns(
        (pl.col("q3") + IQR_MULTIPLIER * pl.col("iqr")).alias("upper_cap")
    )

    df = df.join(q.select(["station_complex_id", "upper_cap"]), on="station_complex_id", how="left", coalesce=True)
    n_capped = df.filter(pl.col("entries") > pl.col("upper_cap")).height
    logger.info(
        "Capping lonjakan ekstrem: %s baris (%.4f%%) di atas Q3 + %.1fxIQR per stasiun akan di-cap.",
        n_capped, 100 * n_capped / df.height, IQR_MULTIPLIER,
    )
    df = df.with_columns(
        pl.when(pl.col("entries") > pl.col("upper_cap"))
        .then(pl.col("upper_cap"))
        .otherwise(pl.col("entries"))
        .round(0)
        .cast(pl.Int64)
        .alias("entries")
    ).drop("upper_cap")

    return df


def run() -> None:
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    lf = scan_raw()
    raw_profile = profile_raw(lf)

    sink_reduced_columns(lf)

    logger.info("Menjalankan agregasi station-hour dari parquet perantara...")
    agg_df = aggregate_to_station_hour()
    n_rows_agg = agg_df.height
    n_stations_final = agg_df["station_complex_id"].n_unique()
    logger.info("Agregasi selesai -> %s baris station-hour, %s stasiun unik.", n_rows_agg, n_stations_final)

    min_ts = raw_profile["min_ts"]
    max_ts = raw_profile["max_ts"]

    reindexed = reindex_full_hourly_range(agg_df, min_ts, max_ts)
    n_rows_reindexed = reindexed.height
    n_gaps_filled = n_rows_reindexed - n_rows_agg
    logger.info(
        "Reindex ke rentang jam penuh [%s s/d %s] -> %s baris (%s gap jam diisi 0).",
        min_ts, max_ts, n_rows_reindexed, n_gaps_filled,
    )

    cleaned = handle_outliers(reindexed)
    cleaned = cleaned.sort(["station_complex_id", "transit_timestamp"])

    cleaned.write_parquet(OUT_PARQUET)

    logger.info("=== RINGKASAN CLEANING ===")
    logger.info("Rentang tanggal aktual   : %s s/d %s", min_ts, max_ts)
    logger.info("Jumlah stasiun aktual    : %s (dari %s station_complex_id mentah, 1 dikecualikan: lihat docstring)",
                 n_stations_final, raw_profile["n_stations"])
    logger.info("Baris mentah (fare/payment granular): %s", raw_profile["n_rows"])
    logger.info("Baris setelah agregasi station-hour  : %s", n_rows_agg)
    logger.info("Baris setelah reindex (gap terisi)   : %s", n_rows_reindexed)
    logger.info("Gap jam yang ditangani (fill 0)       : %s", n_gaps_filled)
    logger.info("Disimpan ke: %s", OUT_PARQUET)

    INTERMEDIATE_PARQUET.unlink(missing_ok=True)


if __name__ == "__main__":
    run()
