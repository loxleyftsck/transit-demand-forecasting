"""
Download dataset MTA Subway Hourly Ridership 2022-2024 dari Kaggle ke data/raw/.

Prasyarat:
1. Buat akun Kaggle, generate API token di https://www.kaggle.com/settings -> "Create New Token".
2. Simpan file kaggle.json yang diunduh ke:
   - Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json
   - Linux/Mac: ~/.kaggle/kaggle.json
3. pip install kaggle (sudah ada di requirements.txt)

Jalankan:
    python src/download_data.py

Jika tidak punya kredensial Kaggle API, download manual dari:
https://www.kaggle.com/datasets/yaminh/mta-subway-hourly-ridership-2022-to-2024
lalu ekstrak CSV-nya ke data/raw/MTA_Subway_Hourly_Ridership.csv
"""
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATASET = "yaminh/mta-subway-hourly-ridership-2022-to-2024"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing = list(RAW_DIR.glob("*.csv"))
    if existing:
        print(f"Dataset sudah ada di {existing[0]}, skip download.")
        return

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        raise SystemExit(
            "Kaggle API tidak tersedia atau kredensial belum diset. "
            "Lihat instruksi di docstring file ini untuk setup manual atau via API.\n"
            f"Detail error: {e}"
        )

    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {DATASET} ke {RAW_DIR} ...")
    api.dataset_download_files(DATASET, path=str(RAW_DIR), unzip=False)

    for zf in RAW_DIR.glob("*.zip"):
        print(f"Extracting {zf.name} ...")
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(RAW_DIR)
        zf.unlink()

    print("Selesai. File tersedia di data/raw/.")


if __name__ == "__main__":
    main()
