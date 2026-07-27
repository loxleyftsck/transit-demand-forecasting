# Prediksi Kepadatan Penumpang Transportasi Massal (Proyek Taktis: Time-Series & Visualisasi)

## Latar Belakang

Proyek ini berfokus pada kesesuaian dengan operasional MRT, menonjolkan kemampuan Exploratory Data Analysis (EDA), forecasting time-series, dan data storytelling untuk konteks operasional.

## Masalah

Operator transportasi massal butuh mengetahui kapan dan di stasiun mana kepadatan penumpang akan tinggi, supaya alokasi armada, jadwal, dan petugas lapangan bisa diatur secara proaktif — bukan reaktif setelah penumpukan terjadi. Saat ini belum ada model prediktif yang mengestimasi volume penumpang per jam per stasiun ke depan, sehingga perencanaan operasional masih mengandalkan pola historis kasar atau intuisi.

## Tujuan

1. Membangun model forecasting untuk memprediksi jumlah penumpang (entries) per stasiun per jam, dengan performa lebih baik dari baseline naif (misal: prediksi = nilai minggu lalu di jam yang sama).
2. Mengidentifikasi pola kepadatan: jam sibuk (peak hour), perbedaan hari kerja vs akhir pekan, stasiun dengan volume tertinggi/paling volatil.
3. Menerjemahkan hasil model menjadi rekomendasi operasional yang mudah dipahami non-teknis (data storytelling) — misal "Stasiun X pada jam Y diprediksi naik 30% dari rata-rata, pertimbangkan tambahan armada."
4. Menyajikan hasil dalam dashboard interaktif untuk eksplorasi mandiri (filter stasiun, rentang waktu, bandingkan aktual vs prediksi).

## Pertanyaan yang Dijawab

- Bagaimana pola harian dan mingguan kepadatan penumpang? Ada jam puncak pagi/sore yang konsisten?
- Stasiun mana yang paling padat, dan mana yang paling tidak stabil (variance tinggi)?
- Seberapa akurat prediksi kepadatan untuk 1 jam ke depan? 24 jam ke depan?
- Fitur apa (lag, hour, day, station) yang paling berpengaruh terhadap prediksi (feature importance)?

## Sumber Data

**Dataset utama (digunakan untuk modeling):**
- [MTA Subway Hourly Ridership 2022–2024 (Kaggle)](https://www.kaggle.com/datasets/yaminh/mta-subway-hourly-ridership-2022-to-2024) — granularitas per jam per stasiun, sumber asli data resmi MTA (data.gov). Kolom: `transit_timestamp`, `transit_mode`, `station_complex_id`, `station_complex`, `borough`, `payment_method`, `fare_type`, `entries`, `transfers`.

**Dataset pendukung/konteks lokal (opsional, untuk EDA tambahan):**
- [Transjakarta — Public Transportation Transaction (Kaggle)](https://www.kaggle.com/datasets/dikisahkan/transjakarta-transportation-transaction) — ~189.501 record transaksi granular (tap-in/tap-out, waktu, lokasi).
- [TransJakarta 2021 (Kaggle)](https://www.kaggle.com/datasets/robincolinkang/transjakarta-2021) — agregat bulanan per rute.

**Sumber data resmi MRT/LRT Jakarta (level bulanan, referensi saja):**
- [Data Penumpang MRT — Satu Data Jakarta](https://data.jakarta.go.id/dataset/data-penumpang-mrt-di-provinsi-dki-jakarta)
- [Data Jumlah Penumpang MRT — data.go.id](https://data.go.id/dataset/dataset/data-jumlah-penumpang-mass-rapid-transit-mrt)
- [Statistik Penumpang MRT — BPS DKI Jakarta](https://jakarta.bps.go.id/en/statistics-table/2/MTMxOCMy/jumlah-penumpang-mass-rapid-transit--mrt--jakarta.html)

**Catatan:** data granular per jam/stasiun untuk MRT Jakarta tidak dipublikasikan secara terbuka. Dataset MTA digunakan sebagai proxy internasional dengan metodologi yang applicable ke konteks operasional MRT Jakarta.

## Metodologi

**Model:** LightGBM dengan lag features (bukan ARIMA/Prophet/LSTM) — dipilih karena:
- Mendukung satu model global untuk banyak stasiun sekaligus (station sebagai fitur kategorikal), lebih efisien waktu dibanding model per-stasiun.
- Robust terhadap data noisy/outlier, tidak membutuhkan data stasioner.
- Training relatif cepat dibanding model per-stasiun atau deep learning.
- Feature importance otomatis mendukung data storytelling.

**Fitur yang direncanakan:**
- Lag: `lag_1h`, `lag_24h` (hari sebelumnya, jam sama), `lag_168h` (minggu sebelumnya)
- Rolling mean: 3 jam, 24 jam
- Kalender: `hour_of_day`, `day_of_week`, `is_weekend`, `month`
- Kategorikal: `station_complex`, `borough`

**Validasi:** split berbasis waktu (bukan random) — porsi terakhir dari rentang data aktual (misal 5-10% dari total durasi) sebagai test set, untuk menghindari data leakage dari lag features dan mengurangi risiko bias akibat window evaluasi yang terlalu sempit.

**Hal yang perlu dicek sebelum modeling:**
1. Data profiling di awal: tentukan rentang tanggal aktual dan jumlah `station_complex` unik pada dataset (bukan asumsi/hardcode) — gunakan seluruh cakupan yang tersedia
2. Kontinuitas timestamp per stasiun (gap jam yang hilang → reindex + fillna)
3. Panjang data cukup untuk lag mingguan (168h)
4. Split train/test berbasis waktu, bukan random

## Metrik Keberhasilan

- **Teknis:** MAE/RMSE model LightGBM lebih rendah dari baseline (naive lag / rolling average), diukur di test set time-based.
- **Naratif:** dashboard dapat menjawab "kapan dan di mana penumpang akan padat" dalam satu tampilan tanpa perlu membaca kode.

## Rencana Alur Kerja

Scope mengikuti data aktual: seluruh rentang waktu yang tersedia di dataset dan seluruh `station_complex` yang ada, ditentukan lewat data profiling di tahap awal — bukan angka asumsi.

1. Setup + data profiling (cek rentang tanggal & jumlah stasiun aktual)
2. EDA (pola harian/mingguan/musiman, missing/outlier, visualisasi tren)
3. Feature engineering + modeling (LightGBM + lag features)
4. Dashboard (Plotly Dash / Streamlit, 1 halaman)
5. Dokumentasi/narasi untuk CV

**Catatan scope:**
- Gunakan seluruh `station_complex` yang tersedia di dataset, kecuali ada stasiun dengan data terlalu sedikit/tidak lengkap (dikeluarkan berdasarkan hasil profiling, bukan dipilih sembarangan)
- Dashboard tetap 1 halaman, tapi dengan filter stasiun mencakup seluruh daftar stasiun aktual
- EDA Transjakarta bersifat bonus, bukan wajib

## Deliverables

1. Notebook EDA + feature engineering + model LightGBM
2. Evaluasi model (MAE/RMSE vs baseline, feature importance)
3. Dashboard interaktif (aktual vs prediksi, filter stasiun/waktu)
4. Ringkasan naratif (data storytelling) untuk konteks operasional
5. *(Stretch goal, opsional)* Visualisasi peta kepadatan per stasiun (Plotly `scatter_mapbox` / `folium`) — karena butuh data lat/long tambahan di luar dataset utama
