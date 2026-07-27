/**
 * Generator deck portfolio untuk project
 * "Hourly Passenger Demand Forecasting for Mass Transit Stations".
 *
 * Jalankan: node presentation/build_deck.js
 * Data chart dibaca dari presentation/deck_data.json (dihasilkan dari data
 * hasil pipeline, bukan angka hardcode).
 */
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const ROOT = path.join(__dirname, "..");
const D = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));

// --- Palet "Transit Midnight" ---
const NAVY = "141B34";
const NAVY2 = "232C50";
const ICE = "CADCFC";
const ICE_SOFT = "EDF3FC";
const AMBER = "F5A623";
const WHITE = "FFFFFF";
const GRAY = "5A6478";
const INK = "1B2233";

const HEAD = "Arial";
const BODY = "Calibri";

const W = 13.33;
const M = 0.62;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Herald Michain";
pres.title = "Hourly Passenger Demand Forecasting for Mass Transit Stations";

// ---------- helpers ----------

/** Motif proyek: "station dot" — lingkaran penanda ala peta transit. */
function stationDot(slide, x, y, size, fill, label, labelColor) {
  slide.addShape(pres.ShapeType.ellipse, {
    x, y, w: size, h: size, fill: { color: fill },
  });
  if (label !== undefined) {
    slide.addText(label, {
      x, y, w: size, h: size, align: "center", valign: "middle",
      fontFace: HEAD, fontSize: size > 0.45 ? 14 : 10, bold: true,
      color: labelColor || WHITE, margin: 0,
    });
  }
}

function darkTitle(slide, title, kicker) {
  if (kicker) {
    slide.addText(kicker.toUpperCase(), {
      x: M, y: 0.42, w: 8, h: 0.28, fontFace: BODY, fontSize: 12,
      bold: true, color: AMBER, charSpacing: 2, margin: 0,
    });
  }
  slide.addText(title, {
    x: M, y: kicker ? 0.74 : 0.5, w: W - M * 2, h: 0.72,
    fontFace: HEAD, fontSize: 32, bold: true, color: WHITE, margin: 0,
  });
}

function lightTitle(slide, title, kicker) {
  if (kicker) {
    slide.addText(kicker.toUpperCase(), {
      x: M, y: 0.42, w: 8, h: 0.28, fontFace: BODY, fontSize: 12,
      bold: true, color: AMBER, charSpacing: 2, margin: 0,
    });
  }
  slide.addText(title, {
    x: M, y: kicker ? 0.74 : 0.5, w: W - M * 2, h: 0.72,
    fontFace: HEAD, fontSize: 32, bold: true, color: INK, margin: 0,
  });
}

function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.08,
    fill: { color: fill },
    line: { color: fill },
  });
}

function quietAxis(extra) {
  return Object.assign({
    catAxisLabelColor: GRAY, valAxisLabelColor: GRAY,
    catAxisLabelFontFace: BODY, valAxisLabelFontFace: BODY,
    catAxisLabelFontSize: 11, valAxisLabelFontSize: 11,
    valGridLine: { color: "E3E8F0", size: 1 },
    catGridLine: { style: "none" },
    showLegend: false,
  }, extra || {});
}

// =====================================================================
// 1. TITLE
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };

  s.addText("PORTFOLIO — DATA SCIENCE / MACHINE LEARNING", {
    x: M, y: 0.85, w: 7.6, h: 0.3, fontFace: BODY, fontSize: 12,
    bold: true, color: AMBER, charSpacing: 2, margin: 0,
  });

  s.addText("Hourly Passenger\nDemand Forecasting", {
    x: M, y: 1.28, w: 7.4, h: 1.75, fontFace: HEAD, fontSize: 44,
    bold: true, color: WHITE, lineSpacing: 46, margin: 0,
  });

  s.addText("Prediksi jumlah penumpang per stasiun per jam untuk 428 stasiun transit massal MTA New York — dari 51 juta baris data mentah sampai dashboard operasional.", {
    x: M, y: 3.15, w: 7.1, h: 0.95, fontFace: BODY, fontSize: 15,
    color: ICE, lineSpacing: 22, margin: 0,
  });

  const chips = ["Polars", "LightGBM", "Optuna", "MLflow", "Plotly", "Streamlit"];
  let cx = M;
  chips.forEach((c) => {
    const cw = 0.28 + c.length * 0.098;
    s.addShape(pres.ShapeType.roundRect, {
      x: cx, y: 4.3, w: cw, h: 0.36, rectRadius: 0.18,
      fill: { color: NAVY2 }, line: { color: "3A456E", width: 1 },
    });
    s.addText(c, {
      x: cx, y: 4.3, w: cw, h: 0.36, align: "center", valign: "middle",
      fontFace: BODY, fontSize: 11, bold: true, color: ICE, margin: 0,
    });
    cx += cw + 0.14;
  });

  s.addText("MTA Subway Hourly Ridership  ·  Mei 2022 – Mei 2024", {
    x: M, y: 5.15, w: 7, h: 0.3, fontFace: BODY, fontSize: 12,
    color: "8A93AD", margin: 0,
  });

  // Hero: sebaran 428 stasiun -> membentuk siluet jaringan transit NYC
  s.addChart(pres.ChartType.scatter,
    [
      { name: "lon", values: D.station_lon },
      { name: "Stasiun", values: D.station_lat },
    ],
    {
      x: 8.05, y: 1.15, w: 4.75, h: 4.6,
      chartColors: [AMBER],
      lineSize: 0, lineDataSymbol: "circle", lineDataSymbolSize: 5,
      lineDataSymbolLineColor: AMBER,
      showLegend: false, showTitle: false,
      catAxisHidden: true, valAxisHidden: true,
      valGridLine: { style: "none" }, catGridLine: { style: "none" },
      chartArea: { fill: { color: NAVY } }, plotArea: { fill: { color: NAVY } },
      border: { pt: 0, color: NAVY },
      valAxisMinVal: 40.5, valAxisMaxVal: 40.92,
      catAxisMinVal: -74.06, catAxisMaxVal: -73.72,
    });

  s.addText("428 stasiun · subway, Staten Island Railway & tram", {
    x: 8.05, y: 5.75, w: 4.75, h: 0.3, align: "center",
    fontFace: BODY, fontSize: 10, italic: true, color: "8A93AD", margin: 0,
  });

  s.addNotes("Deck portfolio: project forecasting permintaan penumpang per jam per stasiun. Titik-titik di kanan adalah 428 stasiun asli yang diplot dari koordinat dataset — membentuk jaringan MTA New York.");
}

// =====================================================================
// 2. MASALAH & TUJUAN
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  lightTitle(s, "Kenapa ini penting?", "Masalah & Tujuan");

  card(s, M, 1.72, 5.5, 4.35, ICE_SOFT);
  stationDot(s, M + 0.38, 2.05, 0.42, AMBER, "!", INK);
  s.addText("Perencanaan operasional masih reaktif", {
    x: M + 0.38, y: 2.62, w: 4.75, h: 0.4, fontFace: HEAD, fontSize: 17,
    bold: true, color: INK, margin: 0,
  });
  s.addText("Operator transit perlu tahu KAPAN dan DI STASIUN MANA kepadatan akan tinggi — supaya alokasi armada, jadwal, dan petugas lapangan bisa disiapkan lebih dulu.\n\nTanpa model prediktif, keputusan bersandar pada pola historis kasar atau intuisi, dan penumpukan baru ditangani setelah terjadi.", {
    x: M + 0.38, y: 3.1, w: 4.75, h: 2.6, fontFace: BODY, fontSize: 13.5,
    color: GRAY, lineSpacing: 21, margin: 0,
  });

  const goals = [
    ["1", "Model akurat", "Prediksi entries per stasiun per jam yang mengalahkan baseline naif secara terukur."],
    ["2", "Pola kepadatan", "Identifikasi jam sibuk, weekday vs weekend, dan stasiun paling padat/volatil."],
    ["3", "Bisa dipakai", "Dashboard interaktif agar tim non-teknis bisa eksplorasi sendiri."],
  ];
  let gy = 1.72;
  goals.forEach(([n, h, t]) => {
    card(s, 6.5, gy, W - 6.5 - M, 1.32, WHITE);
    s.addShape(pres.ShapeType.roundRect, {
      x: 6.5, y: gy, w: W - 6.5 - M, h: 1.32, rectRadius: 0.08,
      fill: { color: WHITE }, line: { color: "DEE5F2", width: 1 },
    });
    stationDot(s, 6.78, gy + 0.31, 0.44, NAVY, n);
    s.addText(h, {
      x: 7.42, y: gy + 0.24, w: 4.6, h: 0.32, fontFace: HEAD, fontSize: 15,
      bold: true, color: INK, margin: 0,
    });
    s.addText(t, {
      x: 7.42, y: gy + 0.6, w: 4.9, h: 0.62, fontFace: BODY, fontSize: 12.5,
      color: GRAY, lineSpacing: 17, margin: 0,
    });
    gy += 1.52;
  });

  s.addNotes("Framing masalah dari sisi operasional, bukan sekadar 'bikin model'. Tiga tujuan konkret yang jadi ukuran keberhasilan project.");
}

// =====================================================================
// 3. DATASET & SCOPE
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  lightTitle(s, "Skala data & cakupan", "Dataset");

  const stats = [
    ["51,2 jt", "baris data mentah", "granular per payment\nmethod & fare class"],
    ["7,5 GB", "ukuran file CSV", "butuh strategi\nmemori khusus"],
    ["428", "stasiun transit", "subway, SIR, tram —\nseluruh jaringan"],
    ["2 tahun", "rentang waktu", "Mei 2022 –\nMei 2024"],
  ];
  let sx = M;
  const sw = (W - M * 2 - 0.36 * 3) / 4;
  stats.forEach(([big, lbl, sub]) => {
    card(s, sx, 1.75, sw, 2.05, NAVY);
    s.addText(big, {
      x: sx, y: 1.95, w: sw, h: 0.72, align: "center",
      fontFace: HEAD, fontSize: 34, bold: true, color: AMBER, margin: 0,
    });
    s.addText(lbl, {
      x: sx, y: 2.66, w: sw, h: 0.3, align: "center",
      fontFace: BODY, fontSize: 13, bold: true, color: WHITE, margin: 0,
    });
    s.addText(sub, {
      x: sx, y: 3.0, w: sw, h: 0.68, align: "center",
      fontFace: BODY, fontSize: 11, color: "9AA5C0", lineSpacing: 14, margin: 0,
    });
    sx += sw + 0.36;
  });

  card(s, M, 4.12, W - M * 2, 1.95, ICE_SOFT);
  stationDot(s, M + 0.36, 4.42, 0.42, AMBER, "✓", INK);
  s.addText("Scope ditentukan dari profiling, bukan asumsi", {
    x: M + 0.96, y: 4.42, w: 7, h: 0.4, fontFace: HEAD, fontSize: 17,
    bold: true, color: INK, valign: "middle", margin: 0,
  });
  s.addText([
    { text: "Rentang tanggal dan jumlah stasiun diturunkan langsung dari min/max timestamp dan nunique() data aktual — tidak ada angka yang di-hardcode.", options: { bullet: true, breakLine: true } },
    { text: "Ditemukan 1 baris mislabel: stasiun Tompkinsville (SIR) tercatat sebagai “subway” hanya untuk 2 jam dengan total 3 penumpang — dikecualikan dengan alasan yang didokumentasikan.", options: { bullet: true, breakLine: true } },
    { text: "Skema kolom aktual berbeda dari deskripsi dataset (ridership vs entries, fare_class_category vs fare_type) — dipetakan ulang di pipeline.", options: { bullet: true } },
  ], {
    x: M + 0.96, y: 4.92, w: W - M * 2 - 1.35, h: 1.05,
    fontFace: BODY, fontSize: 12, color: GRAY, lineSpacing: 16,
    paraSpaceAfter: 4, margin: 0,
  });

  s.addNotes("Poin pembeda: scope diturunkan dari profiling data nyata. Saya juga menemukan data mislabel dan perbedaan skema kolom dari dokumentasi dataset.");
}

// =====================================================================
// 4. PIPELINE
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  darkTitle(s, "Pipeline end-to-end", "Arsitektur");

  const steps = [
    ["01", "Profiling", "Rentang & scope\ndari data aktual"],
    ["02", "Cleaning", "Agregasi, gap fill,\npenanganan outlier"],
    ["03", "EDA", "Pola jam sibuk\n& ranking stasiun"],
    ["04", "Features", "Lag, rolling,\nkalender, kategorikal"],
    ["05", "Split", "Time-based\n80 / 10 / 10"],
    ["06", "Tuning", "Optuna 18 trial\n+ MLflow tracking"],
    ["07", "Evaluasi", "vs baseline naif\n& model untuned"],
    ["08", "Dashboard", "Streamlit +\npeta interaktif"],
  ];
  const cw = (W - M * 2 - 0.28 * 3) / 4;
  steps.forEach((st, i) => {
    const col = i % 4;
    const row = Math.floor(i / 4);
    const x = M + col * (cw + 0.28);
    const y = 1.82 + row * 2.02;
    s.addShape(pres.ShapeType.roundRect, {
      x, y, w: cw, h: 1.75, rectRadius: 0.08,
      fill: { color: NAVY2 }, line: { color: "36406B", width: 1 },
    });
    stationDot(s, x + 0.28, y + 0.26, 0.46, AMBER, st[0], INK);
    s.addText(st[1], {
      x: x + 0.28, y: y + 0.8, w: cw - 0.5, h: 0.3,
      fontFace: HEAD, fontSize: 15, bold: true, color: WHITE, margin: 0,
    });
    s.addText(st[2], {
      x: x + 0.28, y: y + 1.13, w: cw - 0.45, h: 0.52,
      fontFace: BODY, fontSize: 11.5, color: "9AA5C0", lineSpacing: 15, margin: 0,
    });
  });

  s.addText("Setiap tahap adalah script terpisah di src/ — bisa dijalankan ulang berurutan dan menghasilkan output byte-identik (sudah diverifikasi).", {
    x: M, y: 6.15, w: W - M * 2, h: 0.4, fontFace: BODY, fontSize: 12,
    italic: true, color: ICE, margin: 0,
  });

  s.addNotes("Pipeline modular: 8 tahap, tiap tahap satu script yang bisa dijalankan independen. Sudah diuji reproducible — output identik saat dijalankan ulang dari CSV mentah.");
}

// =====================================================================
// 5. CLEANING — keputusan teknis
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  lightTitle(s, "Keputusan yang harus diambil", "Data Cleaning");

  const items = [
    ["7,5 GB tidak muat di memori",
     "Streaming group_by Polars gagal alokasi memori (RAM 10,5 GB). Solusi: lazy scan_csv + sink_parquet untuk memangkas 12 kolom jadi 7 dulu, baru agregasi.",
     "51,2 jt → 7,4 jt baris"],
    ["151.474 gap jam kosong",
     "Semua stasiun terverifikasi aktif sepanjang periode, jadi gap = nol tap tercatat. Diisi 0, bukan interpolasi, supaya sifat count tetap terjaga.",
     "2% dari total baris"],
    ["Lonjakan ekstrem",
     "Di-cap pada Q3 + 3×IQR per stasiun. Threshold longgar (bukan 1,5×) supaya jam sibuk wajar dan event besar tidak ikut terpotong.",
     "70.365 baris (0,93%)"],
  ];
  let y = 1.78;
  items.forEach(([h, t, tag], i) => {
    s.addShape(pres.ShapeType.roundRect, {
      x: M, y, w: W - M * 2, h: 1.42, rectRadius: 0.08,
      fill: { color: i % 2 === 0 ? ICE_SOFT : WHITE },
      line: { color: "DEE5F2", width: 1 },
    });
    stationDot(s, M + 0.34, y + 0.48, 0.44, NAVY, String(i + 1));
    s.addText(h, {
      x: M + 0.98, y: y + 0.24, w: 6.2, h: 0.34,
      fontFace: HEAD, fontSize: 16, bold: true, color: INK, margin: 0,
    });
    s.addText(t, {
      x: M + 0.98, y: y + 0.62, w: 8.1, h: 0.66,
      fontFace: BODY, fontSize: 12.5, color: GRAY, lineSpacing: 17, margin: 0,
    });
    s.addText(tag, {
      x: W - M - 2.75, y: y + 0.5, w: 2.5, h: 0.42, align: "right",
      fontFace: HEAD, fontSize: 14, bold: true, color: AMBER, margin: 0,
    });
    y += 1.56;
  });

  s.addNotes("Tiga masalah nyata yang harus diselesaikan, bukan cleaning template. Setiap keputusan punya alasan yang ditulis di docstring kode.");
}

// =====================================================================
// 6. EDA — pola jam
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  lightTitle(s, "Pola kepadatan yang ditemukan", "Exploratory Data Analysis");

  s.addChart(pres.ChartType.line,
    [
      { name: "Weekday", labels: D.hours, values: D.weekday },
      { name: "Weekend", labels: D.hours, values: D.weekend },
    ],
    quietAxis({
      x: M, y: 1.72, w: 7.85, h: 4.05,
      chartColors: [NAVY, AMBER],
      lineSize: 3, lineSmooth: true,
      showLegend: true, legendPos: "t", legendFontFace: BODY, legendFontSize: 11,
      showTitle: true, title: "Rata-rata entries per jam (seluruh 428 stasiun)",
      titleFontFace: HEAD, titleFontSize: 13, titleColor: INK, titleAlign: "left",
      catAxisTitle: "Jam", catAxisTitleFontSize: 11, catAxisTitleColor: GRAY, showCatAxisTitle: true,
      lineDataSymbol: "none",
    }));

  const facts = [
    ["17:00", "jam puncak sore", "782 entries/jam rata-rata weekday — diikuti puncak pagi jam 08:00"],
    ["1,7×", "weekday vs weekend", "340 vs 202 entries/jam — pola commuter, bukan leisure"],
    ["84,5 jt", "Times Sq-42 St", "stasiun tersibuk, jauh di atas Grand Central (59 jt)"],
  ];
  let fy = 1.85;
  facts.forEach(([big, lbl, t]) => {
    card(s, 8.72, fy, W - 8.72 - M, 1.24, ICE_SOFT);
    s.addText(big, {
      x: 8.95, y: fy + 0.14, w: 3.5, h: 0.44,
      fontFace: HEAD, fontSize: 24, bold: true, color: NAVY, margin: 0,
    });
    s.addText(lbl.toUpperCase(), {
      x: 8.95, y: fy + 0.57, w: 3.5, h: 0.24,
      fontFace: BODY, fontSize: 10, bold: true, color: AMBER, charSpacing: 1, margin: 0,
    });
    s.addText(t, {
      x: 8.95, y: fy + 0.79, w: 3.45, h: 0.42,
      fontFace: BODY, fontSize: 10.5, color: GRAY, lineSpacing: 13, margin: 0,
    });
    fy += 1.42;
  });

  s.addNotes("Dua puncak commuter yang jelas (pagi & sore) dan gap weekday/weekend yang besar. Pola inilah yang nanti terbukti jadi sinyal terkuat di feature importance.");
}

// =====================================================================
// 7. FEATURE ENGINEERING & ANTI-LEAKAGE
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  lightTitle(s, "Fitur & penjagaan anti-leakage", "Feature Engineering");

  const groups = [
    ["Lag", "lag_1h · lag_24h · lag_168h", "Nilai jam lalu, hari lalu, minggu lalu — dihitung per stasiun."],
    ["Rolling", "rolling_mean_3h\nrolling_mean_24h", "shift(1) dulu baru rolling, agar jam t tidak ikut terhitung."],
    ["Kalender", "hour · day_of_week\nis_weekend · month", "Menangkap ritme harian, mingguan, dan musiman."],
    ["Kategorikal", "station_complex · borough\ntransit_mode", "Satu model global untuk 428 stasiun (pandas category dtype)."],
  ];
  const gw = (W - M * 2 - 0.3 * 3) / 4;
  let gx = M;
  groups.forEach(([h, code, t]) => {
    s.addShape(pres.ShapeType.roundRect, {
      x: gx, y: 1.75, w: gw, h: 2.42, rectRadius: 0.08,
      fill: { color: WHITE }, line: { color: "DEE5F2", width: 1 },
    });
    stationDot(s, gx + 0.26, 1.99, 0.36, AMBER);
    s.addText(h, {
      x: gx + 0.72, y: 1.99, w: gw - 0.9, h: 0.36,
      fontFace: HEAD, fontSize: 15, bold: true, color: INK, valign: "middle", margin: 0,
    });
    s.addText(code, {
      x: gx + 0.26, y: 2.48, w: gw - 0.5, h: 0.72,
      fontFace: "Courier New", fontSize: 10.5, color: NAVY, lineSpacing: 15, margin: 0,
    });
    s.addText(t, {
      x: gx + 0.26, y: 3.24, w: gw - 0.5, h: 0.8,
      fontFace: BODY, fontSize: 11.5, color: GRAY, lineSpacing: 16, margin: 0,
    });
    gx += gw + 0.3;
  });

  card(s, M, 4.42, W - M * 2, 1.62, NAVY);
  stationDot(s, M + 0.36, 4.72, 0.44, AMBER, "⚠", INK);
  s.addText("Dua sumber leakage yang sengaja dihindari", {
    x: M + 1.0, y: 4.72, w: 8, h: 0.42, fontFace: HEAD, fontSize: 16,
    bold: true, color: WHITE, valign: "middle", margin: 0,
  });
  s.addText([
    { text: "Kolom transfers di-drop dari fitur — tercatat di jam yang sama dengan target, jadi memakainya = melihat masa depan.", options: { bullet: true, breakLine: true } },
    { text: "Rolling mean dihitung dari shift(1).rolling_mean(), bukan rolling langsung, sehingga window hanya berisi jam sebelum t. Kebenaran lag diverifikasi ulang terhadap shift() manual.", options: { bullet: true } },
  ], {
    x: M + 1.0, y: 5.2, w: W - M * 2 - 1.4, h: 0.75,
    fontFace: BODY, fontSize: 11.5, color: ICE, lineSpacing: 15,
    paraSpaceAfter: 3, margin: 0,
  });

  s.addNotes("Leakage adalah kesalahan paling umum di forecasting time-series. Saya tangani eksplisit di dua titik dan verifikasi lag secara manual.");
}

// =====================================================================
// 8. SPLIT + ANOMALI LIBUR
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  lightTitle(s, "Split berbasis waktu — dan satu koreksi penting", "Validasi");

  // Timeline bar
  const tlx = M, tly = 2.0, tlw = W - M * 2, tlh = 0.72;
  const segs = [
    ["Train", 0.809, NAVY, "6.056.628 baris"],
    ["Validation", 0.100, "45528F", "749.856 baris"],
    ["Test", 0.091, AMBER, "681.804 baris"],
  ];
  let px = tlx;
  segs.forEach(([nm, frac, col, sub]) => {
    const w = tlw * frac;
    s.addShape(pres.ShapeType.rect, {
      x: px, y: tly, w, h: tlh, fill: { color: col }, line: { color: col },
    });
    s.addText(nm, {
      x: px, y: tly, w, h: tlh, align: "center", valign: "middle",
      fontFace: HEAD, fontSize: 14, bold: true,
      color: col === AMBER ? INK : WHITE, margin: 0,
    });
    s.addText(`${(frac * 100).toFixed(1).replace(".", ",")}%\n${sub}`, {
      x: px, y: tly + tlh + 0.12, w: Math.max(w, 1.6), h: 0.6,
      align: frac > 0.3 ? "center" : "left",
      fontFace: BODY, fontSize: 10.5, color: GRAY, lineSpacing: 14, margin: 0,
    });
    px += w;
  });
  s.addText("Mei 2022", {
    x: tlx, y: tly - 0.34, w: 2, h: 0.26, fontFace: BODY, fontSize: 10.5,
    color: GRAY, margin: 0,
  });
  s.addText("Mei 2024", {
    x: W - M - 2, y: tly - 0.34, w: 2, h: 0.26, align: "right",
    fontFace: BODY, fontSize: 10.5, color: GRAY, margin: 0,
  });

  card(s, M, 3.62, 6.2, 2.45, ICE_SOFT);
  stationDot(s, M + 0.32, 3.9, 0.42, AMBER, "!", INK);
  s.addText("Titik potong 80% jatuh di tengah libur Natal", {
    x: M + 0.32, y: 4.46, w: 5.6, h: 0.62,
    fontFace: HEAD, fontSize: 16, bold: true, color: INK, margin: 0,
  });
  s.addText("Ridership 25 Des hanya 1,3 juta/hari vs ~3,5–3,9 juta di hari kerja normal. Kalau dibiarkan, validation set dimulai dari periode anomali dan metrik tuning jadi bias.", {
    x: M + 0.32, y: 5.06, w: 5.55, h: 0.9,
    fontFace: BODY, fontSize: 12.5, color: GRAY, lineSpacing: 17, margin: 0,
  });

  card(s, 7.1, 3.62, W - 7.1 - M, 2.45, NAVY);
  stationDot(s, 7.42, 3.9, 0.42, AMBER, "✓", INK);
  s.addText("Batas digeser ke 2 Januari 2024", {
    x: 7.42, y: 4.46, w: 5.2, h: 0.62,
    fontFace: HEAD, fontSize: 16, bold: true, color: WHITE, margin: 0,
  });
  s.addText("Seluruh periode libur masuk ke training (model tetap belajar polanya), dan validation dimulai dari kondisi normal. Proporsi bergeser tipis ke 80,9 / 10,0 / 9,1 — deviasi yang disengaja dan didokumentasikan.", {
    x: 7.42, y: 5.06, w: 5.15, h: 0.9,
    fontFace: BODY, fontSize: 12.5, color: ICE, lineSpacing: 17, margin: 0,
  });

  s.addNotes("Split time-based itu standar. Yang membedakan: saya cek isi window-nya dan menemukan batas jatuh di periode libur Natal, lalu menggesernya dengan alasan eksplisit.");
}

// =====================================================================
// 9. TUNING & TRACKING
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  darkTitle(s, "Tuning & experiment tracking", "Modeling");

  const left = [
    ["Optuna", "18 trial", "Budget terbatas dan disengaja — 7 hyperparameter dicari di atas train/val penuh (6 juta baris), bukan sampel."],
    ["MLflow", "Local tracking", "Setiap trial otomatis tercatat (params, MAE, RMSE) lewat autolog ke file store lokal."],
  ];
  let ly = 1.85;
  left.forEach(([h, tag, t]) => {
    s.addShape(pres.ShapeType.roundRect, {
      x: M, y: ly, w: 6.05, h: 1.72, rectRadius: 0.08,
      fill: { color: NAVY2 }, line: { color: "36406B", width: 1 },
    });
    s.addText(h, {
      x: M + 0.34, y: ly + 0.22, w: 3, h: 0.36,
      fontFace: HEAD, fontSize: 17, bold: true, color: WHITE, margin: 0,
    });
    s.addText(tag, {
      x: M + 3.3, y: ly + 0.24, w: 2.4, h: 0.32, align: "right",
      fontFace: HEAD, fontSize: 13, bold: true, color: AMBER, margin: 0,
    });
    s.addText(t, {
      x: M + 0.34, y: ly + 0.66, w: 5.4, h: 0.88,
      fontFace: BODY, fontSize: 12, color: "9AA5C0", lineSpacing: 16, margin: 0,
    });
    ly += 1.92;
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: 5.69, w: 6.05, h: 0.92, rectRadius: 0.08,
    fill: { color: "2E2418" }, line: { color: AMBER, width: 1 },
  });
  s.addText("Bug yang tertangkap: n_estimators awalnya tersimpan sebagai batas atas pencarian Optuna, bukan best_iteration_ dari early stopping — model final akan melatih pohon berlebih. Diperbaiki lalu tuning diulang.", {
    x: M + 0.26, y: 5.79, w: 5.55, h: 0.74,
    fontFace: BODY, fontSize: 10.5, color: "F5D9A8", lineSpacing: 14, margin: 0,
  });

  // Best params table
  s.addShape(pres.ShapeType.roundRect, {
    x: 7.05, y: 1.85, w: W - 7.05 - M, h: 4.76, rectRadius: 0.08,
    fill: { color: NAVY2 }, line: { color: "36406B", width: 1 },
  });
  s.addText("Hyperparameter terbaik", {
    x: 7.35, y: 2.06, w: 5, h: 0.34,
    fontFace: HEAD, fontSize: 15, bold: true, color: WHITE, margin: 0,
  });
  const params = [
    ["num_leaves", "215"], ["max_depth", "11"],
    ["learning_rate", "0,0603"], ["n_estimators", "890"],
    ["min_data_in_leaf", "48"], ["feature_fraction", "0,623"],
    ["bagging_fraction", "0,918"],
  ];
  let py = 2.58;
  params.forEach(([k, v], i) => {
    if (i % 2 === 0) {
      s.addShape(pres.ShapeType.rect, {
        x: 7.2, y: py - 0.03, w: W - 7.2 - M - 0.15, h: 0.42,
        fill: { color: "2B3559" }, line: { color: "2B3559" },
      });
    }
    s.addText(k, {
      x: 7.35, y: py, w: 3.1, h: 0.36, fontFace: "Courier New", fontSize: 11,
      color: ICE, valign: "middle", margin: 0,
    });
    s.addText(v, {
      x: 10.5, y: py, w: 1.9, h: 0.36, align: "right",
      fontFace: HEAD, fontSize: 12, bold: true, color: AMBER, valign: "middle", margin: 0,
    });
    py += 0.46;
  });
  s.addText("Val MAE terbaik: 25,95  ·  early stopping 50 rounds", {
    x: 7.35, y: 5.92, w: 5, h: 0.34, fontFace: BODY, fontSize: 11,
    italic: true, color: "9AA5C0", margin: 0,
  });

  s.addNotes("Tuning dengan budget sadar (18 trial), semuanya ter-track di MLflow. Saya juga menemukan dan memperbaiki bug n_estimators yang akan menyebabkan overfit di training final.");
}

// =====================================================================
// 10. HASIL
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  lightTitle(s, "LightGBM mengungguli seluruh baseline", "Hasil di Test Set");

  const m = {};
  D.metrics.forEach((r) => { m[r.model] = r; });
  const names = ["LightGBM\n(tuned)", "LightGBM\n(default)", "Baseline\nnaif"];
  const maes = [m.lgbm_tuned.mae, m.lgbm_untuned.mae, m.naive.mae].map((v) => Math.round(v * 100) / 100);
  const rmses = [m.lgbm_tuned.rmse, m.lgbm_untuned.rmse, m.naive.rmse].map((v) => Math.round(v * 100) / 100);

  s.addChart(pres.ChartType.bar,
    [
      { name: "MAE", labels: names, values: maes },
      { name: "RMSE", labels: names, values: rmses },
    ],
    quietAxis({
      x: M, y: 1.74, w: 7.6, h: 4.05,
      barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
      chartColors: [NAVY, ICE],
      showValue: true, dataLabelPosition: "outEnd",
      dataLabelFontFace: HEAD, dataLabelFontSize: 11, dataLabelColor: INK,
      dataLabelFormatCode: "0.0",
      showLegend: true, legendPos: "t", legendFontFace: BODY, legendFontSize: 11,
      showTitle: true, title: "Semakin rendah semakin baik",
      titleFontFace: BODY, titleFontSize: 12, titleColor: GRAY, titleAlign: "left",
      valAxisMinVal: 0, valAxisMaxVal: 125,
    }));

  card(s, 8.5, 1.74, W - 8.5 - M, 1.86, NAVY);
  s.addText("40,7%", {
    x: 8.5, y: 1.95, w: W - 8.5 - M, h: 0.75, align: "center",
    fontFace: HEAD, fontSize: 42, bold: true, color: AMBER, margin: 0,
  });
  s.addText("lebih akurat dari baseline naif", {
    x: 8.5, y: 2.72, w: W - 8.5 - M, h: 0.32, align: "center",
    fontFace: BODY, fontSize: 12.5, bold: true, color: WHITE, margin: 0,
  });
  s.addText("MAE 25,06 vs 42,24", {
    x: 8.5, y: 3.06, w: W - 8.5 - M, h: 0.3, align: "center",
    fontFace: BODY, fontSize: 11, color: "9AA5C0", margin: 0,
  });

  const rows = [
    ["22,9%", "lebih baik dari LightGBM default", "Tuning Optuna memberi kontribusi nyata, bukan formalitas."],
    ["±25", "penumpang rata-rata meleset", "Pada stasiun yang menangani ribuan tap per jam saat puncak."],
  ];
  let ry = 3.86;
  rows.forEach(([big, lbl, t]) => {
    card(s, 8.5, ry, W - 8.5 - M, 0.94, ICE_SOFT);
    s.addText(big, {
      x: 8.72, y: ry + 0.1, w: 1.5, h: 0.4,
      fontFace: HEAD, fontSize: 21, bold: true, color: NAVY, margin: 0,
    });
    s.addText(lbl, {
      x: 10.2, y: ry + 0.16, w: 2.4, h: 0.32,
      fontFace: BODY, fontSize: 10, bold: true, color: INK, lineSpacing: 12, margin: 0,
    });
    s.addText(t, {
      x: 8.72, y: ry + 0.52, w: 3.85, h: 0.36,
      fontFace: BODY, fontSize: 10, color: GRAY, lineSpacing: 12, margin: 0,
    });
    ry += 1.06;
  });

  s.addText("Baseline rolling average 24 jam (MAE 219,9) dikecualikan dari grafik agar skala tetap terbaca.", {
    x: M, y: 5.95, w: 7.6, h: 0.3, fontFace: BODY, fontSize: 10,
    italic: true, color: GRAY, margin: 0,
  });

  s.addNotes("Slide utama. LightGBM tuned menang telak: 40,7% lebih baik dari naif dan 22,9% dari model default. Rolling average sengaja tidak diplot karena skalanya jauh lebih besar.");
}

// =====================================================================
// 11. PREDIKSI & FEATURE IMPORTANCE
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  lightTitle(s, "Seberapa dekat prediksinya?", "Interpretasi Model");

  s.addImage({
    path: path.join(__dirname, "pred_zoom.png"),
    x: M, y: 1.78, w: 7.75, h: 2.96,
  });
  s.addText("Times Sq-42 St, satu minggu di periode test (8–14 April 2024). Prediksi model menempel rapat pada aktual, sementara baseline naif kerap overshoot di puncak — lihat Rabu, Kamis, dan Sabtu.", {
    x: M, y: 4.82, w: 7.75, h: 0.56, fontFace: BODY, fontSize: 11,
    italic: true, color: GRAY, lineSpacing: 15, margin: 0,
  });

  s.addImage({
    path: path.join(ROOT, "models", "feature_importance.png"),
    x: 8.6, y: 1.72, w: 4.1, h: 2.73,
  });

  card(s, 8.6, 4.62, W - 8.6 - M, 1.55, ICE_SOFT);
  s.addText("Pola mingguan = sinyal terkuat", {
    x: 8.82, y: 4.78, w: 3.8, h: 0.32,
    fontFace: HEAD, fontSize: 13.5, bold: true, color: INK, margin: 0,
  });
  s.addText("lag_168h (nilai minggu lalu) mendominasi gain, disusul lag_24h dan lag_1h. Fitur stasiun justru kecil — histori lag sudah menangkap karakter tiap stasiun secara implisit.", {
    x: 8.82, y: 5.14, w: 3.8, h: 0.95,
    fontFace: BODY, fontSize: 10.5, color: GRAY, lineSpacing: 14, margin: 0,
  });

  s.addText("Baseline naif juga memakai nilai minggu lalu — tapi model tetap menang 40,7%, karena ia mengoreksi nilai itu dengan konteks jam, hari, tren jangka pendek, dan karakter stasiun.", {
    x: M, y: 5.55, w: 7.7, h: 0.62, fontFace: BODY, fontSize: 12,
    color: INK, lineSpacing: 16, margin: 0,
  });

  s.addNotes("Feature importance mengonfirmasi temuan EDA: ritme mingguan dan harian adalah prediktor utama. Menariknya fitur kategorikal stasiun justru kecil kontribusinya.");
}

// =====================================================================
// 12. DASHBOARD
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  lightTitle(s, "Dashboard operasional", "Deliverable");

  const feats = [
    ["Peta 428 stasiun", "Ukuran & warna titik = volume ridership; stasiun terpilih disorot."],
    ["Filter interaktif", "Pilih stasiun mana pun dan rentang waktu di periode test."],
    ["Aktual vs prediksi", "Grafik per jam, lengkap dengan pembanding baseline naif."],
    ["Metrik ganda", "MAE/RMSE keseluruhan berdampingan dengan metrik stasiun terpilih."],
    ["Insight jam sibuk", "Profil per jam & kartu jam puncak untuk tiap stasiun."],
  ];
  let fy = 1.75;
  feats.forEach(([h, t]) => {
    stationDot(s, M + 0.04, fy + 0.08, 0.32, AMBER);
    s.addText(h, {
      x: M + 0.56, y: fy, w: 4.4, h: 0.3,
      fontFace: HEAD, fontSize: 14, bold: true, color: INK, margin: 0,
    });
    s.addText(t, {
      x: M + 0.56, y: fy + 0.32, w: 5.5, h: 0.46,
      fontFace: BODY, fontSize: 11.5, color: GRAY, lineSpacing: 15, margin: 0,
    });
    fy += 0.92;
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: 6.35, w: 6.1, h: 0.5, rectRadius: 0.08,
    fill: { color: INK }, line: { color: INK },
  });
  s.addText("streamlit run dashboard/app.py", {
    x: M + 0.22, y: 6.35, w: 5.7, h: 0.5, valign: "middle",
    fontFace: "Courier New", fontSize: 12, color: "8FE3B0", margin: 0,
  });

  card(s, 6.95, 1.75, W - 6.95 - M, 4.55, NAVY);
  s.addText("Peta jaringan — 428 stasiun", {
    x: 7.25, y: 1.95, w: 5, h: 0.32,
    fontFace: HEAD, fontSize: 14, bold: true, color: WHITE, margin: 0,
  });
  s.addChart(pres.ChartType.scatter,
    [
      { name: "lon", values: D.station_lon },
      { name: "Stasiun", values: D.station_lat },
    ],
    {
      x: 7.1, y: 2.3, w: 5.6, h: 3.55,
      chartColors: [AMBER],
      lineSize: 0, lineDataSymbol: "circle", lineDataSymbolSize: 6,
      lineDataSymbolLineColor: AMBER,
      showLegend: false, showTitle: false,
      catAxisHidden: true, valAxisHidden: true,
      valGridLine: { style: "none" }, catGridLine: { style: "none" },
      chartArea: { fill: { color: NAVY } }, plotArea: { fill: { color: NAVY } },
      border: { pt: 0, color: NAVY },
      valAxisMinVal: 40.5, valAxisMaxVal: 40.92,
      catAxisMinVal: -74.06, catAxisMaxVal: -73.72,
    });
  s.addText("Diplot dari koordinat asli tiap stasiun di dataset", {
    x: 7.25, y: 5.9, w: 5, h: 0.28,
    fontFace: BODY, fontSize: 10, italic: true, color: "9AA5C0", margin: 0,
  });

  s.addNotes("Dashboard sudah diuji untuk stasiun tersibuk maupun paling sepi, keduanya render tanpa error. Peta memakai Plotly scatter_mapbox dengan tile open-street-map, tanpa perlu token.");
}

// =====================================================================
// 13. PENUTUP
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  darkTitle(s, "Yang saya bawa dari project ini", "Refleksi & Langkah Lanjut");

  const learn = [
    ["Rekayasa data skala besar", "Menangani file 7,5 GB di RAM 10,5 GB butuh strategi, bukan sekadar library cepat."],
    ["Keputusan berbasis bukti", "Scope, gap filling, capping outlier, dan batas split — semua diturunkan dari profiling dan didokumentasikan."],
    ["Disiplin validasi", "Anti-leakage di dua titik, split time-based, dan pipeline yang terbukti reproducible byte-identik."],
  ];
  let ly = 1.82;
  learn.forEach(([h, t], i) => {
    s.addShape(pres.ShapeType.roundRect, {
      x: M, y: ly, w: 6.35, h: 1.36, rectRadius: 0.08,
      fill: { color: NAVY2 }, line: { color: "36406B", width: 1 },
    });
    stationDot(s, M + 0.3, ly + 0.45, 0.44, AMBER, String(i + 1), INK);
    s.addText(h, {
      x: M + 0.94, y: ly + 0.24, w: 5.1, h: 0.32,
      fontFace: HEAD, fontSize: 15, bold: true, color: WHITE, margin: 0,
    });
    s.addText(t, {
      x: M + 0.94, y: ly + 0.6, w: 5.15, h: 0.62,
      fontFace: BODY, fontSize: 11.5, color: "9AA5C0", lineSpacing: 15, margin: 0,
    });
    ly += 1.52;
  });

  s.addText("Langkah lanjut", {
    x: 7.3, y: 1.82, w: 5.4, h: 0.36,
    fontFace: HEAD, fontSize: 18, bold: true, color: AMBER, margin: 0,
  });
  s.addText([
    { text: "Forecasting multi-step (24 jam ke depan sekaligus) untuk perencanaan shift dan armada harian.", options: { bullet: true, breakLine: true } },
    { text: "Fitur eksternal: cuaca, event besar, dan gangguan layanan yang diketahui memicu lonjakan.", options: { bullet: true, breakLine: true } },
    { text: "Model per-cluster stasiun, memisahkan hub besar dari stasiun residensial kecil.", options: { bullet: true, breakLine: true } },
    { text: "Temporal Fusion Transformer sebagai pembanding — sengaja di luar scope kali ini karena biaya setup dan training.", options: { bullet: true } },
  ], {
    x: 7.3, y: 2.32, w: 5.4, h: 2.9,
    fontFace: BODY, fontSize: 12.5, color: ICE, lineSpacing: 18,
    paraSpaceAfter: 10, margin: 0,
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 7.3, y: 5.42, w: 5.4, h: 0.92, rectRadius: 0.08,
    fill: { color: NAVY2 }, line: { color: AMBER, width: 1 },
  });
  s.addText("Kode lengkap & dokumentasi", {
    x: 7.55, y: 5.54, w: 5, h: 0.28,
    fontFace: BODY, fontSize: 10.5, color: "9AA5C0", margin: 0,
  });
  s.addText("github.com/loxleyftsck/transit-demand-forecasting", {
    x: 7.55, y: 5.82, w: 5, h: 0.34,
    fontFace: HEAD, fontSize: 12, bold: true, color: WHITE, margin: 0,
  });

  s.addNotes("Penutup: tiga kemampuan utama yang ditunjukkan project ini, plus arah pengembangan berikutnya yang realistis. Repo publik tersedia untuk direview.");
}

const OUT = path.join(__dirname, "MTA_Ridership_Forecasting_Portfolio.pptx");
pres.writeFile({ fileName: OUT }).then(() => console.log("Deck ditulis ke:", OUT));
