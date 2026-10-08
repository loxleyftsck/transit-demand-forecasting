"""Build a short, original portfolio explainer from the public result payload.

The video uses only the project's computed aggregates and a locally stored,
free-to-use subway photograph. It does not expose source rows or run models.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "portfolio-site"
ASSETS = SITE / "assets"
DATA = json.loads((SITE / "data" / "results.json").read_text(encoding="utf-8"))
PHOTO_PATH = ASSETS / "subway-platform.jpg"
VIDEO_PATH = ASSETS / "project-explainer.mp4"
POSTER_PATH = ASSETS / "video-poster.jpg"
MAX_ENTRIES = max(s["total_entries"] for s in DATA["stations"])

W, H, FPS = 960, 540, 20
DURATION = 30
BG = (245, 242, 235)
INK = (24, 37, 31)
MUTED = (100, 113, 106)
GREEN = (49, 95, 75)
ORANGE = (230, 110, 63)
YELLOW = (237, 185, 73)
WHITE = (255, 254, 250)
LINE = (217, 221, 213)


def font(size: int, bold: bool = False, serif: bool = False) -> ImageFont.FreeTypeFont:
    if serif:
        path = Path("C:/Windows/Fonts/georgiab.ttf" if bold else "C:/Windows/Fonts/georgia.ttf")
    else:
        path = Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf")
    return ImageFont.truetype(str(path), size=size)


FONT_CACHE = {}


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, size: int = 20,
         fill=INK, bold: bool = False, serif: bool = False, anchor=None):
    key = (size, bold, serif)
    if key not in FONT_CACHE:
        FONT_CACHE[key] = font(size, bold=bold, serif=serif)
    f = FONT_CACHE[key]
    draw.text(xy, value, font=f, fill=fill, anchor=anchor)


def rounded(draw: ImageDraw.ImageDraw, box, radius=14, fill=WHITE, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def base(kicker: str, number: str, dark: bool = False):
    canvas = Image.new("RGB", (W, H), INK if dark else BG)
    d = ImageDraw.Draw(canvas)
    color = (221, 229, 221) if dark else GREEN
    text(d, (48, 31), kicker.upper(), 14, color, bold=True)
    text(d, (W - 48, 31), number, 14, color, bold=True, anchor="ra")
    d.line((48, 63, W - 48, 63), fill=(68, 84, 75) if dark else LINE, width=1)
    return canvas, d


def cover_frame(progress: float) -> Image.Image:
    photo = Image.open(PHOTO_PATH).convert("RGB")
    scale = 1.03 + 0.07 * progress
    photo = ImageOps.fit(photo, (int(W * scale), int(H * scale)), centering=(0.64 - progress * 0.08, 0.48))
    x = (photo.width - W) // 2
    y = (photo.height - H) // 2
    photo = photo.crop((x, y, x + W, y + H))
    overlay = Image.new("RGBA", (W, H), (14, 27, 22, 175))
    photo = Image.alpha_composite(photo.convert("RGBA"), overlay).convert("RGB")
    d = ImageDraw.Draw(photo)
    text(d, (52, 40), "TRANSIT / DATA     ·     NEW YORK CITY", 14, (229, 238, 228), bold=True)
    d.line((52, 77, W - 52, 77), fill=(192, 207, 193), width=1)
    text(d, (52, 151), "Membaca ritme", 60, WHITE, serif=True)
    text(d, (52, 220), "kota, satu hari", 60, WHITE, serif=True)
    text(d, (52, 289), "ke depan.", 60, WHITE, serif=True)
    text(d, (54, 390), "FORECASTING PENUMPANG PER JAM  ·  MTA SUBWAY", 17, (234, 193, 118), bold=True)
    d.line((54, 441, 281, 441), fill=ORANGE, width=5)
    text(d, (54, 462), "428 stasiun   /   horizon 24 jam", 17, (238, 241, 235))
    return photo


def scope_frame(progress: float) -> Image.Image:
    c, d = base("01 / Skala studi", "00:05 — 00:10")
    text(d, (48, 90), "Satu jaringan. Banyak pola.", 38, INK, serif=True)
    text(d, (50, 144), "Hasil dihitung dari data stasiun-jam pada periode benchmark.", 17, MUTED)
    stats = [("428", "stasiun"), ("678 rb", "observasi dinilai"), ("24 jam", "horizon forecast")]
    for i, (value, label) in enumerate(stats):
        x = 48 + i * 290
        y0 = 210
        rounded(d, (x, y0, x + 260, y0 + 178), fill=WHITE, outline=LINE)
        text(d, (x + 22, y0 + 20), f"0{i+1}", 14, ORANGE, bold=True)
        shown = value if progress > (i * 0.22) else "—"
        text(d, (x + 22, y0 + 63), shown, 46, INK, serif=True)
        text(d, (x + 22, y0 + 126), label.upper(), 14, MUTED, bold=True)
    text(d, (48, 450), "Cakupan stasiun mengikuti hasil profiling dataset; tidak dipatok ke jumlah asumsi.", 14, GREEN)
    return c


def benchmark_frame(progress: float) -> Image.Image:
    c, d = base("02 / Benchmark 24 jam", "00:10 — 00:15")
    rows = sorted(DATA["benchmark_metrics"], key=lambda item: item["mae"])
    text(d, (48, 88), "Chronos-2 memimpin tipis.", 38, INK, serif=True)
    text(d, (50, 139), "MAE lebih rendah berarti prediksi lebih dekat pada nilai aktual.", 17, MUTED)
    max_mae = max(x["mae"] for x in rows)
    top, span = 205, 580
    for i, row in enumerate(rows):
        y = top + i * 55
        ratio = min(1, max(0, progress * 1.18 - i * 0.035))
        label = row["model"]
        winner = i == 0
        text(d, (50, y + 5), label, 17, INK if winner else MUTED, bold=winner)
        d.rounded_rectangle((285, y + 6, 285 + span, y + 28), radius=10, fill=(232, 234, 227))
        width = int(span * row["mae"] / max_mae * ratio)
        if width:
            d.rounded_rectangle((285, y + 6, 285 + width, y + 28), radius=10, fill=ORANGE if winner else (132, 158, 141))
        text(d, (888, y + 2), f"{row['mae']:.2f}", 17, INK, bold=winner, anchor="ra")
    best = rows[0]
    text(d, (50, 494), f"MAE {best['mae']:.2f}  ·  RMSE {best['rmse']:.2f} entries / stasiun-jam", 16, GREEN, bold=True)
    return c


def rhythm_frame(progress: float) -> Image.Image:
    c, d = base("03 / Ritme operasional", "00:15 — 00:20")
    profile = sorted(DATA["hourly_profile"], key=lambda row: row["hour"])
    peak = max(profile, key=lambda row: row["mean_entries"])
    text(d, (48, 88), "Puncak tercatat pukul 17.00.", 38, INK, serif=True)
    text(d, (50, 140), "Rata-rata entries per stasiun-jam pada profil evaluasi one-step.", 17, MUTED)
    x0, x1, y0, y1 = 90, 880, 420, 215
    maximum = max(row["mean_entries"] for row in profile) * 1.08
    d.line((x0, y0, x1, y0), fill=LINE, width=2)
    coords = []
    for row in profile:
        x = x0 + (x1 - x0) * row["hour"] / 23
        y = y0 - (y0 - y1) * row["mean_entries"] / maximum
        coords.append((x, y))
        if row["hour"] % 3 == 0:
            text(d, (int(x), y0 + 16), f"{row['hour']:02d}", 14, MUTED, anchor="mm")
    for a, b in zip(coords, coords[1:]):
        d.line((a[0], a[1], b[0], b[1]), fill=GREEN, width=4)
    for row, (x, y) in zip(profile, coords):
        color = ORANGE if row["hour"] == peak["hour"] else GREEN
        r = 8 if row["hour"] == peak["hour"] else 4
        d.ellipse((x-r, y-r, x+r, y+r), fill=color)
    px, py = coords[peak["hour"]]
    d.line((px, py-12, px, 183), fill=ORANGE, width=2)
    rounded(d, (px - 86, 158, px + 86, 188), radius=14, fill=ORANGE)
    text(d, (px, 173), f"17.00  ·  {peak['mean_entries']:.0f}", 14, WHITE, bold=True, anchor="mm")
    text(d, (50, 482), "Entries adalah catatan gate; satu orang dapat tercatat lebih dari sekali.", 14, MUTED)
    return c


def network_frame(progress: float) -> Image.Image:
    c, d = base("04 / Jaringan stasiun", "00:20 — 00:25", dark=True)
    text(d, (48, 89), "Ruang kota, dibaca dari jaringan.", 38, WHITE, serif=True)
    text(d, (50, 141), "Peta interaktif menempatkan volume pada setiap stasiun yang tersedia.", 17, (198, 211, 201))
    stations = DATA["stations"]
    lats = [s["latitude"] for s in stations]
    lons = [s["longitude"] for s in stations]
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)
    left, right, top, bottom = 68, 610, 204, 458
    for s in stations:
        x = left + (s["longitude"] - lon_min) / (lon_max - lon_min) * (right - left)
        y = bottom - (s["latitude"] - lat_min) / (lat_max - lat_min) * (bottom - top)
        r = 2 + 4 * min(1, s["total_entries"] / MAX_ENTRIES)
        d.ellipse((x-r, y-r, x+r, y+r), fill=ORANGE if s["borough"] == "Manhattan" else (115, 177, 145))
    d.line((650, 213, 650, 452), fill=(68, 84, 75), width=1)
    text(d, (695, 218), "428", 60, WHITE, serif=True)
    text(d, (697, 292), "STASIUN", 15, (208, 221, 211), bold=True)
    text(d, (697, 342), "1 jaringan", 24, (235, 194, 125), serif=True)
    text(d, (697, 381), "volume, lokasi,", 15, (208, 221, 211))
    text(d, (697, 405), "dan ritme waktu.", 15, (208, 221, 211))
    return c


def limits_frame(progress: float) -> Image.Image:
    c, d = base("05 / Cara membaca hasil", "00:25 — 00:30")
    text(d, (48, 93), "Forecast membantu perencanaan.", 38, INK, serif=True)
    text(d, (50, 145), "Bukan pengganti evaluasi operasional dan validasi baru.", 17, MUTED)
    rounded(d, (48, 206, 474, 385), fill=INK)
    text(d, (74, 229), "BATAS INTERPRETASI", 14, (235, 194, 125), bold=True)
    text(d, (74, 269), "Entries ≠ orang unik", 30, WHITE, serif=True)
    text(d, (74, 322), "Jendela ini eksploratif, bukan holdout", 16, (214, 223, 215))
    text(d, (74, 347), "baru yang sepenuhnya tak tersentuh.", 16, (214, 223, 215))
    text(d, (530, 230), "UNTUK", 14, GREEN, bold=True)
    text(d, (530, 266), "Menguji pola", 25, INK, serif=True)
    text(d, (530, 301), "Membandingkan model", 25, INK, serif=True)
    text(d, (530, 336), "Menyusun kapasitas layanan", 25, INK, serif=True)
    d.line((48, 441, W - 48, 441), fill=LINE, width=1)
    text(d, (48, 469), "MTA RIDERSHIP FORECAST  ·  HOURLY  ·  2022—2024", 14, GREEN, bold=True)
    return c


SCENES = [cover_frame, scope_frame, benchmark_frame, rhythm_frame, network_frame, limits_frame]


def make_frame(index: int) -> np.ndarray:
    scene = min(len(SCENES) - 1, index // (FPS * 5))
    local = (index % (FPS * 5)) / (FPS * 5 - 1)
    frame = SCENES[scene](local)
    if scene == 0 and index == 0:
        frame.save(POSTER_PATH, quality=90)
    arr = np.asarray(frame)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def main():
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg tidak ditemukan. Instal ffmpeg lalu jalankan ulang skrip ini.")
    process = subprocess.Popen([
        ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
        "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(VIDEO_PATH),
    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert process.stdin is not None
    try:
        for index in range(DURATION * FPS):
            process.stdin.write(make_frame(index).tobytes())
    finally:
        process.stdin.close()
    code = process.wait()
    if code:
        raise RuntimeError(f"ffmpeg exited with code {code}")
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    ok, first_frame = cap.read()
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not ok or frames != DURATION * FPS:
        raise RuntimeError(f"Video validation failed: readable={ok}, frames={frames}")
    print(f"Created {VIDEO_PATH.relative_to(ROOT)} · {frames} frames · {DURATION}s · {VIDEO_PATH.stat().st_size:,} bytes")
    print(f"Created {POSTER_PATH.relative_to(ROOT)} · {POSTER_PATH.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
