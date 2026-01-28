import cv2
import os

# ============================================================
# CONFIG
# ============================================================

IMG_DIR = "Data_Final/images/aug"     # bisa ganti: base / broken / hom / aug
LBL_DIR = "Data_Final/labels/aug"

DEBUG_OUT = "Data_Final/debug_bbox"
os.makedirs(DEBUG_OUT, exist_ok=True)

SHOW = True       # True = tampilkan window
SAVE = True       # True = simpan ke folder debug
MAX_DEBUG = 50    # batasi jumlah gambar biar kamu tetap waras

# ============================================================
# YOLO → PIXEL CONVERTER
# ============================================================

def yolo_to_pixel(box, w, h):
    cx, cy, bw, bh = box
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    return x1, y1, x2, y2

# ============================================================
# MAIN DEBUG LOOP
# ============================================================

count = 0

for fname in os.listdir(IMG_DIR):
    if not fname.endswith(".jpg"):
        continue

    img_path = os.path.join(IMG_DIR, fname)
    lbl_path = os.path.join(LBL_DIR, fname.replace(".jpg", ".txt"))

    if not os.path.exists(lbl_path):
        continue

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    with open(lbl_path) as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        cls = int(parts[0])
        box = list(map(float, parts[1:]))

        x1, y1, x2, y2 = yolo_to_pixel(box, w, h)

        # draw bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            img, f"ID {cls}", (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )


    if SAVE:
        cv2.imwrite(os.path.join(DEBUG_OUT, fname), img)

    count += 1
    if count >= MAX_DEBUG:
        break

cv2.destroyAllWindows()
print("Debug selesai. Kalau bbox-nya salah, ini waktunya kamu marah ke kodenya.")
