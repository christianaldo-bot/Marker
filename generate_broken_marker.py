import cv2
import numpy as np
import os
import random

# ============================================================
# CONFIG
# ============================================================

ROOT = "Dataset_2"

IMG_BROKEN = os.path.join(ROOT, "image", "broken_image")
IMG_HOMO   = os.path.join(ROOT, "image", "homography_broken_image")
LBL_BROKEN = os.path.join(ROOT, "label", "broken_label")
LBL_HOMO   = os.path.join(ROOT, "label", "homography_broken_label")

for d in [IMG_BROKEN, IMG_HOMO, LBL_BROKEN, LBL_HOMO]:
    os.makedirs(d, exist_ok=True)

CANVAS_H, CANVAS_W = 300, 550
MARKER_SIZE = 200

POS1 = (50, 50)     # y, x
POS2 = (50, 300)

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)

# ============================================================
# EFFECT FUNCTIONS
# ============================================================

def gap_horizontal(img, ratio=0.06):
    h, w = img.shape[:2]
    gap = int(h * ratio)
    s = random.randint(0, h)
    return np.vstack([img[:s], np.full((gap, w, 3), 255, np.uint8), img[s:]])

def gap_vertical(img, ratio=0.06):
    h, w = img.shape[:2]
    gap = int(w * ratio)
    s = random.randint(0, w)
    return np.hstack([img[:, :s], np.full((h, gap, 3), 255, np.uint8), img[:, s:]])

def shift_horizontal(img, ratio=0.08):
    h, w = img.shape[:2]
    cut = h // 2
    shift = int(w * ratio)
    nw = w + shift
    out = np.full((h, nw, 3), 255, np.uint8)
    out[:cut, shift:shift+w] = img[:cut]
    out[cut:, :w] = img[cut:]
    return out

def faded(img, fade=0.5):
    return np.clip(img*(1-fade)+255*fade, 0, 255).astype(np.uint8)

def salt_pepper(img, amount=0.02):
    out = img.copy()
    h, w = img.shape[:2]
    n = int(h*w*amount)
    ys, xs = np.random.randint(0,h,n), np.random.randint(0,w,n)
    out[ys,xs] = 255
    ys, xs = np.random.randint(0,h,n), np.random.randint(0,w,n)
    out[ys,xs] = 0
    return out

def overburning(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)
    d = cv2.dilate((b==0).astype(np.uint8)*255, np.ones((3,3),np.uint8))
    return cv2.cvtColor(np.where(d==255,0,255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

EFFECTS = {
    "gap_horizontal": gap_horizontal,
    "gap_vertical": gap_vertical,
    "shift_horizontal": shift_horizontal,
    "faded": faded,
    "salt_pepper": salt_pepper,
    "overburning": overburning,
}

# ============================================================
# MARKER + BBOX UTILS
# ============================================================

def generate_marker(mid):
    m = cv2.aruco.generateImageMarker(DICT, mid, MARKER_SIZE)
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)

def paste(canvas, img, pos):
    y, x = pos
    h, w = img.shape[:2]
    canvas[y:y+h, x:x+w] = img
    return np.array([
        [x, y],
        [x+w, y],
        [x+w, y+h],
        [x, y+h]
    ], np.float32)

def bbox_yolo(pts, w, h):
    x = np.clip(pts[:,0], 0, w)
    y = np.clip(pts[:,1], 0, h)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xc = (xmin+xmax)/2/w
    yc = (ymin+ymax)/2/h
    bw = (xmax-xmin)/w
    bh = (ymax-ymin)/h
    return xc, yc, bw, bh

# ============================================================
# HOMOGRAPHY
# ============================================================

def rotate(img, pts, name, idx):
    a = random.uniform(-25,25)
    M = cv2.getRotationMatrix2D((CANVAS_W/2,CANVAS_H/2),a,1)
    out = cv2.warpAffine(img,M,(CANVAS_W,CANVAS_H))
    pts = cv2.transform(pts.reshape(-1,1,2),M).reshape(-1,2)
    return out, pts, f"{name}_rotate_{idx}"

def zoom(img, pts, z, tag):
    M = np.array([[z,0,(1-z)*CANVAS_W/2],[0,z,(1-z)*CANVAS_H/2]],np.float32)
    out = cv2.warpAffine(img,M,(CANVAS_W,CANVAS_H))
    pts = cv2.transform(pts.reshape(-1,1,2),M).reshape(-1,2)
    return out, pts, f"{tag}"

def tilt(img, pts, idx):
    src = np.array([[0,0],[CANVAS_W,0],[CANVAS_W,CANVAS_H],[0,CANVAS_H]],np.float32)
    dst = src + np.random.randint(-40,40,src.shape).astype(np.float32)
    H,_ = cv2.findHomography(src,dst)
    out = cv2.warpPerspective(img,H,(CANVAS_W,CANVAS_H))
    pts = cv2.perspectiveTransform(pts.reshape(-1,1,2),H).reshape(-1,2)
    return out, pts, f"tilt_{idx}"

# ============================================================
# SAVE HELPER
# ============================================================

def save(img, pts, base, tag):
    name = f"{base}_{tag}"
    cv2.imwrite(os.path.join(IMG_HOMO,f"{name}.png"),img)
    p1,p2 = pts[:4],pts[4:]
    with open(os.path.join(LBL_HOMO,f"{name}.txt"),"w") as f:
        f.write(f"{id1} " + " ".join(map(str, bbox_yolo(p1, CANVAS_W, CANVAS_H))) + "\n")
        f.write(f"{id2} " + " ".join(map(str, bbox_yolo(p2, CANVAS_W, CANVAS_H))) + "\n")



# ============================================================
# MAIN PIPELINE
# ============================================================

for pair in range(500):
    id1, id2 = pair*2, pair*2+1

    for eff_name, eff in EFFECTS.items():

        canvas = np.full((CANVAS_H,CANVAS_W,3),255,np.uint8)

        m1 = eff(generate_marker(id1))
        m2 = eff(generate_marker(id2))

        p1 = paste(canvas, m1, POS1)
        p2 = paste(canvas, m2, POS2)

        fname = f"marker_{id1}_{id2}_{eff_name}"
        cv2.imwrite(os.path.join(IMG_BROKEN,f"{fname}.png"),canvas)

        with open(os.path.join(LBL_BROKEN,f"{fname}.txt"),"w") as f:
            f.write(f"{id1} " + " ".join(map(str, bbox_yolo(p1, CANVAS_W, CANVAS_H))) + "\n")
            f.write(f"{id2} " + " ".join(map(str, bbox_yolo(p2, CANVAS_W, CANVAS_H))) + "\n")

        # ---- HOMOGRAPHY ----
        base_pts = np.vstack([p1,p2])
        for i in range(4):
            img, pts, tag = rotate(canvas, base_pts, eff_name, i+1)
            save(img,pts,fname,tag)

        img, pts, tag = zoom(canvas, base_pts,1.2,"zoom_in")
        save(img,pts,fname,tag)

        img, pts, tag = zoom(canvas, base_pts,0.8,"zoom_out")
        save(img,pts,fname,tag)

        for i in range(4):
            img, pts, tag = tilt(canvas, base_pts, i+1)
            save(img,pts,fname,tag)

print("SELESAI. 30000 GAMBAR. HIDUP LANJUT.")
