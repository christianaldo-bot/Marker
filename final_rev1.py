import cv2
import numpy as np
import os
import random
import imgaug.augmenters as iaa

# ============================================================
# CONFIG
# ============================================================

BASE = "Data_Final"

IMG_BASE   = os.path.join(BASE, "images/base")
IMG_BROKEN = os.path.join(BASE, "images/broken")
IMG_HOM    = os.path.join(BASE, "images/hom")
IMG_AUG    = os.path.join(BASE, "images/aug")
IMG_NEG    = os.path.join(BASE, "images/negative")

LBL_BASE   = os.path.join(BASE, "labels/base")
LBL_BROKEN = os.path.join(BASE, "labels/broken")
LBL_HOM    = os.path.join(BASE, "labels/hom")
LBL_AUG    = os.path.join(BASE, "labels/aug")

for d in [
    IMG_BASE, IMG_BROKEN, IMG_HOM, IMG_AUG, IMG_NEG,
    LBL_BASE, LBL_BROKEN, LBL_HOM, LBL_AUG
]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# GLOBAL PARAMETER
# ============================================================

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)

MARKER_SIZE = 200
GAP = int(MARKER_SIZE * 0.2)

CANVAS_H = MARKER_SIZE * 3
CANVAS_W = MARKER_SIZE * 4 + GAP

POS1 = (MARKER_SIZE, MARKER_SIZE)
POS2 = (MARKER_SIZE * 2 + GAP, MARKER_SIZE)

N_MARKER_ID = 4
N_AUG = 5
N_NEG = 20


# ============================================================
# YOLO UTIL
# ============================================================

def corners_to_yolo(c, w, h):
    xs, ys = c[:,0], c[:,1]
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    return [
        ((xmin + xmax) / 2) / w,
        ((ymin + ymax) / 2) / h,
        (xmax - xmin) / w,
        (ymax - ymin) / h
    ]

def save_label(path, id1, id2, c1, c2, w, h):
    b1 = corners_to_yolo(c1, w, h)
    b2 = corners_to_yolo(c2, w, h)
    with open(path, "w") as f:
        f.write(f"{id1} {' '.join(map(str,b1))}\n")
        f.write(f"{id2} {' '.join(map(str,b2))}\n")

# ============================================================
# MARKER GENERATION
# ============================================================

def make_marker(marker_id):
    img = cv2.aruco.generateImageMarker(DICT, marker_id, MARKER_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    corner = np.array([[0,0],[w,0],[w,h],[0,h]], np.float32)
    return img, corner

# ============================================================
# DEFECT SAMPLER
# ============================================================

def sample_param(dname):
    p = DEFECT_PARAM[dname]
    param = {}
    for k,v in p.items():
        if isinstance(v, tuple):
            param[k] = random.uniform(*v)
        else:
            param[k] = random.choice(v)
    return param

# ============================================================
# DEFECT FUNCTIONS
# ============================================================

def gap_horizontal(img, corner, p):
    h, w = img.shape[:2]
    gap = int(h * p["thick"])
    s = int(h * p["cut"])
    out = np.vstack([img[:s], np.full((gap,w,3),255,np.uint8), img[s:]])
    corner = corner.copy()
    corner[corner[:,1] >= s, 1] += gap
    return out, corner

def gap_vertical(img, corner, p):
    h, w = img.shape[:2]
    gap = int(w * p["thick"])
    s = int(w * p["cut"])
    out = np.hstack([img[:,:s], np.full((h,gap,3),255,np.uint8), img[:,s:]])
    corner = corner.copy()
    corner[corner[:,0] >= s, 0] += gap
    return out, corner

def shift_horizontal(img, corner, p):
    h, w = img.shape[:2]
    shift = int(w * p["thick"])
    cut = int(h * p["cut"])
    out = np.full((h, w+shift, 3), 255, np.uint8)
    corner = corner.copy()
    if p["direction"] == "right":
        out[:cut, shift:] = img[:cut]
        out[cut:, :w] = img[cut:]
        corner[corner[:,1] < cut, 0] += shift
    else:
        out[:cut, :w] = img[:cut]
        out[cut:, shift:] = img[cut:]
        corner[corner[:,1] >= cut, 0] += shift
    return out, corner

def faded(img, corner, p):
    out = np.clip(img*(1-p["fade"]) + 255*p["fade"],0,255).astype(np.uint8)
    return out, corner

def salt_pepper(img, corner, p):
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(int(p["amount"])):
        y = random.randint(0,h-1)
        x = random.randint(0,w-1)
        s = int(p["noise_size"])
        out[y:y+s, x:x+s] = random.choice([0,255])
    return out, corner

def overburning(img, corner, p):
    k = int(p["kernel"])
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(g,127,255,cv2.THRESH_BINARY)
    d = cv2.dilate((b==0).astype(np.uint8)*255, np.ones((k,k),np.uint8))
    out = cv2.cvtColor(np.where(d==255,0,255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return out, corner

DEFECT_FUNC = {
    "clean": None,
    "gap_horizontal": gap_horizontal,
    "gap_vertical": gap_vertical,
    "shift_horizontal": shift_horizontal,
    "faded": faded,
    "salt_pepper": salt_pepper,
    "overburning": overburning
}


# ============================================================
# DEFECT PARAMETER CONFIG (EDIT HERE)
# ============================================================

DEFECT_PARAM = {
    "gap_horizontal": {
        "thick": (0.04, 0.08),
        "cut": (0.3, 0.7)
    },
    "gap_vertical": {
        "thick": (0.04, 0.08),
        "cut": (0.3, 0.7)
    },
    "shift_horizontal": {
        "thick": (0.01, 0.04),
        "cut": (0.4, 0.6),
        "direction": ["left", "right"]
    },
    "faded": {
        "fade": (0.3, 0.6)
    },
    "salt_pepper": {
        "amount": (3, 10),
        "noise_size": (3, 7)
    },
    "overburning": {
        "kernel": (3, 7)
    }
}


# ============================================================
# CANVAS
# ============================================================

def paste(canvas, img, corner, pos):
    x,y = pos
    h,w = img.shape[:2]
    canvas[y:y+h, x:x+w] = img
    return corner + np.array([x,y],np.float32)

# ============================================================
# HOMOGRAPHY
# ============================================================

def rotate_h(img,c1,c2):
    h,w = img.shape[:2]
    a = random.uniform(-25,25)
    M = cv2.getRotationMatrix2D((w/2,h/2),a,1)
    out = cv2.warpAffine(img,M,(w,h),borderValue=(255,255,255))
    return out, cv2.transform(c1.reshape(-1,1,2),M).reshape(-1,2), cv2.transform(c2.reshape(-1,1,2),M).reshape(-1,2)

def zoom_h(img,c1,c2,z):
    h,w = img.shape[:2]
    M = np.array([[z,0,(1-z)*w/2],[0,z,(1-z)*h/2]],np.float32)
    out = cv2.warpAffine(img,M,(w,h),borderValue=(255,255,255))
    return out, cv2.transform(c1.reshape(-1,1,2),M).reshape(-1,2), cv2.transform(c2.reshape(-1,1,2),M).reshape(-1,2)

def tilt_h(img,c1,c2):
    h,w = img.shape[:2]
    src = np.array([[0,0],[w,0],[w,h],[0,h]],np.float32)
    dst = src + np.random.randint(-40,40,src.shape).astype(np.float32)
    H,_ = cv2.findHomography(src,dst)
    out = cv2.warpPerspective(img,H,(w,h),borderValue=(255,255,255))
    return out, cv2.perspectiveTransform(c1.reshape(-1,1,2),H).reshape(-1,2), cv2.perspectiveTransform(c2.reshape(-1,1,2),H).reshape(-1,2)

# ============================================================
# AUGMENTATION (CAMERA)
# ============================================================

AUG = iaa.Sequential([
    iaa.MotionBlur(k=5),
    iaa.AdditiveGaussianNoise(scale=(0.01*255,0.03*255)),
    iaa.Multiply((0.7,1.3)),
    iaa.LinearContrast((0.6,1.4)),
    iaa.JpegCompression(compression=(40,90))
], random_order=True)

# ============================================================
# MAIN PIPELINE
# ============================================================

def run():

    for i in range(0, N_MARKER_ID, 2):
        id1, id2 = i, i+1

        for dname, dfn in DEFECT_FUNC.items():

            m1,c1 = make_marker(id1)
            m2,c2 = make_marker(id2)

            if dname != "clean":
                param = sample_param(dname)
                m1,c1 = dfn(m1,c1,param)
                m2,c2 = dfn(m2,c2,param)

            canvas = np.ones((CANVAS_H,CANVAS_W,3),np.uint8)*255
            c1g = paste(canvas,m1,c1,POS1)
            c2g = paste(canvas,m2,c2,POS2)

            name = f"marker_{id1}_{id2}_{dname}"
            img_dir = IMG_BASE if dname=="clean" else IMG_BROKEN
            lbl_dir = LBL_BASE if dname=="clean" else LBL_BROKEN

            cv2.imwrite(os.path.join(img_dir,name+".jpg"),canvas)
            save_label(os.path.join(lbl_dir,name+".txt"),id1,id2,c1g,c2g,canvas.shape[1],canvas.shape[0])

            homs = []
            for _ in range(4): homs.append(rotate_h(canvas,c1g,c2g))
            homs.append(zoom_h(canvas,c1g,c2g,0.7))
            homs.append(zoom_h(canvas,c1g,c2g,1.3))
            for _ in range(4): homs.append(tilt_h(canvas,c1g,c2g))

            for hidx,(himg,hc1,hc2) in enumerate(homs):
                hname = f"{name}_hom-{hidx+1}"
                cv2.imwrite(os.path.join(IMG_HOM,hname+".jpg"),himg)
                save_label(os.path.join(LBL_HOM,hname+".txt"),id1,id2,hc1,hc2,himg.shape[1],himg.shape[0])

                for a in range(N_AUG):
                    aug = AUG(image=himg)
                    an = f"{hname}_aug-{a+1}"
                    cv2.imwrite(os.path.join(IMG_AUG,an+".jpg"),aug)
                    save_label(os.path.join(LBL_AUG,an+".txt"),id1,id2,hc1,hc2,aug.shape[1],aug.shape[0])

    # NEGATIVE SAMPLE
    for i in range(N_NEG):
        neg = np.ones((CANVAS_H,CANVAS_W,3),np.uint8)*random.randint(200,255)
        cv2.imwrite(os.path.join(IMG_NEG,f"neg_{i+1}.jpg"),neg)

    print("DATASET SELESAI & SIAP TRAINING")

# run()
if __name__ == "__main__":
    run()
