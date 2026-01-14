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

LBL_BASE   = os.path.join(BASE, "labels/base")
LBL_BROKEN = os.path.join(BASE, "labels/broken")
LBL_HOM    = os.path.join(BASE, "labels/hom")
LBL_AUG    = os.path.join(BASE, "labels/aug")

for d in [IMG_BASE, IMG_BROKEN, IMG_HOM, IMG_AUG,
          LBL_BASE, LBL_BROKEN, LBL_HOM, LBL_AUG]:
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

# ============================================================
# YOLO UTIL
# ============================================================

def corners_to_yolo(c, w, h):
    '''
    Docstring for corners_to_yolo
    
    :param c: corner position
    :param w: image witdh
    :param h: image height
    '''
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
    '''
    Docstring for save_label
    
    :param path: path to the label folder
    :param id1: ID for marker 1
    :param id2: ID for marker 2
    :param c1: corner position of marker 1
    :param c2: corner position of marker 2
    :param w: image witdh
    :param h: image height
    '''
    b1 = corners_to_yolo(c1, w, h)
    b2 = corners_to_yolo(c2, w, h)
    with open(path, "w") as f:
        f.write(f"{id1} {' '.join(map(str,b1))}\n")
        f.write(f"{id2} {' '.join(map(str,b2))}\n")

# ============================================================
# GENERATE MARKER
# ============================================================

def make_marker(marker_id):
    '''
    Docstring for make_marker
    
    :param marker_id: MARKER ID
    '''
    img = cv2.aruco.generateImageMarker(DICT, marker_id, MARKER_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    corner = np.array([
        [0,0],
        [w,0],
        [w,h],
        [0,h]
    ], np.float32)

    return img, corner

# ============================================================
# DEFECT FUNCTIONS (IMAGE + CORNER)
# ============================================================

def gap_horizontal(img, corner, thick=0.06, cut=0.5):
    '''
    Docstring for gap_horizontal
    
    :param img: IMAGE MARKER
    :param corner: corner position
    :param thick: gap thickness
    :param cut: cutting line position
    '''
    h, w = img.shape[:2]
    gap = int(h * thick)
    s = int(h * cut)

    out = np.vstack([
        img[:s],
        np.full((gap, w, 3), 255, np.uint8),
        img[s:]
    ])

    corner = corner.copy()
    corner[corner[:,1] >= s, 1] += gap
    return out, corner

def gap_vertical(img, corner, thick=0.06, cut=0.5):
    '''
    Docstring for gap_vertical
    
    :param img: IMAGE MARKER
    :param corner: corner position
    :param thick: gap thickness
    :param cut: cutting line position
    '''
    h, w = img.shape[:2]
    gap = int(w * thick)
    s = int(w * cut)

    out = np.hstack([
        img[:, :s],
        np.full((h, gap, 3), 255, np.uint8),
        img[:, s:]
    ])

    corner = corner.copy()
    corner[corner[:,0] >= s, 0] += gap
    return out, corner

def shift_horizontal(img, corner, thick=0.02, cut=0.5):
    '''
    Docstring for shift_horizontal
    
    :param img: IMAGE MARKER
    :param corner: corner position
    :param thick: shift thickness
    :param cut: cutting line position
    '''
    h, w = img.shape[:2]
    shift = int(w * thick)
    cut_px = int(h * cut)

    direction = random.choice(["left","right"])
    new_w = w + shift
    out = np.full((h, new_w, 3), 255, np.uint8)

    corner = corner.copy()

    if direction == "right":
        out[:cut_px, shift:shift+w] = img[:cut_px]
        out[cut_px:, :w] = img[cut_px:]
        corner[corner[:,1] < cut_px, 0] += shift
    else:
        out[:cut_px, :w] = img[:cut_px]
        out[cut_px:, shift:shift+w] = img[cut_px:]
        corner[corner[:,1] >= cut_px, 0] += shift

    return out, corner

def faded(img, corner, fade=0.5):
    '''
    Docstring for faded
    
    :param img: IMAGE MARKER
    :param corner: corner position
    :param fade: faded level
    '''
    out = np.clip(img*(1-fade)+255*fade, 0, 255).astype(np.uint8)
    return out, corner

def salt_pepper(img, corner, amount=5, noise_size=5):
    '''
    Docstring for salt_pepper
    
    :param img: IMAGE MARKER
    :param corner: corner position
    :param amount: amount of noise
    :param cut: noise size
    '''
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(amount):
        y = random.randint(0,h-1)
        x = random.randint(0,w-1)
        y1 = max(0,y-noise_size//2)
        y2 = min(h,y1+noise_size)
        x1 = max(0,x-noise_size//2)
        x2 = min(w,x1+noise_size)
        out[y1:y2,x1:x2] = random.choice([0,255])
    return out, corner

def overburning(img, corner, thick=(5,5)):
    '''
    Docstring for overburning
    
    :param img: IMAGE MARKER
    :param corner: corner position
    :param thick: overburning size
    '''
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)
    d = cv2.dilate((b==0).astype(np.uint8)*255, np.ones(thick,np.uint8))
    out = cv2.cvtColor(np.where(d==255,0,255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return out, corner

PRINTER_DEFECTS = {
    "clean": lambda i,c: (i,c),
    "gap_horizontal": gap_horizontal,
    "gap_vertical": gap_vertical,
    "shift_horizontal": shift_horizontal,
    "faded": faded,
    "salt_pepper": salt_pepper,
    "overburning": overburning
}

# ============================================================
# CANVAS PASTE
# ============================================================

def paste(canvas, img, corner, pos):
    '''
    Docstring for paste
    
    :param canvas: canvas
    :param img: marker image
    :param corner: corner position of marker 
    :param pos: top-left position of the marker on the canvas
    '''
    x, y = pos
    h, w = img.shape[:2]
    canvas[y:y+h, x:x+w] = img
    return corner + np.array([x,y], np.float32)

# ============================================================
# HOMOGRAPHY
# ============================================================

def rotate_h(img, c1, c2):
    '''
    Docstring for rotate_h
    
    :param img: image (contain 2 markers)
    :param c1: corner position of marker 1
    :param c2: corner position of marker 2
    '''
    h,w = img.shape[:2]
    a = random.uniform(-25,25)
    M = cv2.getRotationMatrix2D((w/2,h/2), a, 1)
    out = cv2.warpAffine(img, M, (w,h), borderValue=(255,255,255))
    c1 = cv2.transform(c1.reshape(-1,1,2), M).reshape(-1,2)
    c2 = cv2.transform(c2.reshape(-1,1,2), M).reshape(-1,2)
    return out, c1, c2

def zoom_h(img, c1, c2, z):
    '''
    Docstring for zoom_h
    
    :param img: image (contain 2 markers)
    :param c1: corner position of marker 1
    :param c2: corner position of marker 2
    :param z: zoom level
    '''
    h,w = img.shape[:2]
    M = np.array([[z,0,(1-z)*w/2],[0,z,(1-z)*h/2]],np.float32)
    out = cv2.warpAffine(img,M,(w,h),borderValue=(255,255,255))
    c1 = cv2.transform(c1.reshape(-1,1,2),M).reshape(-1,2)
    c2 = cv2.transform(c2.reshape(-1,1,2),M).reshape(-1,2)
    return out,c1,c2

def tilt_h(img, c1, c2):
    '''
    Docstring for tilt_h
    
    :param img: image (contain 2 markers)
    :param c1: corner position of marker 1
    :param c2: corner position of marker 2
    '''
    h,w = img.shape[:2]
    src = np.array([[0,0],[w,0],[w,h],[0,h]],np.float32)
    dst = src + np.random.randint(-40,40,src.shape).astype(np.float32)
    H,_ = cv2.findHomography(src,dst)
    out = cv2.warpPerspective(img,H,(w,h),borderValue=(255,255,255))
    c1 = cv2.perspectiveTransform(c1.reshape(-1,1,2),H).reshape(-1,2)
    c2 = cv2.perspectiveTransform(c2.reshape(-1,1,2),H).reshape(-1,2)
    return out,c1,c2

# ============================================================
# AUGMENTATION
# ============================================================

AUG = iaa.Sequential([
    iaa.MotionBlur(k=5),
    iaa.AdditiveGaussianNoise(scale=(0.01*255,0.03*255)),
    iaa.Multiply((0.7,1.3)),
    iaa.LinearContrast((0.6,1.4)),
    iaa.JpegCompression(compression=(40,90)),
], random_order=True)

# ============================================================
# MAIN PIPELINE
# ============================================================

def run():
    for i in range(0, N_MARKER_ID, 2):
        id1,id2 = i,i+1

        for dname,dfn in PRINTER_DEFECTS.items():
            
            # ============================================================
            # 1. GENERATE MARKER
            # ============================================================

            m1,c1 = make_marker(id1)
            m2,c2 = make_marker(id2)

            # ============================================================
            # 2. APPLYING DEFECTS TO THE MARKER
            # ============================================================

            m1,c1 = dfn(m1,c1)
            m2,c2 = dfn(m2,c2)

            # ============================================================
            # 3. GENERATE MAKRER PAIRS
            # ============================================================

            canvas = np.ones((CANVAS_H,CANVAS_W,3),np.uint8)*255
            c1g = paste(canvas,m1,c1,POS1)
            c2g = paste(canvas,m2,c2,POS2)

            name = f"marker_{id1}_{id2}_{dname}"
            if dname == "clean":
                cv2.imwrite(os.path.join(IMG_BASE,name+".jpg"),canvas)
                save_label(os.path.join(LBL_BASE,name+".txt"),
                        id1,id2,c1g,c2g,canvas.shape[1],canvas.shape[0])
            else:            
                cv2.imwrite(os.path.join(IMG_BROKEN,name+".jpg"),canvas)
                save_label(os.path.join(LBL_BROKEN,name+".txt"),
                       id1,id2,c1g,c2g,canvas.shape[1],canvas.shape[0])

            # ============================================================
            # 4. APPLY HOMOGRAPHY TRANFORMATION
            # ============================================================

            homs=[]
            for _ in range(4): homs.append(rotate_h(canvas,c1g,c2g))
            homs.append(zoom_h(canvas,c1g,c2g,0.7))
            homs.append(zoom_h(canvas,c1g,c2g,1.3))
            for _ in range(4): homs.append(tilt_h(canvas,c1g,c2g))

            for hidx,(himg,hc1,hc2) in enumerate(homs):
                hname=f"{name}_hom-{hidx+1}"
                cv2.imwrite(os.path.join(IMG_HOM,hname+".jpg"),himg)
                save_label(os.path.join(LBL_HOM,hname+".txt"),
                           id1,id2,hc1,hc2,himg.shape[1],himg.shape[0])

                # ============================================================
                # 5. APPLY AUGMENTATION
                # ============================================================
                for a in range(N_AUG):
                    aug=AUG(image=himg)
                    an=f"{hname}_aug-{a+1}"
                    cv2.imwrite(os.path.join(IMG_AUG,an+".jpg"),aug)
                    save_label(os.path.join(LBL_AUG,an+".txt"),
                               id1,id2,hc1,hc2,aug.shape[1],aug.shape[0])

    print("Selesai. Dataset ini akhirnya masuk akal.")

if __name__ == "__main__":
    run()
