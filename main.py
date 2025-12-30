import cv2
import numpy as np
import os
import random
import imgaug.augmenters as iaa


# ============================================================
# 1. Folder Setup
# ============================================================

BASE = "dataset"

IMG_BASE = os.path.join(BASE, "images/base")
IMG_HOM  = os.path.join(BASE, "images/hom")
IMG_AUG  = os.path.join(BASE, "images/aug")

LBL_HOM  = os.path.join(BASE, "labels/hom")
LBL_AUG  = os.path.join(BASE, "labels/aug")

for folder in [IMG_BASE, IMG_HOM, IMG_AUG, LBL_HOM, LBL_AUG]:
    os.makedirs(folder, exist_ok=True)



# ============================================================
# 2. Generate Base Marker
# ============================================================

def generate_base_markers():
    # Get AR Marker Dictionary
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)

    # List to save all information of Marker
    data = []

    # Make 500 base images of AR Marker
    for marker_id in range(500):
        # Define AR Marker ID 
        id1 = marker_id * 2
        id2 = marker_id * 2 + 1

        # Size of the Canvas and Marker
        canvas_H = 300
        canvas_W = 550
        marker_size = 200

        # Create white Canvas
        canvas = np.ones((canvas_H, canvas_W, 3), dtype=np.uint8) * 255

        # Determine Marker position on the canvas (top left point)
        x1, y1 = 50, 50
        x2, y2 = 300, 50

        # Generate AR Marker
        m1 = cv2.aruco.generateImageMarker(dictionary, id1, marker_size)
        m2 = cv2.aruco.generateImageMarker(dictionary, id2, marker_size)

        # Convert Color CHannel of Marker to RGB
        m1 = cv2.cvtColor(m1, cv2.COLOR_GRAY2BGR)
        m2 = cv2.cvtColor(m2, cv2.COLOR_GRAY2BGR)

        # Stick AR Marker to the Canvas
        canvas[y1:y1+marker_size, x1:x1+marker_size] = m1
        canvas[y2:y2+marker_size, x2:x2+marker_size] = m2

        # Save Marker position on the coordinate format (bounding box)
        corners1 = np.array([[x1, y1], [x1+marker_size, y1],
                             [x1+marker_size, y1+marker_size], [x1, y1+marker_size]], np.float32)

        corners2 = np.array([[x2, y2], [x2+marker_size, y2],
                             [x2+marker_size, y2+marker_size], [x2, y2+marker_size]], np.float32)

        # Save the image on the folder
        filename = f"marker_{id1}_{id2}.png"
        cv2.imwrite(os.path.join(IMG_BASE, filename), canvas)

        # Save the all information about marker (image, marker ID, marker position, marker file)
        data.append({
            "img": canvas,
            "id1": id1,
            "id2": id2,
            "c1": corners1,
            "c2": corners2,
            "name": f"marker_{id1}_{id2}"
        })

    return data



# ============================================================
# 3. Homography Transforms
# ============================================================

def rotate_h(img, c1, c2):
    # Determine a random number for angle
    angle = random.uniform(-25, 25)
    # Get height and witdh of the image
    h, w = img.shape[:2]
    # Make a matrix rotation
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    # Tranform the image according the matrix position
    out = cv2.warpAffine(img, M, (w, h))    # warpAffine --> applying geometric transformations to an image

    # Change corner position coordinate according to image tranformation
    c1n = cv2.transform(c1.reshape(-1,1,2), M).reshape(-1,2)
    c2n = cv2.transform(c2.reshape(-1,1,2), M).reshape(-1,2)
    return out, c1n, c2n


def zoom_h(img, c1, c2, zoom):
    h, w = img.shape[:2]
    # Make a matrix tranformation affine 2D (zoom in and zoom out from center of image)
    M = np.array([[zoom, 0, (1-zoom)*w/2], [0, zoom, (1-zoom)*h/2]], np.float32)
    out = cv2.warpAffine(img, M, (w, h))
    c1n = cv2.transform(c1.reshape(-1,1,2), M).reshape(-1,2)
    c2n = cv2.transform(c2.reshape(-1,1,2), M).reshape(-1,2)
    return out, c1n, c2n


def tilt_h(img, c1, c2):
    h, w = img.shape[:2]
    # Determine starting point
    src = np.array([[0,0],[w,0],[w,h],[0,h]], np.float32)
    # Determine target corners that have been randomly tilted
    dst = src + np.random.randint(-40, 40, src.shape).astype(np.float32)    
    # Find matrix tranformation 
    H, _ = cv2.findHomography(src, dst)
    out = cv2.warpPerspective(img, H, (w, h))
    c1n = cv2.perspectiveTransform(c1.reshape(-1,1,2), H).reshape(-1,2)
    c2n = cv2.perspectiveTransform(c2.reshape(-1,1,2), H).reshape(-1,2)
    return out, c1n, c2n



# ============================================================
# 4. YOLO Bounding Box
# ============================================================

def corners_to_yolo(c, w, h):
    # Separate coordinate X and coordinate Y from corner coordinate
    xs, ys = c[:,0], c[:,1]
    # Find the minimum and maximum coordinate of X and Y
    xmin, xmax = max(0,min(xs)), min(w-1,max(xs))
    ymin, ymax = max(0,min(ys)), min(h-1,max(ys))
    return [(xmin+xmax)/2/w, (ymin+ymax)/2/h,   # calculate center of bounding box. (xmin+xmax) <-- marker position
            (xmax-xmin)/w, (ymax-ymin)/h]       # calculate bounding box size



# ============================================================
# 5.  Label
# ============================================================

def save_label(path, id1, id2, c1, c2, w, h):
    # Determine Bounding Box
    b1 = corners_to_yolo(c1, w, h)
    b2 = corners_to_yolo(c2, w, h)
    # Open file and write
    with open(path, "w") as f:
        f.write(f"{id1} {b1[0]} {b1[1]} {b1[2]} {b1[3]}\n")
        f.write(f"{id2} {b2[0]} {b2[1]} {b2[2]} {b2[3]}\n")



# ============================================================
# 6. ImgAug
# ============================================================

AUG = iaa.Sequential([
    iaa.MotionBlur(k=5),
    iaa.AdditiveGaussianNoise(scale=(0.01*255, 0.03*255)),
    iaa.Multiply((0.7, 1.3)),
    iaa.LinearContrast((0.6, 1.4)),
    iaa.JpegCompression(compression=(40, 90)),
], random_order=True)



# ============================================================
# 7. Main Pipeline
# ============================================================

def run():

    print("Generating base markers...")
    base_data = generate_base_markers()

    idx = 0

    for item in base_data:

        base_img = item["img"]
        id1, id2 = item["id1"], item["id2"]
        c1, c2   = item["c1"], item["c2"]
        name     = item["name"]

        # ----------------------------------------------------
        # 10 Homography
        # ----------------------------------------------------
        hom_list = []

        for _ in range(4): hom_list.append( rotate_h(base_img, c1, c2) )
        hom_list.append( zoom_h(base_img, c1, c2, 0.7) )
        hom_list.append( zoom_h(base_img, c1, c2, 1.3) )
        for _ in range(4): hom_list.append( tilt_h(base_img, c1, c2) )


        # ----------------------------------------------------
        # Save Homography Images
        # ----------------------------------------------------
        for i, (himg, hc1, hc2) in enumerate(hom_list):

            img_name = f"{name}_hom-{i+1}.jpg"
            img_path = os.path.join(IMG_HOM, img_name)
            lbl_path = os.path.join(LBL_HOM, img_name.replace(".jpg", ".txt"))

            cv2.imwrite(img_path, himg)
            save_label(lbl_path, id1, id2, hc1, hc2, himg.shape[1], himg.shape[0])


            # ------------------------------------------------
            # 10 Augmentations
            # ------------------------------------------------
            for a in range(10):

                aug_img = AUG(image=himg)

                aug_name = f"{name}_hom-{i+1}_aug-{a+1}.jpg"
                aug_path = os.path.join(IMG_AUG, aug_name)
                aug_lbl  = os.path.join(LBL_AUG, aug_name.replace(".jpg", ".txt"))

                cv2.imwrite(aug_path, aug_img)
                save_label(aug_lbl, id1, id2, hc1, hc2, aug_img.shape[1], aug_img.shape[0])

                idx += 1

    print("Selesai. Foldermu sekarang rapi seperti hidup orang lain.")



if __name__ == "__main__":
    run()



# import cv2
# import numpy as np
# import random

# dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
# for marker_id in range(500):
#     # Define AR Marker ID 
#     id1 = marker_id * 2
#     id2 = marker_id * 2 + 1

#     # Size of the Canvas and Marker
#     canvas_H = 300
#     canvas_W = 550
#     marker_size = 200

#     # Create white Canvas
#     canvas = np.ones((canvas_H, canvas_W, 3), dtype=np.uint8) * 255

#     # Determine Marker position on the canvas (top left point)
#     x1, y1 = 50, 50
#     x2, y2 = 300, 50

#         # Generate AR Marker
#     m1 = cv2.aruco.generateImageMarker(dictionary, id1, marker_size)
#     m2 = cv2.aruco.generateImageMarker(dictionary, id2, marker_size)

#         # Convert Color CHannel of Marker to RGB
#     m1 = cv2.cvtColor(m1, cv2.COLOR_GRAY2BGR)
#     m2 = cv2.cvtColor(m2, cv2.COLOR_GRAY2BGR)

#         # Stick AR Marker to the Canvas
#     canvas[y1:y1+marker_size, x1:x1+marker_size] = m1
#     canvas[y2:y2+marker_size, x2:x2+marker_size] = m2





# # Generate marker
# marker_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
# marker_size = 200

# m1 = cv2.aruco.generateImageMarker(marker_dict, 1, marker_size)
# m2 = cv2.aruco.generateImageMarker(marker_dict, 2, marker_size)

# m1 = cv2.cvtColor(m1, cv2.COLOR_GRAY2BGR)
# m2 = cv2.cvtColor(m2, cv2.COLOR_GRAY2BGR)



# def gap_horizontal(img, ratio=0.06):
#     color = (255, 255, 255)
#     h, w = img.shape[:2]
#     gap = int(h * ratio)
#     start = np.random.randint(0, h + 1)

#     top = img[:start]
#     bottom = img[start:]

#     new_gap = np.full((gap, w, img.shape[2]), color, dtype=img.dtype)

#     return np.vstack([top, new_gap, bottom])

# def gap_vertical(img, ratio=0.06):
#     color = (255, 255, 255)
#     h, w = img.shape[:2]
#     gap = int(w * ratio)
#     start = np.random.randint(0, w + 1)

#     left = img[:, :start]
#     right = img[:, start:]

#     # gap vertikal (putih)
#     new_gap = np.full((h, gap, img.shape[2]), color, dtype=img.dtype)

#     return np.hstack([left, new_gap, right])

# import numpy as np

# def shift_horizontal(img, ratio=0.08, color=(255, 255, 255)):
#     h, w = img.shape[:2]
#     cut = h // 2

#     top = img[:cut]
#     bottom = img[cut:]

#     shift = int(w * ratio)

#     # canvas baru untuk top agar menyesuaikan shift
#     new_w = w + shift
#     new_top = np.full((top.shape[0], new_w, top.shape[2]), color, dtype=top.dtype)

#     if np.random.rand() < 0.5:
#         # geser ke kanan → isi dari kolom shift ke w+shift
#         new_top[:, shift:shift+w] = top
#     else:
#         # geser ke kiri → isi dari kolom 0 ke w
#         new_top[:, :w] = top

#     # sesuaikan bottom supaya lebar sama
#     new_bottom = np.full((bottom.shape[0], new_w, bottom.shape[2]), color, dtype=bottom.dtype)
#     new_bottom[:, :w] = bottom

#     return np.vstack([new_top, new_bottom])


# def make_faded(img, fade_ratio=0.5):
#     """
#     Membuat gambar terlihat 'tipis' / memudar
#     fade_ratio: 0 (tidak pudar) - 1 (putih total)
#     """
#     # pastikan float supaya aman untuk perkalian
#     img_float = img.astype(np.float32)

#     # campur dengan putih
#     faded = img_float * (1 - fade_ratio) + 255 * fade_ratio

#     # kembalikan ke uint8
#     faded = np.clip(faded, 0, 255).astype(np.uint8)
#     return faded





# def add_salt_pepper_noise(img, amount=0.02):
#     """
#     Menambahkan noise salt-and-pepper.
#     amount: proporsi pixel yang terkena noise (0.0 - 1.0)
#     """
#     noisy = img.copy()
#     h, w = img.shape[:2]
#     num_noise = int(amount * h * w)

#     # salt (putih)
#     coords = [np.random.randint(0, i, num_noise) for i in (h, w)]
#     noisy[coords[0], coords[1]] = 255

#     # pepper (hitam)
#     coords = [np.random.randint(0, i, num_noise) for i in (h, w)]
#     noisy[coords[0], coords[1]] = 0

#     return noisy

# def overburning_effect(img, kernel_size=3, iterations=1):
#     """
#     Membuat efek 'overburning' dengan dilasi.
#     - img: gambar grayscale atau biner (0 & 255)
#     - kernel_size: ukuran kernel dilasi, semakin besar semakin tebal
#     - iterations: berapa kali dilasi diterapkan
    
#     Output: gambar dengan area putih yang melebar (lebih tebal)
#     """
#     # pastikan gambar grayscale atau biner
#     if len(img.shape) == 3 and img.shape[2] == 3:
#         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     else:
#         img_gray = img.copy()

#     # threshold agar benar-benar biner
#     _, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

#      # Invers gambar: putih jadi hitam, hitam jadi putih
#     black_mask = (img_bin == 0).astype(np.uint8) * 255

#     # buat kernel (structuring element)
#     kernel = np.ones((kernel_size, kernel_size), np.uint8)

#     # dilasi
#     img_dilated = cv2.dilate(black_mask, kernel, iterations=iterations)

#     # Sekarang buat output:
#     # Semua area yang jadi hitam setelah dilasi tetap hitam (0)
#     # Sisanya putih (255)
#     img_out = np.where(img_dilated == 255, 0, 255).astype(np.uint8)

#     # Konversi ke 3 channel warna
#     img_out_color = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)

#     return img_out_color



# hasil_normal = m1
# hasil_baru = overburning_effect(m1)








# h1, w1 = hasil_normal.shape[:2]
# h2, w2 = hasil_baru.shape[:2]

# spacing = int(min(w1, w2) * 0.2)

# canvas_h = max(h1, h2) + 120
# canvas_w = w1 + w2 + spacing + 120

# canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255


# y1 = (canvas_h - h1) // 2
# y2 = (canvas_h - h2) // 2

# x1 = (canvas_w - (w1 + w2 + spacing)) // 2
# x2 = x1 + w1 + spacing

# canvas[y1:y1+h1, x1:x1+w1] = hasil_normal
# canvas[y2:y2+h2, x2:x2+w2] = hasil_baru


# cv2.imwrite("final_dual_marker_dynamic_size.png", canvas)
# print("Selesai. Ukuran marker jujur, dunia tidak disterilkan.")












# import cv2
# import numpy as np
# import os

# # =========================
# # CONFIG
# # =========================

# OUT_DIR = "output_markers"
# os.makedirs(OUT_DIR, exist_ok=True)

# CANVAS_H = 300
# CANVAS_W = 550
# MARKER_SIZE = 200

# POS1 = (50, 50)   # (y, x)
# POS2 = (50, 300)

# DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)

# # =========================
# # EFFECT FUNCTIONS
# # =========================

# def gap_horizontal(img, ratio=0.06, color=(255, 255, 255)):
#     h, w = img.shape[:2]
#     gap = int(h * ratio)
#     start = np.random.randint(0, h + 1)
#     top = img[:start]
#     bottom = img[start:]
#     gap_area = np.full((gap, w, 3), color, dtype=img.dtype)
#     return np.vstack([top, gap_area, bottom])


# def gap_vertical(img, ratio=0.06, color=(255, 255, 255)):
#     h, w = img.shape[:2]
#     gap = int(w * ratio)
#     start = np.random.randint(0, w + 1)
#     left = img[:, :start]
#     right = img[:, start:]
#     gap_area = np.full((h, gap, 3), color, dtype=img.dtype)
#     return np.hstack([left, gap_area, right])


# def shift_horizontal(img, ratio=0.08, color=(255, 255, 255)):
#     h, w = img.shape[:2]
#     cut = h // 2
#     shift = int(w * ratio)

#     top = img[:cut]
#     bottom = img[cut:]

#     new_w = w + shift
#     new_top = np.full((top.shape[0], new_w, 3), color, dtype=img.dtype)
#     new_bottom = np.full((bottom.shape[0], new_w, 3), color, dtype=img.dtype)

#     if np.random.rand() < 0.5:
#         new_top[:, shift:shift + w] = top
#     else:
#         new_top[:, :w] = top

#     new_bottom[:, :w] = bottom
#     return np.vstack([new_top, new_bottom])


# def make_faded(img, fade_ratio=0.5):
#     img_f = img.astype(np.float32)
#     faded = img_f * (1 - fade_ratio) + 255 * fade_ratio
#     return np.clip(faded, 0, 255).astype(np.uint8)


# def add_salt_pepper_noise(img, amount=0.02):
#     noisy = img.copy()
#     h, w = img.shape[:2]
#     num = int(amount * h * w)

#     y, x = np.random.randint(0, h, num), np.random.randint(0, w, num)
#     noisy[y, x] = 255

#     y, x = np.random.randint(0, h, num), np.random.randint(0, w, num)
#     noisy[y, x] = 0

#     return noisy


# def overburning_effect(img, kernel_size=3, iterations=1):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

#     black_mask = (binary == 0).astype(np.uint8) * 255
#     kernel = np.ones((kernel_size, kernel_size), np.uint8)

#     dilated = cv2.dilate(black_mask, kernel, iterations=iterations)
#     result = np.where(dilated == 255, 0, 255).astype(np.uint8)

#     return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


# EFFECTS = {
#     "gap_horizontal": gap_horizontal,
#     "gap_vertical": gap_vertical,
#     "shift_horizontal": shift_horizontal,
#     "faded": make_faded,
#     "salt_pepper": add_salt_pepper_noise,
#     "overburning": overburning_effect,
# }

# # =========================
# # HELPER
# # =========================

# def generate_marker(marker_id):
#     m = cv2.aruco.generateImageMarker(DICT, marker_id, MARKER_SIZE)
#     return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)


# def paste_safe(canvas, img, top_left):
#     H, W = canvas.shape[:2]
#     h, w = img.shape[:2]
#     y, x = top_left

#     y_end = min(y + h, H)
#     x_end = min(x + w, W)

#     h_valid = y_end - y
#     w_valid = x_end - x

#     if h_valid <= 0 or w_valid <= 0:
#         return  # marker terlalu besar → dilewati (tidak crash)

#     canvas[y:y_end, x:x_end] = img[:h_valid, :w_valid]


# # =========================
# # MAIN LOOP
# # =========================

# for effect_name, effect_func in EFFECTS.items():
#     effect_dir = os.path.join(OUT_DIR, effect_name)
#     os.makedirs(effect_dir, exist_ok=True)

#     for pair_idx in range(500):
#         id1 = pair_idx * 2
#         id2 = pair_idx * 2 + 1

#         m1 = generate_marker(id1)
#         m2 = generate_marker(id2)

#         # SAME effect type, applied independently
#         m1_d = effect_func(m1)
#         m2_d = effect_func(m2)

#         canvas = np.full((CANVAS_H, CANVAS_W, 3), 255, dtype=np.uint8)

#         paste_safe(canvas, m1_d, POS1)
#         paste_safe(canvas, m2_d, POS2)

#         fname = f"{effect_name}_pair_{pair_idx:03d}_id_{id1}_{id2}.png"
#         cv2.imwrite(os.path.join(effect_dir, fname), canvas)

#     print(f"Selesai efek: {effect_name}")

# print("SEMUA 3000 GAMBAR SELESAI. Dataset hidup, skripsi selamat.")












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



