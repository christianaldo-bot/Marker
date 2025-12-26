# import cv2
# import numpy as np
# import os
# import random
# import imgaug.augmenters as iaa


# # ============================================================
# # 1. Folder Setup
# # ============================================================

# BASE = "dataset"

# IMG_BASE = os.path.join(BASE, "images/base")
# IMG_HOM  = os.path.join(BASE, "images/hom")
# IMG_AUG  = os.path.join(BASE, "images/aug")

# LBL_HOM  = os.path.join(BASE, "labels/hom")
# LBL_AUG  = os.path.join(BASE, "labels/aug")

# for folder in [IMG_BASE, IMG_HOM, IMG_AUG, LBL_HOM, LBL_AUG]:
#     os.makedirs(folder, exist_ok=True)



# # ============================================================
# # 2. Generate Base Marker
# # ============================================================

# def generate_base_markers():
#     # Get AR Marker Dictionary
#     dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)

#     # List to save all information of Marker
#     data = []

#     # Make 500 base images of AR Marker
#     for marker_id in range(500):
#         # Define AR Marker ID 
#         id1 = marker_id * 2
#         id2 = marker_id * 2 + 1

#         # Size of the Canvas and Marker
#         canvas_H = 300
#         canvas_W = 550
#         marker_size = 200

#         # Create white Canvas
#         canvas = np.ones((canvas_H, canvas_W, 3), dtype=np.uint8) * 255

#         # Determine Marker position on the canvas (top left point)
#         x1, y1 = 50, 50
#         x2, y2 = 300, 50

#         # Generate AR Marker
#         m1 = cv2.aruco.generateImageMarker(dictionary, id1, marker_size)
#         m2 = cv2.aruco.generateImageMarker(dictionary, id2, marker_size)

#         # Convert Color CHannel of Marker to RGB
#         m1 = cv2.cvtColor(m1, cv2.COLOR_GRAY2BGR)
#         m2 = cv2.cvtColor(m2, cv2.COLOR_GRAY2BGR)

#         # Stick AR Marker to the Canvas
#         canvas[y1:y1+marker_size, x1:x1+marker_size] = m1
#         canvas[y2:y2+marker_size, x2:x2+marker_size] = m2

#         # Save Marker position on the coordinate format (bounding box)
#         corners1 = np.array([[x1, y1], [x1+marker_size, y1],
#                              [x1+marker_size, y1+marker_size], [x1, y1+marker_size]], np.float32)

#         corners2 = np.array([[x2, y2], [x2+marker_size, y2],
#                              [x2+marker_size, y2+marker_size], [x2, y2+marker_size]], np.float32)

#         # Save the image on the folder
#         filename = f"marker_{id1}_{id2}.png"
#         cv2.imwrite(os.path.join(IMG_BASE, filename), canvas)

#         # Save the all information about marker (image, marker ID, marker position, marker file)
#         data.append({
#             "img": canvas,
#             "id1": id1,
#             "id2": id2,
#             "c1": corners1,
#             "c2": corners2,
#             "name": f"marker_{id1}_{id2}"
#         })

#     return data



# # ============================================================
# # 3. Homography Transforms
# # ============================================================

# def rotate_h(img, c1, c2):
#     # Determine a random number for angle
#     angle = random.uniform(-25, 25)
#     # Get height and witdh of the image
#     h, w = img.shape[:2]
#     # Make a matrix rotation
#     M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
#     # Tranform the image according the matrix position
#     out = cv2.warpAffine(img, M, (w, h))    # warpAffine --> applying geometric transformations to an image

#     # Change corner position coordinate according to image tranformation
#     c1n = cv2.transform(c1.reshape(-1,1,2), M).reshape(-1,2)
#     c2n = cv2.transform(c2.reshape(-1,1,2), M).reshape(-1,2)
#     return out, c1n, c2n


# def zoom_h(img, c1, c2, zoom):
#     h, w = img.shape[:2]
#     # Make a matrix tranformation affine 2D (zoom in and zoom out from center of image)
#     M = np.array([[zoom, 0, (1-zoom)*w/2], [0, zoom, (1-zoom)*h/2]], np.float32)
#     out = cv2.warpAffine(img, M, (w, h))
#     c1n = cv2.transform(c1.reshape(-1,1,2), M).reshape(-1,2)
#     c2n = cv2.transform(c2.reshape(-1,1,2), M).reshape(-1,2)
#     return out, c1n, c2n


# def tilt_h(img, c1, c2):
#     h, w = img.shape[:2]
#     # Determine starting point
#     src = np.array([[0,0],[w,0],[w,h],[0,h]], np.float32)
#     # Determine target corners that have been randomly tilted
#     dst = src + np.random.randint(-40, 40, src.shape).astype(np.float32)    
#     # Find matrix tranformation 
#     H, _ = cv2.findHomography(src, dst)
#     out = cv2.warpPerspective(img, H, (w, h))
#     c1n = cv2.perspectiveTransform(c1.reshape(-1,1,2), H).reshape(-1,2)
#     c2n = cv2.perspectiveTransform(c2.reshape(-1,1,2), H).reshape(-1,2)
#     return out, c1n, c2n



# # ============================================================
# # 4. YOLO Bounding Box
# # ============================================================

# def corners_to_yolo(c, w, h):
#     # Separate coordinate X and coordinate Y from corner coordinate
#     xs, ys = c[:,0], c[:,1]
#     # Find the minimum and maximum coordinate of X and Y
#     xmin, xmax = max(0,min(xs)), min(w-1,max(xs))
#     ymin, ymax = max(0,min(ys)), min(h-1,max(ys))
#     return [(xmin+xmax)/2/w, (ymin+ymax)/2/h,   # calculate center of bounding box. (xmin+xmax) <-- marker position
#             (xmax-xmin)/w, (ymax-ymin)/h]       # calculate bounding box size



# # ============================================================
# # 5.  Label
# # ============================================================

# def save_label(path, id1, id2, c1, c2, w, h):
#     # Determine Bounding Box
#     b1 = corners_to_yolo(c1, w, h)
#     b2 = corners_to_yolo(c2, w, h)
#     # Open file and write
#     with open(path, "w") as f:
#         f.write(f"{id1} {b1[0]} {b1[1]} {b1[2]} {b1[3]}\n")
#         f.write(f"{id2} {b2[0]} {b2[1]} {b2[2]} {b2[3]}\n")



# # ============================================================
# # 6. ImgAug
# # ============================================================

# AUG = iaa.Sequential([
#     iaa.MotionBlur(k=5),
#     iaa.AdditiveGaussianNoise(scale=(0.01*255, 0.03*255)),
#     iaa.Multiply((0.7, 1.3)),
#     iaa.LinearContrast((0.6, 1.4)),
#     iaa.JpegCompression(compression=(40, 90)),
# ], random_order=True)



# # ============================================================
# # 7. Main Pipeline
# # ============================================================

# def run():

#     print("Generating base markers...")
#     base_data = generate_base_markers()

#     idx = 0

#     for item in base_data:

#         base_img = item["img"]
#         id1, id2 = item["id1"], item["id2"]
#         c1, c2   = item["c1"], item["c2"]
#         name     = item["name"]

#         # ----------------------------------------------------
#         # 10 Homography
#         # ----------------------------------------------------
#         hom_list = []

#         for _ in range(4): hom_list.append( rotate_h(base_img, c1, c2) )
#         hom_list.append( zoom_h(base_img, c1, c2, 0.7) )
#         hom_list.append( zoom_h(base_img, c1, c2, 1.3) )
#         for _ in range(4): hom_list.append( tilt_h(base_img, c1, c2) )


#         # ----------------------------------------------------
#         # Save Homography Images
#         # ----------------------------------------------------
#         for i, (himg, hc1, hc2) in enumerate(hom_list):

#             img_name = f"{name}_hom-{i+1}.jpg"
#             img_path = os.path.join(IMG_HOM, img_name)
#             lbl_path = os.path.join(LBL_HOM, img_name.replace(".jpg", ".txt"))

#             cv2.imwrite(img_path, himg)
#             save_label(lbl_path, id1, id2, hc1, hc2, himg.shape[1], himg.shape[0])


#             # ------------------------------------------------
#             # 10 Augmentations
#             # ------------------------------------------------
#             for a in range(10):

#                 aug_img = AUG(image=himg)

#                 aug_name = f"{name}_hom-{i+1}_aug-{a+1}.jpg"
#                 aug_path = os.path.join(IMG_AUG, aug_name)
#                 aug_lbl  = os.path.join(LBL_AUG, aug_name.replace(".jpg", ".txt"))

#                 cv2.imwrite(aug_path, aug_img)
#                 save_label(aug_lbl, id1, id2, hc1, hc2, aug_img.shape[1], aug_img.shape[0])

#                 idx += 1

#     print("Selesai. Foldermu sekarang rapi seperti hidup orang lain.")



# if __name__ == "__main__":
#     run()



import cv2
import numpy as np

# Generate marker
marker_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
marker_size = 200

m1 = cv2.aruco.generateImageMarker(marker_dict, 1, marker_size)
m2 = cv2.aruco.generateImageMarker(marker_dict, 2, marker_size)

m1 = cv2.cvtColor(m1, cv2.COLOR_GRAY2BGR)
m2 = cv2.cvtColor(m2, cv2.COLOR_GRAY2BGR)



def gap_horizontal(img, ratio=0.06):
    h, w = img.shape[:2]
    gap = int(h * ratio)
    cut = h // 2

    top = img[gap:cut]
    bottom = img[cut:-gap]

    return np.vstack([top, bottom])

def gap_vertical(img, ratio=0.06):
    h, w = img.shape[:2]
    gap = int(w * ratio)
    cut = w // 2

    left = img[:, gap:cut]
    right = img[:, cut:-gap]

    return np.hstack([left, right])

def shift_horizontal(img, ratio=0.08):
    h, w = img.shape[:2]
    shift = int(w * ratio)

    # geser ke kanan = sisi kiri hilang
    return img[:, shift:]



def apply_all_variations(img):
    img = gap_horizontal(img)
    img = gap_vertical(img)
    img = shift_horizontal(img)
    return img

marker1_var = gap_horizontal(m1, 0.1)
marker2_var = shift_horizontal(m2, 0.1)
# kamu bisa ganti kombinasi variasinya sesuka hati




h1, w1 = marker1_var.shape[:2]
h2, w2 = marker2_var.shape[:2]

spacing = int(min(w1, w2) * 0.2)

canvas_h = max(h1, h2) + 120
canvas_w = w1 + w2 + spacing + 120

canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255


y1 = (canvas_h - h1) // 2
y2 = (canvas_h - h2) // 2

x1 = (canvas_w - (w1 + w2 + spacing)) // 2
x2 = x1 + w1 + spacing

canvas[y1:y1+h1, x1:x1+w1] = marker1_var
canvas[y2:y2+h2, x2:x2+w2] = marker2_var


cv2.imwrite("final_dual_marker_dynamic_size.png", canvas)
print("Selesai. Ukuran marker jujur, dunia tidak disterilkan.")
