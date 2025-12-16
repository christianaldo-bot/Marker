# import cv2
# import numpy as np
# import os
# import imgaug.augmenters as iaa
# import random


# # ===== GENERATE DATASET ======


# # === 1. Setup Folder ===

# # Main folder
# Main_folder = "Dataset"

# marker_image = "Dataset/Marker_Image"
# os.makedirs(marker_image, exist_ok=True)
# marker_label = "Dataset/Marker_label"
# os.makedirs(marker_label, exist_ok=True)

# # === 2. Generate ArUco Marker ===
# def generate_marker():
#     # input ArUco Dictionary
#     dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
    
#     all_markers = []

#     # Generate marker
#     for marker_id in range(500):
#         # determine the ID of marker
#         id1 = marker_id * 2
#         id2 = marker_id * 2 + 1

#         # Determine the size of canvas  
#         canvas_H = 300
#         canvas_W = 550
#         canvas = np.ones((canvas_H, canvas_W, 3), dtype=np.uint8) * 255
        
#         # determine location of the marker on the canvas
#         x1, y1 = 50, 50
#         x2, y2 = 300, 50

#         # marker size
#         marker_size = 200

#         # make the marker
#         marker_1 = cv2.aruco.generateImageMarker(dictionary, id1, marker_size)
#         marker_2 = cv2.aruco.generateImageMarker(dictionary, id2, marker_size)

#         m1 = cv2.cvtColor(marker_1, cv2.COLOR_GRAY2BGR)
#         m2 = cv2.cvtColor(marker_2, cv2.COLOR_GRAY2BGR)

#         # stick marker to canvas
#         canvas[y1 : y1 + marker_size, x1 : x1 + marker_size] = m1
#         canvas[y2 : y2 + marker_size, x2 : x2 + marker_size] = m2


#         # determine the corner of every marker
#         corners_1 = np.array([
#             [x1,           y1          ],  
#             [x1+marker_size, y1        ],  
#             [x1+marker_size, y1+marker_size],  
#             [x1,           y1+marker_size]   
#         ], dtype=np.float32)

#         corners_2 = np.array([
#             [x2,           y2          ],
#             [x2+marker_size, y2        ],
#             [x2+marker_size, y2+marker_size],
#             [x2,           y2+marker_size]
#         ], dtype=np.float32)


#         # save file on the folder
#         img_path = os.path.join(marker_image, f"marker_{marker_id}.png")
#         cv2.imwrite(img_path, canvas)
        

#         # save corner information
#         all_markers.append({
#             "id1": id1,
#             "id2": id2,
#             "image_path": img_path,
#             "corners1": corners_1,
#             "corners2": corners_2,
#             "canvas_w": canvas_W,
#             "canvas_h": canvas_H
#         })

#     return all_markers



# # === 3. Create Homography and Wrap the Canvas ===
# def apply_homography(img, corners_1, corners_2):
#     h, w = img.shape[:2]

#     # 1. Vertex point of the canvas before the transform
#     src_pts = np.array([
#         [0, 0],
#         [w, 0],
#         [w, h],
#         [0, h]
#     ], dtype=np.float32)


#     # 2. Random Camera Effect

#      # a. Rotate (kamera diputar)
#     angle = np.random.uniform(-20, 20)  # degree
#     center = (w / 2, h / 2)
#     rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

#     # b. Zoom (kamera far/close)
#     zoom_factor = np.random.uniform(0.7, 1.4)  # <1 far, >1 close

#     # c. Tilt (ubah bentuk canvas)
#     dst_pts = src_pts.copy()
#     dst_pts = dst_pts.astype(np.float32)
#     dst_pts += np.random.randint(-40, 40, size=dst_pts.shape).astype(np.float32)

#     # d. combine zoom and rotation
#     # Scale (zoom)
#     zoom_matrix = np.array([
#         [zoom_factor, 0, (1 - zoom_factor) * center[0]],
#         [0, zoom_factor, (1 - zoom_factor) * center[1]]
#     ], dtype=np.float32)

#     # Gabungkan rotate dan zoom
#     combined_matrix = rot_matrix @ np.vstack([zoom_matrix, [0, 0, 1]])

#     # Apply rotate+zoom dulu
#     rotated_zoomed = cv2.warpAffine(img, combined_matrix[:2], (w, h))


#     # 3. Calculate matriks Homography (H)
#     H, _ = cv2.findHomography(src_pts, dst_pts)

#     # 4. Transform image
#     warped = cv2.warpPerspective(rotated_zoomed, H, (w, h))

#     # 5. Transform corner marker
#     # a. apply to the corners
#     temp_corners_1 = cv2.transform(corners_1.reshape(-1,1,2), combined_matrix[:2]).reshape(-1,2)
#     temp_corners_2 = cv2.transform(corners_2.reshape(-1,1,2), combined_matrix[:2]).reshape(-1,2)

#     # b. apply homography
#     new_corners_c1 = cv2.perspectiveTransform(temp_corners_1.reshape(-1,1,2), H).reshape(-1,2)
#     new_corners_c2 = cv2.perspectiveTransform(temp_corners_2.reshape(-1,1,2), H).reshape(-1,2)

#     return warped, new_corners_c1, new_corners_c2



# # === 4. Give Bounding Box and Label ====
# def corners_to_yolo(corners, img_w, img_h):
#     # corners shape: (4,2)
#     xs = corners[:,0]
#     ys = corners[:,1]

#     xmin = max(0, min(xs))
#     xmax = min(img_w-1, max(xs))
#     ymin = max(0, min(ys))
#     ymax = min(img_h-1, max(ys))

#     # Hitung center dan size untuk YOLO
#     x_center = (xmin + xmax) / 2 / img_w
#     y_center = (ymin + ymax) / 2 / img_h
#     w = (xmax - xmin) / img_w
#     h = (ymax - ymin) / img_h

#     return x_center, y_center, w, h


# # === 5. Save into YOLO dataset (image + 2 labels) ===
# def save_yolo_dataset(img, id1, id2, corners1, corners2,
#                       marker_image, marker_label, index):

#     h, w = img.shape[:2]

#     # calculate bounding box
#     bbox1 = corners_to_yolo(corners1, w, h)
#     bbox2 = corners_to_yolo(corners2, w, h)


#     # path file
#     img_path = os.path.join(marker_image, f"img_{index}.jpg")
#     label_path = os.path.join(marker_label, f"img_{index}.txt")

#     # simpan gambar
#     cv2.imwrite(img_path, img)

#     # simpan label (dua baris)
#     with open(label_path, "w") as f:
#         f.write(f"{id1} {bbox1[0]} {bbox1[1]} {bbox1[2]} {bbox1[3]}\n")
#         f.write(f"{id2} {bbox2[0]} {bbox2[1]} {bbox2[2]} {bbox2[3]}\n")



# # === 6. Augmentation using Imgaug ===

# def apply_imgaug(img):

#     seq = iaa.Sequential([

#         # Blur simulasi kamera goyang / fokus meleset
#         iaa.Sometimes(0.4, iaa.MotionBlur(k=5)),
#         iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0.5, 1.2))),

#         # Noise
#         iaa.Sometimes(0.4, iaa.AdditiveGaussianNoise(scale=(0.01*255, 0.03*255))),

#         # Brightness
#         iaa.Sometimes(0.4, iaa.Multiply((0.7, 1.3))),  

#         # Contrast
#         iaa.Sometimes(0.4, iaa.LinearContrast((0.6, 1.4))),

#         # JPEG compression - simulasi HP kualitas buruk
#         iaa.Sometimes(0.5, iaa.JpegCompression(compression=(40, 90))),

#         # Optional sharpening
#         iaa.Sometimes(0.2, iaa.Sharpen(alpha=(0.1, 0.3), lightness=(0.9, 1.1))),

#     ], random_order=True)

#     return seq(image=img)


# # === 6. Combine all steps into a full pipeline ===
# def run_full_pipeline():

#     print("[INFO] Generating base markers...")
#     all_markers = generate_marker()

#     print("[INFO] Applying transformations and saving YOLO dataset...")
    
#     index = 0
#     for data in all_markers:

#         # load the base clean marker image
#         img = cv2.imread(data["image_path"])

#         # read the ID and original corners
#         id1 = data["id1"]
#         id2 = data["id2"]
#         c1  = data["corners1"]
#         c2  = data["corners2"]

#         # homography
#         warped, new_c1, new_c2 = apply_homography(img, c1, c2)

#         # save dataset
#         save_yolo_dataset(
#             img        = warped,
#             id1        = id1,
#             id2        = id2,
#             corners1   = new_c1,
#             corners2   = new_c2,
#             marker_image = marker_image,
#             marker_label = marker_label,
#             index      = index
#         )

#         index += 1

#     print("[DONE] Dataset generation finished!")



# # Run program
# if __name__ == "__main__":
#     run_full_pipeline()


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
