import cv2
import numpy as np
import os
import random


# ===== GENERATE DATASET ======


# === 1. Make an Folder ===
marker_image = "Dataset/Marker_Image"
os.makedirs(marker_image, exist_ok=True)
marker_label = "Dataset/Marker_label"
os.makedirs(marker_label, exist_ok=True)

# === 2. Generate ArUco Marker ===
def generate_marker():
    # input ArUco Dictionary
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
    
    all_markers = []

    # Generate marker
    for marker_id in range(500):
        # determine the ID of marker
        id1 = marker_id * 2
        id2 = marker_id * 2 + 1

        # Determine the size of canvas  
        canvas_H = 300
        canvas_W = 550
        canvas = np.ones((canvas_H, canvas_W, 3), dtype=np.uint8) * 255
        
        # determine location of the marker on the canvas
        x1, y1 = 50, 50
        x2, y2 = 300, 50

        # marker size
        marker_size = 200

        # make the marker
        marker_1 = cv2.aruco.generateImageMarker(dictionary, id1, marker_size)
        marker_2 = cv2.aruco.generateImageMarker(dictionary, id2, marker_size)

        m1 = cv2.cvtColor(marker_1, cv2.COLOR_GRAY2BGR)
        m2 = cv2.cvtColor(marker_2, cv2.COLOR_GRAY2BGR)

        # stick marker to canvas
        canvas[y1 : y1 + marker_size, x1 : x1 + marker_size] = m1
        canvas[y2 : y2 + marker_size, x2 : x2 + marker_size] = m2


        # determine the corner of every marker
        corners_1 = np.array([
            [x1,           y1          ],  
            [x1+marker_size, y1        ],  
            [x1+marker_size, y1+marker_size],  
            [x1,           y1+marker_size]   
        ], dtype=np.float32)

        corners_2 = np.array([
            [x2,           y2          ],
            [x2+marker_size, y2        ],
            [x2+marker_size, y2+marker_size],
            [x2,           y2+marker_size]
        ], dtype=np.float32)


        # save file on the folder
        img_path = os.path.join(marker_image, f"marker_{marker_id}.png")
        cv2.imwrite(img_path, canvas)
        

        # save corner information
        all_markers.append({
            "id1": id1,
            "id2": id2,
            "image_path": img_path,
            "corners1": corners_1,
            "corners2": corners_2,
            "canvas_w": canvas_W,
            "canvas_h": canvas_H
        })

    return all_markers



# === 3. Create Homography and Wrap the Canvas ===
def apply_homography(img, corners_1, corners_2):
    h, w = img.shape[:2]

    # 1. Vertex point of the canvas before the transform
    src_pts = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)


    # 2. Random Camera Effect

     # a. Rotate (kamera diputar)
    angle = np.random.uniform(-20, 20)  # degree
    center = (w / 2, h / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # b. Zoom (kamera far/close)
    zoom_factor = np.random.uniform(0.7, 1.4)  # <1 far, >1 close

    # c. Tilt (ubah bentuk canvas)
    dst_pts = src_pts.copy()
    dst_pts = dst_pts.astype(np.float32)
    dst_pts += np.random.randint(-40, 40, size=dst_pts.shape).astype(np.float32)

    # d. combine zoom and rotation
    # Scale (zoom)
    zoom_matrix = np.array([
        [zoom_factor, 0, (1 - zoom_factor) * center[0]],
        [0, zoom_factor, (1 - zoom_factor) * center[1]]
    ], dtype=np.float32)

    # Gabungkan rotate dan zoom
    combined_matrix = rot_matrix @ np.vstack([zoom_matrix, [0, 0, 1]])

    # Apply rotate+zoom dulu
    rotated_zoomed = cv2.warpAffine(img, combined_matrix[:2], (w, h))


    # 3. Calculate matriks Homography (H)
    H, _ = cv2.findHomography(src_pts, dst_pts)

    # 4. Transform image
    warped = cv2.warpPerspective(rotated_zoomed, H, (w, h))

    # 5. Transform corner marker
    # a. apply to the corners
    temp_corners_1 = cv2.transform(corners_1.reshape(-1,1,2), combined_matrix[:2]).reshape(-1,2)
    temp_corners_2 = cv2.transform(corners_2.reshape(-1,1,2), combined_matrix[:2]).reshape(-1,2)

    # b. apply homography
    new_corners_c1 = cv2.perspectiveTransform(temp_corners_1.reshape(-1,1,2), H).reshape(-1,2)
    new_corners_c2 = cv2.perspectiveTransform(temp_corners_2.reshape(-1,1,2), H).reshape(-1,2)

    return warped, new_corners_c1, new_corners_c2



# === 4. Give Bounding Box and Label ====
def corners_to_yolo(corners, img_w, img_h):
    # corners shape: (4,2)
    xs = corners[:,0]
    ys = corners[:,1]

    xmin = max(0, min(xs))
    xmax = min(img_w-1, max(xs))
    ymin = max(0, min(ys))
    ymax = min(img_h-1, max(ys))

    # Hitung center dan size untuk YOLO
    x_center = (xmin + xmax) / 2 / img_w
    y_center = (ymin + ymax) / 2 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h

    return x_center, y_center, w, h


# === 5. Save into YOLO dataset (image + 2 labels) ===
def save_yolo_dataset(img, id1, id2, corners1, corners2,
                      marker_image, marker_label, index):

    h, w = img.shape[:2]

    # calculate bounding box
    bbox1 = corners_to_yolo(corners1, w, h)
    bbox2 = corners_to_yolo(corners2, w, h)


    # path file
    img_path = os.path.join(marker_image, f"img_{index}.jpg")
    label_path = os.path.join(marker_label, f"img_{index}.txt")

    # simpan gambar
    cv2.imwrite(img_path, img)

    # simpan label (dua baris)
    with open(label_path, "w") as f:
        f.write(f"{id1} {bbox1[0]} {bbox1[1]} {bbox1[2]} {bbox1[3]}\n")
        f.write(f"{id2} {bbox2[0]} {bbox2[1]} {bbox2[2]} {bbox2[3]}\n")



# === 6. Combine all steps into a full pipeline ===
def run_full_pipeline():

    print("[INFO] Generating base markers...")
    all_markers = generate_marker()

    print("[INFO] Applying transformations and saving YOLO dataset...")
    
    index = 0
    for data in all_markers:

        # load the base clean marker image
        img = cv2.imread(data["image_path"])

        # read the ID and original corners
        id1 = data["id1"]
        id2 = data["id2"]
        c1  = data["corners1"]
        c2  = data["corners2"]

        # homography
        warped, new_c1, new_c2 = apply_homography(img, c1, c2)

        # save dataset
        save_yolo_dataset(
            img        = warped,
            id1        = id1,
            id2        = id2,
            corners1   = new_c1,
            corners2   = new_c2,
            marker_image = marker_image,
            marker_label = marker_label,
            index      = index
        )

        index += 1

    print("[DONE] Dataset generation finished!")



# Run program
if __name__ == "__main__":
    run_full_pipeline()
