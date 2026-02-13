# normal test 


# import os
# import cv2
# from ultralytics import YOLO

# # =========================
# # LOAD YOLO MODEL
# # =========================
# model = YOLO("runs/experiment_5/aruco_yolo_2_training/weights/best.pt")

# # =========================
# # CLASS NAME (HARUS SAMA DENGAN data.yaml)
# # =========================
# CLASS_NAMES = {
#     0: "aruco_0",
#     1: "aruco_1",
#     2: "aruco_2",
#     3: "aruco_3",
#     4: "aruco_4",
#     5: "aruco_5",
#     6: "aruco_6",
#     7: "aruco_7",
#     8: "aruco_8",
#     9: "aruco_9",
# }


# # =========================
# # IMAGE FOLDER
# # =========================
# image_folder = "test_image/mix_image"

# for filename in os.listdir(image_folder):
#     image_path = os.path.join(image_folder, filename)
#     img = cv2.imread(image_path)

#     if img is None:
#         continue

#     # =========================
#     # YOLO INFERENCE
#     # =========================
#     results = model(img, conf=0.25, iou=0.5)

#     for r in results:
#         if r.boxes is None:
#             continue

#         for box in r.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             cls_id = int(box.cls[0])
#             conf = float(box.conf[0])

#             label = f"{CLASS_NAMES[cls_id]} ({conf:.2f})"

#             # draw bbox
#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             # draw label
#             cv2.putText(
#                 img, label,
#                 (x1, y1 - 8),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (0, 0, 255),
#                 2
#             )

#             print(f"[{filename}] Detected: {label}")

#     cv2.imshow("YOLO ONLY - Marker ID Detection", img)
#     cv2.waitKey(0)

# cv2.destroyAllWindows()






# import os
# import cv2
# from ultralytics import YOLO

# # =====================================
# # CONFIG
# # =====================================
# MODEL_PATH   = ("runs/experiment_2/aruco_yolo_1_training/weights/best.pt")
# IMAGE_FOLDER = "test_image/base_image"
# CONF_THRES   = 0.25

# # =====================================
# # LOAD YOLO MODEL
# # =====================================
# model = YOLO(MODEL_PATH)

# # =====================================
# # ARUCO CONFIG
# # =====================================
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
# aruco_params = cv2.aruco.DetectorParameters()
# detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# # =====================================
# # PROCESS IMAGES
# # =====================================
# for filename in sorted(os.listdir(IMAGE_FOLDER)):
#     image_path = os.path.join(IMAGE_FOLDER, filename)
#     img = cv2.imread(image_path)

#     if img is None:
#         print(f"[SKIP] Cannot read image: {filename}")
#         continue

#     orig = img.copy()
#     print(f"\n📷 Processing: {filename}")

#     # =============================
#     # YOLO DETECTION
#     # =============================
#     results = model(img, conf=CONF_THRES, verbose=False)

#     for r in results:
#         if r.boxes is None:
#             print("  ❌ No YOLO detection")
#             continue

#         for bi, box in enumerate(r.boxes):
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             conf = float(box.conf[0])

#             # Crop marker region
#             crop = orig[y1:y2, x1:x2]
#             if crop.size == 0:
#                 print("  ⚠️ Empty crop, skipped")
#                 continue

#             gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

#             # =============================
#             # ARUCO DETECTION
#             # =============================
#             corners, ids, rejected = detector.detectMarkers(gray)

#             # Draw YOLO bounding box
#             cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             # YOLO confidence
#             cv2.putText(
#                 orig, f"YOLO {conf:.2f}",
#                 (x2 - 90, y1 - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6,
#                 (255, 0, 0), 2
#             )

#             if ids is not None:
#                 ids = ids.flatten()
#                 print(f"  ✅ Box {bi}: ArUco IDs = {ids.tolist()}")

#                 for i, marker_id in enumerate(ids):
#                     cv2.putText(
#                         orig,
#                         f"ID: {marker_id}",
#                         (x1, y1 - 10 - i * 25),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.8,
#                         (0, 0, 255),
#                         2
#                     )
#             else:
#                 print(f"  ⚠️ Box {bi}: ArUco NOT detected")
#                 cv2.putText(
#                     orig,
#                     "ID: Not detected",
#                     (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.8,
#                     (0, 0, 255),
#                     2
#                 )

#     # =============================
#     # SHOW RESULT
#     # =============================
#     cv2.imshow("YOLO + ArUco Detection", orig)
#     key = cv2.waitKey(0)

#     if key == 27:  # ESC
#         break



# cv2.destroyAllWindows()





# import os
# import cv2
# from ultralytics import YOLO

# # =========================
# # LOAD YOLO MODEL
# # =========================
# model = YOLO("runs/experiment_3/aruco_yolo_1_training/weights/best.pt")

# # =========================
# # CLASS NAMES
# # =========================
# CLASS_NAMES = {
#     0: "aruco_0",
#     1: "aruco_1",
#     2: "aruco_2",
#     3: "aruco_3",
#     4: "aruco_4",
#     5: "aruco_5",
#     6: "aruco_6",
#     7: "aruco_7",
#     8: "aruco_8",
#     9: "aruco_9",
# }

# # =========================
# # ROTATION CONFIG
# # =========================
# ROTATIONS = {
#     0: None,
#     90: cv2.ROTATE_90_CLOCKWISE,
#     180: cv2.ROTATE_180,
#     270: cv2.ROTATE_90_COUNTERCLOCKWISE
# }

# # =========================
# # IMAGE FOLDER
# # =========================
# image_folder = "test_image/rotate_image/rotate_image"

# for filename in os.listdir(image_folder):
#     img_original = cv2.imread(os.path.join(image_folder, filename))
#     if img_original is None:
#         continue

#     best_total_conf = 0
#     best_img = None
#     best_results = None
#     best_angle = None

#     # =========================
#     # ROTATE + YOLO INFERENCE
#     # =========================
#     for angle, rot_code in ROTATIONS.items():
#         img = img_original if rot_code is None else cv2.rotate(img_original, rot_code)
#         results = model(img, conf=0.25, iou=0.5)

#         total_conf = 0
#         if results and results[0].boxes is not None:
#             for box in results[0].boxes:
#                 total_conf += float(box.conf[0])

#         # cek apakah total confidence rotasi ini lebih tinggi
#         if total_conf > best_total_conf:
#             best_total_conf = total_conf
#             best_img = img.copy()
#             best_results = results
#             best_angle = angle

#     # =========================
#     # TAMPILKAN HASIL TERBAIK
#     # =========================
#     if best_results is None or best_results[0].boxes is None:
#         print(f"[{filename}] No detection found")
#         continue

#     for box in best_results[0].boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         cls_id = int(box.cls[0])
#         conf = float(box.conf[0])
#         label = f"{CLASS_NAMES[cls_id]} ({conf:.2f})"

#         cv2.rectangle(best_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(
#             best_img, label,
#             (x1, y1 - 8),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (0, 0, 255),
#             2
#         )

#         print(f"[{filename}] Detected: {label} | Rot {best_angle}°")

#     cv2.imshow("FINAL DETECTION", best_img)
#     cv2.waitKey(0)

# cv2.destroyAllWindows()





# import os
# import cv2
# from ultralytics import YOLO
# import numpy as np

# # -------------------------------
# # SETUP MODEL
# # -------------------------------
# model = YOLO("runs/experiment_3/aruco_yolo_1_training/weights/best.pt")
# CLASS_NAMES = model.names

# image_folder = "test_image/rotate_image"
# output_folder = "results"
# os.makedirs(output_folder, exist_ok=True)

# # -------------------------------
# # LOOP SEMUA GAMBAR
# # -------------------------------
# for filename in os.listdir(image_folder):
#     if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
#         continue

#     img_path = os.path.join(image_folder, filename)
#     img = cv2.imread(img_path)
#     if img is None:
#         print(f"[WARNING] Gagal membaca {filename}")
#         continue

#     # -------------------------------
#     # STAGE 1: DETEKSI BOUNDING BOX AWAL
#     # -------------------------------
#     results_stage1 = model(img, conf=0.25, iou=0.5)

#     crop_count = 0
#     for r in results_stage1:
#         if r.boxes is None or len(r.boxes) == 0:
#             continue

#         for box in r.boxes:
#             # Ambil koordinat bounding box (convert tensor ke numpy)
#             coords = box.xyxy.cpu().numpy().astype(int).flatten()
#             x1, y1, x2, y2 = coords

#             # Pastikan koordinat tidak keluar dari gambar
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

#             # Crop + zoom (2x)
#             cropped = img[y1:y2, x1:x2]
#             if cropped.size == 0:
#                 continue  # skip jika crop kosong

#             zoomed = cv2.resize(cropped, (0,0), fx=2, fy=2)

#             # -------------------------------
#             # STAGE 2: DETEKSI PADA ZOOMED CROP
#             # -------------------------------
#             results_stage2 = model(zoomed, conf=0.25, iou=0.5)

#             for r2 in results_stage2:
#                 if r2.boxes is None or len(r2.boxes) == 0:
#                     continue

#                 for box2 in r2.boxes:
#                     coords2 = box2.xyxy.cpu().numpy().astype(int).flatten()
#                     x1c, y1c, x2c, y2c = coords2

#                     cls_id = int(box2.cls.cpu().numpy()[0])
#                     conf = float(box2.conf.cpu().numpy()[0])
#                     label = f"{CLASS_NAMES[cls_id]} ({conf:.2f})"

#                     # Gambar bounding box dan label pada zoomed crop
#                     cv2.rectangle(zoomed, (x1c, y1c), (x2c, y2c), (0,255,0), 2)
#                     cv2.putText(zoomed, label, (x1c, y1c-10), 
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

#                     print(f"[{filename}] Marker Detected: {label}")

#             # -------------------------------
#             # Simpan hasil zoom + deteksi
#             # -------------------------------
#             crop_count += 1
#             out_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_crop{crop_count}.jpg")
#             cv2.imwrite(out_path, zoomed)
#             print(f"[INFO] Saved: {out_path}")

# print("[DONE] Semua gambar selesai diproses.")



import os
import cv2
from ultralytics import YOLO
import numpy as np

# -------------------------------
# SETUP MODEL
# -------------------------------
model = YOLO("runs/experiment_3/aruco_yolo_1_training/weights/best.pt")
CLASS_NAMES = model.names

image_folder = "test_image/mix_image"
output_folder = "results"
os.makedirs(output_folder, exist_ok=True)

# -------------------------------
# FUNCTION ROTATE
# -------------------------------
def rotate_image(image, angle):
    if angle == 0:
        return image
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

# -------------------------------
# LOOP SEMUA GAMBAR
# -------------------------------
for filename in os.listdir(image_folder):

    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(image_folder, filename)
    img = cv2.imread(img_path)

    if img is None:
        continue

    # -------------------------------
    # STAGE 1: BOUNDING BOX DETECTION
    # -------------------------------
    results_stage1 = model(img, conf=0.25, iou=0.5)

    crop_count = 0

    for r in results_stage1:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        for box in r.boxes:

            coords = box.xyxy.cpu().numpy().astype(int).flatten()
            x1, y1, x2, y2 = coords

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

            cropped = img[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            zoomed = cv2.resize(cropped, (640, 640))

            best_conf = 0
            best_cls = None
            best_bbox = None

            # -------------------------------
            # STAGE 2: ROTATION LOOP
            # -------------------------------
            best_conf = 0
            best_cls = None
            best_bbox = None
            best_image = None   # <-- save the best rotation

            for angle in [0, 90, 180, 270]:

                rotated = rotate_image(zoomed, angle)
                results_stage2 = model(rotated, conf=0.25, iou=0.5)

                for r2 in results_stage2:
                    if r2.boxes is None or len(r2.boxes) == 0:
                        continue

                    for box2 in r2.boxes:

                        conf2 = float(box2.conf.cpu().numpy()[0])
                        cls_id2 = int(box2.cls.cpu().numpy()[0])

                        if conf2 > best_conf:
                            best_conf = conf2
                            best_cls = cls_id2
                            best_bbox = box2.xyxy.cpu().numpy().astype(int).flatten()
                            best_image = rotated.copy()  # <-- simpan rotasi terbaik


            # -------------------------------
            # DRAW FINAL BEST RESULT
            # -------------------------------
            if best_cls is not None and best_image is not None:

                label = f"{CLASS_NAMES[best_cls]} ({best_conf:.2f})"

                x1c, y1c, x2c, y2c = best_bbox
                cv2.rectangle(best_image, (x1c, y1c), (x2c, y2c), (0,255,0), 2)

                cv2.putText(
                    best_image,
                    f"FINAL ID: {label}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0,0,255),
                    3
                )

                print(f"[{filename}] Final ID: {label}")

                crop_count += 1
                out_path = os.path.join(
                    output_folder,
                    f"{os.path.splitext(filename)[0]}_crop{crop_count}.jpg"
                )

                cv2.imwrite(out_path, best_image)
                print(f"[INFO] Saved: {out_path}")



