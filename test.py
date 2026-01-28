# import os
# import cv2
# from ultralytics import YOLO

# # =====================================
# # CONFIG
# # =====================================
# MODEL_PATH   = "runs/detect/aruco_yolo/weights/best.pt"
# IMAGE_FOLDER = "test_image"
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





import os
import cv2
from ultralytics import YOLO

# =========================
# LOAD YOLO MODEL
# =========================
model = YOLO("runs/detect/aruco_yolo/weights/best.pt")

# =========================
# CLASS NAME (HARUS SAMA DENGAN data.yaml)
# =========================
CLASS_NAMES = {
    0: "aruco_0",
    1: "aruco_1",
    2: "aruco_2",
    3: "aruco_3",
    4: "aruco_4",
    5: "aruco_5",
    6: "aruco_6",
    7: "aruco_7",
    8: "aruco_8",
}

# =========================
# IMAGE FOLDER
# =========================
image_folder = "test_image"

for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)
    img = cv2.imread(image_path)

    if img is None:
        continue

    # =========================
    # YOLO INFERENCE
    # =========================
    results = model(img, conf=0.25, iou=0.5)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            label = f"{CLASS_NAMES[cls_id]} ({conf:.2f})"

            # draw bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # draw label
            cv2.putText(
                img, label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            print(f"[{filename}] Detected: {label}")

    cv2.imshow("YOLO ONLY - Marker ID Detection", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
