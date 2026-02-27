# import os
# import cv2
# from ultralytics import YOLO

# # =========================
# # LOAD YOLO MODEL
# # =========================
# model = YOLO("runs/experiment_6/aruco_yolo_1_training/weights/best.pt")

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
# image_folder = "test_image/base_image"

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

# # =========================
# # LOAD YOLO MODEL
# # =========================
# model = YOLO("runs/experiment_6/aruco_yolo_3_training/weights/best.pt")

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
#     11: "aruco_11",
#     10: "aruco_10",
#     12: "aruco_12",
#     13: "aruco_13",
#     14: "aruco_14",
#     15: "aruco_15",
#     16: "aruco_16",
#     17: "aruco_17",
#     18: "aruco_18",
#     19: "aruco_19",
#     20: "aruco_20",
#     21: "aruco_21",
#     22: "aruco_22",
#     23: "aruco_23",
#     24: "aruco_24",
#     25: "aruco_25",
#     26: "aruco_26",
#     27: "aruco_27",
#     28: "aruco_28",
#     29: "aruco_29",
#     30: "aruco_30",
#     31: "aruco_31",
#     32: "aruco_32",
#     33: "aruco_33",
#     34: "aruco_34",
#     35: "aruco_35",
#     36: "aruco_36",
#     37: "aruco_37",
#     38: "aruco_38",
#     39: "aruco_39",
#     40: "aruco_40",
#     41: "aruco_41",
#     42: "aruco_42",
#     43: "aruco_43",
#     44: "aruco_44",
#     45: "aruco_45",
#     46: "aruco_46",
#     47: "aruco_47",
#     48: "aruco_48",
#     49: "aruco_49",
#     50: "aruco_50",
#     51: "aruco_51",
#     52: "aruco_52",
#     53: "aruco_53",
#     54: "aruco_54",
#     55: "aruco_55",
#     56: "aruco_56",
#     57: "aruco_57",
#     58: "aruco_58",
#     59: "aruco_59",
#     60: "aruco_60",
#     61: "aruco_61",
#     62: "aruco_62",
#     63: "aruco_63",
#     64: "aruco_64",
#     65: "aruco_65",
#     66: "aruco_66",
#     67: "aruco_67",
#     68: "aruco_68",
#     69: "aruco_69",
#     70: "aruco_70",
#     71: "aruco_71",
#     72: "aruco_72",
#     73: "aruco_73",
#     74: "aruco_74",
#     75: "aruco_75",
#     76: "aruco_76",
#     77: "aruco_77",
#     78: "aruco_78",
#     79: "aruco_79",
#     80: "aruco_80",
#     81: "aruco_81",
#     82: "aruco_82",
#     83: "aruco_83",
#     84: "aruco_84",
#     85: "aruco_85",
#     86: "aruco_86",
#     87: "aruco_87",
#     88: "aruco_88",
#     89: "aruco_89",
#     90: "aruco_90",
#     91: "aruco_91",
#     92: "aruco_92",
#     93: "aruco_93",
#     94: "aruco_94",
#     95: "aruco_95",
#     96: "aruco_96",
#     97: "aruco_97",
#     98: "aruco_98",
#     99: "aruco_99",
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

















import os
import cv2
from ultralytics import YOLO
import numpy as np

# =========================
# LOAD YOLO MODEL
# =========================
model = YOLO("runs/experiment_6/aruco_yolo_3_training/weights/best.pt")
  
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
    9: "aruco_9",
    11: "aruco_11",
    10: "aruco_10",
    12: "aruco_12",
    13: "aruco_13",
    14: "aruco_14",
    15: "aruco_15",
    16: "aruco_16",
    17: "aruco_17",
    18: "aruco_18",
    19: "aruco_19",
    20: "aruco_20",
    21: "aruco_21",
    22: "aruco_22",
    23: "aruco_23",
    24: "aruco_24",
    25: "aruco_25",
    26: "aruco_26",
    27: "aruco_27",
    28: "aruco_28",
    29: "aruco_29",
    30: "aruco_30",
    31: "aruco_31",
    32: "aruco_32",
    33: "aruco_33",
    34: "aruco_34",
    35: "aruco_35",
    36: "aruco_36",
    37: "aruco_37",
    38: "aruco_38",
    39: "aruco_39",
    40: "aruco_40",
    41: "aruco_41",
    42: "aruco_42",
    43: "aruco_43",
    44: "aruco_44",
    45: "aruco_45",
    46: "aruco_46",
    47: "aruco_47",
    48: "aruco_48",
    49: "aruco_49",
    50: "aruco_50",
    51: "aruco_51",
    52: "aruco_52",
    53: "aruco_53",
    54: "aruco_54",
    55: "aruco_55",
    56: "aruco_56",
    57: "aruco_57",
    58: "aruco_58",
    59: "aruco_59",
    60: "aruco_60",
    61: "aruco_61",
    62: "aruco_62",
    63: "aruco_63",
    64: "aruco_64",
    65: "aruco_65",
    66: "aruco_66",
    67: "aruco_67",
    68: "aruco_68",
    69: "aruco_69",
    70: "aruco_70",
    71: "aruco_71",
    72: "aruco_72",
    73: "aruco_73",
    74: "aruco_74",
    75: "aruco_75",
    76: "aruco_76",
    77: "aruco_77",
    78: "aruco_78",
    79: "aruco_79",
    80: "aruco_80",
    81: "aruco_81",
    82: "aruco_82",
    83: "aruco_83",
    84: "aruco_84",
    85: "aruco_85",
    86: "aruco_86",
    87: "aruco_87",
    88: "aruco_88",
    89: "aruco_89",
    90: "aruco_90",
    91: "aruco_91",
    92: "aruco_92",
    93: "aruco_93",
    94: "aruco_94",
    95: "aruco_95",
    96: "aruco_96",
    97: "aruco_97",
    98: "aruco_98",
    99: "aruco_99",
}


image_folder = "test_image/all_image"




for filename in os.listdir(image_folder):

    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(image_folder, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"[WARNING] Gagal membaca {filename}")
        continue

    results = model(img, conf=0.25, iou=0.5)

    detections = []

    # -------------------------------
    # KUMPULKAN SEMUA DETEKSI
    # -------------------------------
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        for box in r.boxes:
            coords = box.xyxy.cpu().numpy().astype(int).flatten()
            conf = float(box.conf.cpu().numpy()[0])
            cls_id = int(box.cls.cpu().numpy()[0])

            detections.append({
                "coords": coords,
                "conf": conf,
                "cls_id": cls_id
            })

    # -------------------------------
    # SORT BY CONFIDENCE (TINGGI → RENDAH)
    # -------------------------------
    detections = sorted(detections,
                        key=lambda x: x["conf"],
                        reverse=True)

    # -------------------------------
    # AMBIL MAKSIMAL 2 DETEKSI
    # -------------------------------
    detections = detections[:2]

    # -------------------------------
    # GAMBAR HASIL FINAL
    # -------------------------------
    for det in detections:

        x1, y1, x2, y2 = det["coords"]
        conf = det["conf"]
        cls_id = det["cls_id"]

        label = f"{CLASS_NAMES[cls_id]} ({conf:.2f})"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(
            img,
            label,
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,0,255),
            2
        )

        print(f"[{filename}] Final: {label}")

    # -------------------------------
    # SAVE
    # -------------------------------
    cv2.imshow("YOLO ONLY - Marker ID Detection", img)
    cv2.waitKey(0)

print("[DONE] Semua gambar selesai.")
