from ultralytics import YOLO





def main():
    model = YOLO("yolov8m.pt")  # 🔥 ganti ke model medium

    model.train(
        data="Dataset_yolo/experiment_9/data.yaml",
        epochs=50,
        imgsz=1280,
        batch=1,
        device=0,
        workers=2,
        name="experiment_9/aruco_yolo_1_training_m"
    )

if __name__ == "__main__":
    main()


# from ultralytics import YOLO

# def main():
#     model = YOLO("runs/detect/experiment_8/aruco_yolo_1_training_m/weights/last.pt")

#     model.train(
#         data="Dataset_yolo/experiment_8/data.yaml",
#         epochs=50,
#         imgsz=832,
#         batch=4,
#         device=0,
#         workers = 2,
#         name="experiment_8/aruco_yolo_1_training_m",
#         resume=True
#     )

# if __name__ == "__main__":
#     main()


# import torch
# import ultralytics
# import os

# print("=== ULTRALYTICS ===")
# print(ultralytics.__version__)

# print("\n=== PYTORCH ===")
# print(torch.__version__)

# print("\n=== CUDA ===")
# print("Available:", torch.cuda.is_available())

# if torch.cuda.is_available():
#     print("GPU:", torch.cuda.get_device_name(0))
#     print("GPU Count:", torch.cuda.device_count())
# else:
#     print("GPU: Not detected")

# print("\n=== SYSTEM CHECK ===")
# os.system("nvidia-smi")




# from ultralytics import YOLO

# def main():
#     model = YOLO("runs/detect/experiment_7/aruco_yolo_3_training/weights/last.pt")

#     model.train(
#         data="Dataset_yolo/experiment_8/data.yaml",
#         epochs=50,
#         imgsz=640,
#         batch=8,
#         device=0,
#         workers = 2,
#         name="experiment_7/aruco_yolo_4_training"
#     )

# if __name__ == "__main__":
#     main()

