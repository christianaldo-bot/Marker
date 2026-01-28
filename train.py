from ultralytics import YOLO

# 1️⃣ Load model (pretrained)
model = YOLO("yolov8n.pt")

# 2️⃣ Training
model.train(
    data="Dataset_yolo/data.yaml",      # open the folder
    epochs=20,                          # model check all image
    imgsz=640,                          # all imga will be rezise
    batch=8,                            # model check the total image on 1 times
    name="aruco_yolo_1_training"
)