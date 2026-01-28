import os
import shutil
import random

# =========================================
# 1️⃣ Buat folder YOLO
BASE = "Dataset_yolo"

IMG_ALL   = os.path.join(BASE, "images/all")
IMG_TRAIN = os.path.join(BASE, "images/train")
IMG_VAL   = os.path.join(BASE, "images/val")

LBL_ALL   = os.path.join(BASE, "labels/all")
LBL_TRAIN = os.path.join(BASE, "labels/train")
LBL_VAL   = os.path.join(BASE, "labels/val")

for d in [IMG_ALL, IMG_TRAIN, IMG_VAL, LBL_ALL, LBL_TRAIN, LBL_VAL]:
    os.makedirs(d, exist_ok=True)

# =========================================
# 2️⃣ Folder sumber
BASE = "Data_Final"

IMG_BASE   = os.path.join(BASE, "images/base")
IMG_BROKEN = os.path.join(BASE, "images/broken")
IMG_HOM    = os.path.join(BASE, "images/hom")
IMG_AUG    = os.path.join(BASE, "images/aug")
IMG_NEG    = os.path.join(BASE, "images/negative")

LBL_BASE   = os.path.join(BASE, "labels/base")
LBL_BROKEN = os.path.join(BASE, "labels/broken")
LBL_HOM    = os.path.join(BASE, "labels/hom")
LBL_AUG    = os.path.join(BASE, "labels/aug")

data_image = [IMG_BASE, IMG_BROKEN, IMG_HOM, IMG_AUG, IMG_NEG]
data_label = [LBL_BASE, LBL_BROKEN, LBL_HOM, LBL_AUG]

# =========================================
# 3️⃣ Salin semua gambar & label ke folder ALL
for folder in data_image:
    for filename in os.listdir(folder):
        path_file = os.path.join(folder, filename)
        if os.path.isfile(path_file):
            shutil.copy2(path_file, os.path.join(IMG_ALL, filename))

for folder in data_label:
    for filename in os.listdir(folder):
        path_file = os.path.join(folder, filename)
        if os.path.isfile(path_file):
            shutil.copy2(path_file, os.path.join(LBL_ALL, filename))

# =========================================
# 4️⃣ Bagi dataset 80% : 20% ke train/val
all_images = [f for f in os.listdir(IMG_ALL) if f.endswith(".jpg")]
random.shuffle(all_images)

split_index = int(0.8 * len(all_images))
train_files = all_images[:split_index]
val_files   = all_images[split_index:]

def copy_files(file_list, img_dest, lbl_dest):
    for filename in file_list:
        # salin gambar
        shutil.copy2(os.path.join(IMG_ALL, filename), os.path.join(img_dest, filename))
        # salin label
        label_name = os.path.splitext(filename)[0] + ".txt"
        label_src = os.path.join(LBL_ALL, label_name)
        label_dst = os.path.join(lbl_dest, label_name)
        if os.path.exists(label_src):
            shutil.copy2(label_src, label_dst)
        else:
            print(f"Warning: label tidak ditemukan untuk {filename}, dilewatkan")


# Salin ke train & val
copy_files(train_files, IMG_TRAIN, LBL_TRAIN)
copy_files(val_files, IMG_VAL, LBL_VAL)

print(f"Dataset terbagi: {len(train_files)} train, {len(val_files)} val")
