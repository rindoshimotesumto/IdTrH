import os
import shutil
import random

SOURCE = "UTKFace"
TRAIN = "dataset/train"
VAL = "dataset/val"

os.makedirs(TRAIN, exist_ok=True)
os.makedirs(VAL, exist_ok=True)

male_train = os.path.join(TRAIN, "0")
female_train = os.path.join(TRAIN, "1")
male_val = os.path.join(VAL, "0")
female_val = os.path.join(VAL, "1")

for d in [male_train, female_train, male_val, female_val]:
    os.makedirs(d, exist_ok=True)

files = [f for f in os.listdir(SOURCE) if f.endswith(".jpg")]
random.shuffle(files)

split = int(len(files) * 0.9)
train_files = files[:split]
val_files = files[split:]

def move(files, target):
    for f in files:
        gender = f.split("_")[1]  # 0=male 1=female
        src = os.path.join(SOURCE, f)
        dst = os.path.join(target, gender, f)
        shutil.copy(src, dst)

move(train_files, TRAIN)
move(val_files, VAL)

print("Готово!")