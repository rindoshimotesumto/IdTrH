import os
import shutil
import pandas as pd

# === ТВОИ ПУТИ ===
CSV_PATH = "img/fairface-labels.csv"  # путь к CSV (переименуй если он другой)
TRAIN_DIR = "img/fairface-img-margin025-trainval/train"
VAL_DIR = "img/fairface-img-margin025-trainval/val"

OUT = "gender_dataset"

# создаём структуру
paths = [
    f"{OUT}/train/male",
    f"{OUT}/train/female",
    f"{OUT}/val/male",
    f"{OUT}/val/female",
]

for p in paths:
    os.makedirs(p, exist_ok=True)

# читаем CSV
df = pd.read_csv(CSV_PATH)

# фильтруем только нужное
df = df[df["gender"].isin(["Male", "Female"])]

print("Всего записей:", len(df))

# создаём словарь: имя файла → пол
gender_map = dict(zip(df["file"], df["gender"]))

# функция копирования
def move(file_name, src_dir, dst_dir):
    src = os.path.join(src_dir, file_name)
    if os.path.isfile(src):
        shutil.copy(src, dst_dir)

# обрабатываем train
for file in os.listdir(TRAIN_DIR):
    gender = gender_map.get(file, None)
    if gender:
        dst = f"{OUT}/train/{gender.lower()}/"
        move(file, TRAIN_DIR, dst)

# обрабатываем val
for file in os.listdir(VAL_DIR):
    gender = gender_map.get(file, None)
    if gender:
        dst = f"{OUT}/val/{gender.lower()}/"
        move(file, VAL_DIR, dst)

print("Готово! Датасет собран в папке gender_dataset")
