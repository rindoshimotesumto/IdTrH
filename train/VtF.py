import cv2
import os
from IdTrH import config

# Папка куда будут сохраняться кадры
SAVE_DIR = "IdTrH/dataset/sleep"

# Создаем папку (если нет)
# os.makedirs(SAVE_DIR, exist_ok=True)

# Загружаем источник видео
cap = cv2.VideoCapture(config.MEDIA["AI"])

i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Не могу прочитать кадр!")
        break

    # сохраняем кадр
    cv2.imwrite(f"{SAVE_DIR}/frame_{i:05d}.jpg", frame)
    i += 1

cap.release()
print("Готово! Сохранено кадров:", i)
