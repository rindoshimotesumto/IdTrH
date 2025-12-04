import cv2
import threading
from ultralytics import YOLO

# ===========================
#   Загружаем модели на GPU
# ===========================
model = YOLO("yolo11s.pt").to("cuda")

# Для пола (если нужно, пока не используем)
gender_model = YOLO("runs/classify/train/weights/best.pt").to("cuda")

tracker_config = "bytetrack.yaml"

# ===========================
#   Список камер
# ===========================
CAMERAS = [
    # одиночная камера
    # "https://hd-auth.skylinewebcams.com/live.m3u8?a=dinthtcfp52sld5jrugp406lb4",

    # несколько камер
    "link",
    "link",
    # "rtsp://user:pass@ip:554/stream",
]


# ===========================
#   Функция обработки камеры
# ===========================
def run_camera(source, cam_id):

    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"[CAM {cam_id}] ❌ Камера не открылась: {source}")
        return

    print(f"[CAM {cam_id}] ✔ Камера запущена")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO tracking
        results = model.track(
            frame,
            device="cuda",
            conf=0.4,
            tracker=tracker_config,
            persist=True
        )

        # отрисовка
        if results:
            r = results[0]
            if r.boxes.id is not None:
                for box in r.boxes:
                    if int(box.cls[0]) != 0:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    track_id = int(box.id[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"ID:{track_id}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow(f"Camera {cam_id}", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyWindow(f"Camera {cam_id}")


# ===========================
#   Запуск: один режим для всех
# ===========================
def main():

    if len(CAMERAS) == 1:
        # --------------------------
        #   Режим одной камеры
        # --------------------------
        print("💡 Режим: ОДНА камера")
        run_camera(CAMERAS[0], 0)

    else:
        # --------------------------
        #   Режим мультикамер
        # --------------------------
        print("📡 Режим: МНОГО КАМЕР")

        threads = []

        for i, cam_source in enumerate(CAMERAS):
            t = threading.Thread(target=run_camera, args=(cam_source, i))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()


# ===========================
#   Запуск программы
# ===========================
if __name__ == "__main__":
    main()
