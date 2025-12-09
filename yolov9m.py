import threading
import cv2
from ultralytics import YOLO
from config import CAMERAS

def run_camera(source, cam_id):
    model = YOLO("yolov9m.pt").to("cuda")  # модель внутри потока!
    
    cap = cv2.VideoCapture(source)

    while True:
        hCount = 0
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame, classes=[0], conf=0.4, imgsz=512, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                hCount+=1

            cv2.putText(frame, f"Odamlar: {hCount}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow(f"Camera {cam_id}", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()


def main():
    threads = []

    for i, cam_source in enumerate(CAMERAS):
        t = threading.Thread(target=run_camera, args=(cam_source, i), daemon=True)
        t.start()
        threads.append(t)

    # главный поток не блокирует, но приложение живет
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
