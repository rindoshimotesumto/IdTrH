from ultralytics import YOLO
import cv2

# Модели
detector = YOLO("best_person.pt")        # модель детекции человека
gender_model = YOLO("best_gender.pt")    # модель пола (классификация)

# Камера
url = "rtsp://YOUR_SERVER_LINK"
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # 1. Детекция людей
    results = detector(frame, conf=0.4)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # 2. Определение пола
            gender_res = gender_model(crop)
            top = gender_res[0].probs.top1  # 0=male,1=female
            gender = "Male" if top == 0 else "Female"

            # Рисуем
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, gender, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0,255,0), 2)

    cv2.imshow("ID System", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
