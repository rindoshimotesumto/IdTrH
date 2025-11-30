import cv2
from camera.camera import Camera
from detectors.person_detectory import PersonDetector
from classifiers.gender_classifier import GenderClassifier
from utils.logger import EventLogger
from utils.drawer import Drawer
from trackers.bytetrack import BYTETracker

camera = Camera()
detector = PersonDetector()
gender_cls = GenderClassifier()
logger = EventLogger()
drawer = Drawer()
tracker = BYTETracker()

while True:
    frame = camera.read()
    if frame is None:
        break

    # 1) Детекция людей
    detections = detector.detect(frame)      # [{bbox:(..), conf:0.92}]

    # 2) Трекинг → присвоение ID
    tracks = tracker.update(detections)

    # 3) Формируем список persons с ID
    persons = []
    genders = []

    for tid, track in tracks.items():
        bbox = track.bbox

        # Добавляем объект
        persons.append({
            "id": tid,
            "bbox": bbox,
            "conf": 1.0  # если нужно, можно расширить
        })

        # 4) Классификация пола
        gender = gender_cls.analyze(bbox, frame)
        genders.append(gender)

        # 5) Логируем
        logger.log(gender, 1.0, bbox)

    # 6) Счётчик людей
    person_count = len(persons)

    # 7) Рисуем
    output = drawer.draw(frame, persons, genders, person_count)

    cv2.imshow("AI System", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()