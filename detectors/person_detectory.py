from ultralytics import YOLO
from utils.filters import is_valid_person

class PersonDetector:
    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path).to("cuda")

    def detect(self, frame):
        """
        Возвращает список людей: [{'bbox': (x1,y1,x2,y2), 'conf': 0.88}, ...]
        """
        results = self.model(frame, device=0, verbose=False)
        persons = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if not is_valid_person(cls, conf, x1, y1, x2, y2):
                    continue

                persons.append({
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf
                })

        return persons
