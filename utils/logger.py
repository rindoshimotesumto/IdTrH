import json
import datetime
import os

class EventLogger:
    def __init__(self, filename="events.jsonl"):
        self.filename = filename

        # создаём файл, если его нет
        if not os.path.exists(self.filename):
            with open(self.filename, "w", encoding="utf-8") as f:
                pass

    def log(self, gender, confidence, bbox):
        """
        Запись события в формате JSON Line:
        каждая строка — отдельный JSON-объект
        """
        event = {
            "timestamp": datetime.datetime.now().isoformat(),
            "gender": gender,
            "confidence": confidence,
            "bbox": {
                "x1": bbox[0],
                "y1": bbox[1],
                "x2": bbox[2],
                "y2": bbox[3],
            }
        }

        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
