import json
import time
import cv2
import os

def log_low_activity(person_id, percent, frame):
    timestamp = int(time.time())

    # JSON лог
    data = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "person_id": person_id,
        "activity_percent": percent
    }

    with open("./activity_log.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
