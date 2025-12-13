import time
import math
import json

last_pos = {}
last_move_time = {}
activity = {}
logged = set()

STATIC_MAX = 40
STATIC_START = 50
DROP_EVERY = 1
DROP_VALUE = 10


def log_low_activity(person_id, percent):
    data = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "person_id": person_id,
        "activity_percent": percent
    }

    with open("activity_log.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def get_activity_percent(person_id, bbox):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    now = time.time()

    if person_id not in last_pos:
        last_pos[person_id] = (cx, cy)
        last_move_time[person_id] = now
        activity[person_id] = 100
        return 100

    px, py = last_pos[person_id]
    dist = math.hypot(cx - px, cy - py)

    if dist > 5:
        activity[person_id] = 100
        last_move_time[person_id] = now
        logged.discard(person_id)

    else:
        idle_time = now - last_move_time[person_id]

        if idle_time < DROP_EVERY:
            activity[person_id] = STATIC_START
        else:
            if activity[person_id] > STATIC_MAX:
                activity[person_id] = STATIC_MAX
            else:
                drops = int(idle_time // DROP_EVERY)
                activity[person_id] = STATIC_MAX - drops * DROP_VALUE

                if activity[person_id] <= 10 and person_id not in logged:
                    log_low_activity(person_id, activity[person_id])
                    logged.add(person_id)

    if activity[person_id] < 0:
        activity[person_id] = 0

    last_pos[person_id] = (cx, cy)
    return activity[person_id]