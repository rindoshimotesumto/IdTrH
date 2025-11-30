import numpy as np

class Track:
    def __init__(self, track_id, bbox):
        self.id = track_id
        self.bbox = bbox  # (x1,y1,x2,y2)
        self.lost = 0

class BYTETracker:
    def __init__(self, max_lost=10, iou_threshold=0.3):
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.tracks = {}

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    def update(self, detections):
        """
        detections: list of dict -> [{ 'bbox': (x1,y1,x2,y2), 'conf': 0.88 }]
        """

        # 1) если нет треков → создать все
        if len(self.tracks) == 0:
            for det in detections:
                self.tracks[self.next_id] = Track(self.next_id, det["bbox"])
                self.next_id += 1
            return self.tracks

        # 2) матчинг треков
        used_tracks = set()
        used_dets = set()

        for tid, track in list(self.tracks.items()):
            best_iou = 0
            best_det = None

            for i, det in enumerate(detections):
                if i in used_dets:
                    continue

                iou = self.iou(track.bbox, det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_det = i

            if best_iou > self.iou_threshold:
                # обновляем трек
                track.bbox = detections[best_det]["bbox"]
                track.lost = 0
                used_tracks.add(tid)
                used_dets.add(best_det)
            else:
                track.lost += 1

        # 3) удаляем потерянные треки
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].lost > self.max_lost:
                del self.tracks[tid]

        # 4) создаём новые треки для оставшихся детекций
        for i, det in enumerate(detections):
            if i not in used_dets:
                self.tracks[self.next_id] = Track(self.next_id, det["bbox"])
                self.next_id += 1

        return self.tracks