import torch
import cv2
import numpy as np
from classifiers.gender_model import GenderNet

device = "cuda" if torch.cuda.is_available() else "cpu"

# Загружаем нашу модель пола
model = GenderNet().to(device)
model.load_state_dict(torch.load("gender_model.pth", map_location=device))
model.eval()

classes = ["Male", "Female"]


class GenderClassifier:

    def analyze(self, person_bbox, frame):
        """
        Получает bbox и кадр -> возвращает 'Male', 'Female', 'Unknown'
        """
        x1, y1, x2, y2 = person_bbox

        # Вырезаем верхнюю половину тела (голова + плечи)
        head = frame[y1 : y1 + (y2 - y1) // 2, x1 : x2]

        try:
            # Преобразуем в тензор
            img = cv2.resize(head, (128, 128))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = torch.tensor(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img)
                pred = torch.argmax(output, 1).item()

            return classes[pred]
        except:
            return "Unknown"
