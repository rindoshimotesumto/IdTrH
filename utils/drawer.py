import cv2

class Drawer:

    def draw(self, frame, persons, genders, person_count):
        """
        frame: кадр
        persons: [{'id': int, 'bbox': (x1,y1,x2,y2)}]
        genders: ['Erkak', 'Ayol']
        person_count: int
        """

        for i, p in enumerate(persons):
            x1, y1, x2, y2 = p["bbox"]
            gender = genders[i]

            # Бокс
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            # Текст: ID + gender
            text = f"ID {p['id']} | {gender}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,255,0), 2)

        # Счётчик людей (вверху слева)
        cv2.putText(frame, f"Odam soni: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,0), 2)

        return frame
