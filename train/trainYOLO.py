from ultralytics import YOLO

def main():
    model = YOLO("yolo11s.pt")

    model.train(
        data="path",
        epochs=70,
        imgsz=960,
        batch=8,
        device=0,
        workers=0,

        # сильные но полезные аугментации:
        hsv=0.015,
        scale=0.5,
        translate=0.1,
        fliplr=0.5,
        brightness=0.3,
        contrast=0.3,

        multi_scale=True
    )

if __name__ == '__main__':
    main()
