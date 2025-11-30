from ultralytics import YOLO

def main():
    model = YOLO("yolo11m.pt")

    model.train(
        data="person.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=0
    )

if __name__ == '__main__':
    main()
