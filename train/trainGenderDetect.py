from ultralytics import YOLO

def main():
    # Загружаем классификационную модель
    model = YOLO("yolo11s-cls.pt")

    model.train(
        data = "gender_dataset",
        epochs=20,
        imgsz=224,
        batch=32,
        device=0,
        workers=2,
        patience=5,
        augment=True,
    )

if __name__ == "__main__":
    main()
