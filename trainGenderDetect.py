from ultralytics import YOLO

def main():
    model = YOLO("runs/classify/train3/weights/last.pt")  # путь к last.pt

    model.train(
        data="gender_dataset",
        epochs=30,      # СТОЛЬКО эпох В СУММЕ нужно, не добавочно
        imgsz=224,
        batch=16,
        device=0,
        workers=0,      # для винды
        resume=True     # <---- ВАЖНО
    )

if __name__ == "__main__":
    main()
