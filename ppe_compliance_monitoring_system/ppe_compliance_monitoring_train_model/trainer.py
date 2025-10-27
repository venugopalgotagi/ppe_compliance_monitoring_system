import os.path
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv(verbose=True)

for i in range(1,int(os.getenv("MAX_ITERATIONS"))):
    if not os.path.isdir('runs'):
        model_pt = model = YOLO("yolo11n-cls.pt")
    elif i==2:
        model_pt = f"runs/classify/train/weights/best.pt"
    else:
        model_pt = f"runs/classify/train{i-1}/weights/best.pt"

    model = YOLO(model_pt)
    model.train(data='datasets', epochs=int(os.getenv("EPOCHS")),imgsz=int(os.getenv("IMG_RESOLUTION")))