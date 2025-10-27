from skimage import io as skio
import io
from ultralytics import YOLO
import cv2


from fastapi import FastAPI, UploadFile, File

app = FastAPI()

@app.post("/predict_img")
async def predict_img(img: UploadFile = File(...)):
    img = await img.read()
    img = skio.imread(io.BytesIO(img))
    model = YOLO("predictor_model/yolo11n-cls.pt")
    results = model.predict(source=img)

    for r in results:
        print(f"result  {r.probs.top1conf}")
        names = r.names
        probs = r.probs.top1
        print(r)
        is_with_ppe = names[probs]
        #r.save(filename="./output/{}.png")
        if r.probs.top1conf > 0.9:
            return "Looks good..."
        else:
            return "Proper PPE not detected"





