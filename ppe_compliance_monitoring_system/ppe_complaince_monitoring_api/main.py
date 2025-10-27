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
    model = YOLO("predictor_model/best.pt")
    results = model.predict(img)

    for r in results:
        print(r)
        names = r.names
        probs = r.probs.top1
        is_with_ppe = names[probs]
        r.save(filename="./output/{}.png")
        if is_with_ppe == 'with_ppe':
            return "Looks good..."
        else:
            return "Proper PPE not detected"





