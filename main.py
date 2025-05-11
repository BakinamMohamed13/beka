from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import joblib
from io import BytesIO
from PIL import Image
import os
app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Point to the model inside the 'models' folder
model_path = os.path.join(BASE_DIR, "RF-Tuned-balancedPalm.pkl")
# Load your model
model = joblib.load(model_path)

# Define preprocessing and feature extraction
def preprocess_image(image_data, target_size=(128, 128)):
    image = Image.open(BytesIO(image_data)).convert('RGB')
    image = image.resize(target_size)
    image_np = np.array(image) / 255.0
    return image_np

def extract_color_histogram(image, bins=(8, 8, 8)):
    img = (image * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.reshape(1, -1)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        img = preprocess_image(image_data)
        features = extract_color_histogram(img)
        prob = model.predict_proba(features)[0]
        label_index = int(np.argmax(prob))
        confidence = float(prob[label_index]) * 100

        class_labels = {1: "Non-Anemic", 0: "Anemic"}
        result = {
            "prediction": class_labels[label_index]
        }
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
