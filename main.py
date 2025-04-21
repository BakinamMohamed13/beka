from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image
import numpy as np
import io
import pickle
import uvicorn

app = FastAPI()

# CORS to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change if you have a specific URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pickle model
with open("XGB-Tuned-balancedPalm.pkl", "rb") as f:  # Change the name if needed
    model = pickle.load(f)

# Image preprocessing
def preprocess_image(image_data):
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = image.resize((224, 224))  # Adjust size as per model's requirement
        image_array = np.array(image) / 255.0  # Normalize to 0-1
        flat = image_array.flatten().reshape(1, -1)  # Flatten to 1D
        return flat
    except Exception as e:
        raise ValueError(f"Image processing error: {str(e)}")

# API Endpoint for prediction
@app.post("/predict")
async def predict_anemia(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        processed_image = preprocess_image(contents)
        prediction = model.predict(processed_image)
        label = "Anemic" if prediction[0] == 1 else "Non-Anemic"
        return {"label": label}
    except Exception as e:
        return {"error": f"Error: {str(e)}"}

# Serve the HTML page for the frontend form
@app.get("/")
async def serve_html():
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Start the server when the script is run directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Ensure port 8000 is used
