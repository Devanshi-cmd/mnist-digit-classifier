from fastapi import FastAPI, UploadFile, File
from tensorflow import keras
from PIL import Image
import numpy as np
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MNIST Digit Classifier API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


model = keras.models.load_model("mnist_ann.keras")

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
    img = img.resize((28, 28))
    img_arr = np.array(img) / 255.0
    img_arr = img_arr.reshape(1, 28, 28)  # ANN expects (batch, 28, 28)
    return img_arr

@app.get("/")
def root():
    return {"message": "MNIST digit classifier API. POST an image to /predict"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_arr = preprocess(image_bytes)
    pred = model.predict(img_arr)
    digit = int(pred.argmax())
    confidence = float(pred.max())

    return {
        "predicted_digit": digit,
        "confidence": round(confidence * 100, 2)
    }
