# MNIST Digit Classifier

A full-stack handwritten digit recognizer — draw a digit on a canvas and get a live prediction from a trained neural network.

**Try it live:** https://devanshi-cmd.github.io/mnist-digit-classifier/digit-predictor.html

## How it works

1. Draw a digit (0–9) on the canvas
2. The drawing is sent to a FastAPI backend as an image
3. A Keras ANN model (trained on MNIST) preprocesses and classifies it
4. The predicted digit and confidence score are returned and displayed

## Tech stack

- **Backend:** FastAPI, TensorFlow/Keras, Pillow, NumPy
- **Frontend:** HTML5 Canvas, vanilla JavaScript
- **Deployment:** Render (API) + GitHub Pages (frontend)

## Running locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open `digit-predictor.html` in your browser and point the API URL field at `http://localhost:8000/predict`.

## API

**POST** `/predict` — accepts an image file (multipart/form-data), returns:
```json
{
  "predicted_digit": 7,
  "confidence": 94.32
}
```

> Note: hosted on Render's free tier, so the API may take 30–50s to wake up after inactivity.
