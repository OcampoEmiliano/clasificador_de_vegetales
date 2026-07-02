import io
import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from model import load_model, transform, DEVICE
from classes import CLASSES

app = FastAPI(title="Vegetables MLP API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(_BACKEND_DIR, "modelo_vegetales.pth"))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        predicted_idx = output.argmax(dim=1).item()
        confidence = probs[predicted_idx].item()

    return {
        "class": CLASSES[predicted_idx],
        "confidence": round(confidence, 4),
        "probabilities": {
            cls: round(probs[i].item(), 4)
            for i, cls in enumerate(CLASSES)
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
