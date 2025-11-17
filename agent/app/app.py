from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import numpy as np
from PIL import Image
import io
import os
import traceback


from .model_utils import load_model, pil_from_bytes, preprocess_pil_image, LABELS



USE_MTCNN = False # cambia a True si instalas mtcnn y opencv


MODEL_PATH = os.environ.get("MODEL_PATH", "./models/attitude_model.keras")
app = FastAPI(title="Attitude Classifier API")


# CORS (ajusta orígenes en producción)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo en arranque
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    model = None
    print(f"No se pudo cargar el modelo en {MODEL_PATH}:", e)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(image: UploadFile = File(...), detect_face: Optional[bool] = True):
    if image.content_type.split('/')[0] != "image":
        raise HTTPException(status_code=400, detail="El archivo subido no es una imagen.")

    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado en el servidor.")

    try:
        body = await image.read()
        pil = pil_from_bytes(body)

        # opcional: detect_face está presente en la interfaz, pero por defecto usamos la imagen completa
        if detect_face and USE_MTCNN:
            try:
                from mtcnn import MTCNN
                detector = MTCNN()
                pil = model_utils_detect_and_crop(pil, detector)
            except Exception:
                pass

        x = preprocess_pil_image(pil)
        preds = model.predict(x)[0]
        idx = int(np.argmax(preds))
        label = LABELS[idx] if idx < len(LABELS) else str(idx)
        confidence = float(preds[idx])
        probs = {LABELS[i]: float(preds[i]) for i in range(len(LABELS))}


        return JSONResponse({
        "label": label,
        "confidence": confidence,
        "probabilities": probs
        })


    except Exception as e:
       traceback.print_exc()
    raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")


# pequeña función local opcional para usar si activas MTCNN
def model_utils_detect_and_crop(pil_img, detector):
    import numpy as _np
    img = _np.asarray(pil_img.convert('RGB'))
    results = detector.detect_faces(img)
    if not results:
        return pil_img
    best = max(results, key=lambda r: r.get('confidence', 0))
    x, y, w, h = best['box']
    h_img, w_img = img.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)
    crop = img[y1:y2, x1:x2]
    return Image.fromarray(crop)