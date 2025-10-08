import io
import cv2
import numpy as np
import traceback
import time
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from typing import Optional

from app.api.utils.nms import apply_nms
from app.api.utils.preprocess_image import preprocess_image
from app.api.utils.postprocess_predictions import (
    postprocess_single_prediction,
    get_detection_summary
)
from app.api.utils.draw_detections import draw_detections
from app.api.utils.load_model import get_model, predict

router = APIRouter()


def process_uploaded_image(contents: bytes) -> np.ndarray:
    # Byte verisini PIL Image nesnesine dönüştür
    image_pil = Image.open(io.BytesIO(contents))

    # PIL Image'ı NumPy dizisine dönüştür
    image_np = np.array(image_pil)

    # Görüntü formatını BGR'ye dönüştür
    if len(image_np.shape) == 2:
        # Gri tonlamalı görüntü ise BGR'ye çevir
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    elif image_np.shape[2] == 4:
        # RGBA formatındaysa BGR'ye çevir
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
    elif image_np.shape[2] == 3:
        # RGB formatındaysa BGR'ye çevir
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    return image_np


@router.post("/detect")
async def detect_pawns(
    file: UploadFile = File(...),
    confidence_threshold: Optional[float] = Query(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    iou_threshold: Optional[float] = Query(0.5, ge=0.0, le=1.0, description="IoU threshold for NMS")
):
    # Model ve device bilgilerini al
    model, device, model_info = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model cannot be loaded.")

    try:
        # Dosya tipi kontrolü
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image.")

        # Inference zamanını ölç
        start_time = time.perf_counter()

        # Yüklenen dosyanın içeriğini oku
        contents = await file.read()

        # Görüntüyü işle
        image_np = process_uploaded_image(contents)

        # Preprocessing
        image_tensor, original_size = preprocess_image(image_np)

        # Model inference
        predictions = predict([image_tensor])  # List içinde gönder

        # Postprocessing - tek görüntü için ilk prediction'ı al
        raw_detections = postprocess_single_prediction(
            predictions[0], original_size, confidence_threshold
        )

        # NMS uygula
        detections = apply_nms(raw_detections, iou_threshold=iou_threshold)

        # İnference zamanını hesapla
        inference_time = time.perf_counter() - start_time

        # Özet bilgileri oluştur
        summary = get_detection_summary(detections)

        return JSONResponse(content={
            "success": True,
            "file_name": file.filename,
            "detections": detections,
            "summary": summary,
            "processing_info": {
                "confidence_threshold": confidence_threshold,
                "iou_threshold": iou_threshold,
                "inference_time_ms": round(inference_time * 1000, 2),
                "image_size": {
                    "width": original_size[1],
                    "height": original_size[0]
                }
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        print(f"\nDetection Error:")
        print(f"File: {file.filename}")
        print(f"Error: {str(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")

        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")


@router.post("/detect-visual")
async def detect_pawns_visual(
    file: UploadFile = File(...),
    confidence_threshold: Optional[float] = Query(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    iou_threshold: Optional[float] = Query(0.5, ge=0.0, le=1.0, description="IoU threshold for NMS"),
    image_quality: Optional[int] = Query(95, ge=50, le=100, description="JPEG quality (50-100)")
):
    # Model ve device bilgilerini al
    model, device, model_info = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model cannot be loaded.")

    try:
        # Dosya tipi kontrolü
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image.")

        # Yüklenen dosyanın içeriğini oku
        contents = await file.read()

        # Görüntüyü işle
        image_np = process_uploaded_image(contents)

        # Preprocessing
        image_tensor, original_size = preprocess_image(image_np)

        # Model inference
        predictions = predict([image_tensor])  # List içinde gönder

        # Postprocessing - tek görüntü için ilk prediction'ı al
        raw_detections = postprocess_single_prediction(
            predictions[0], original_size, confidence_threshold
        )

        # NMS uygula
        detections = apply_nms(raw_detections, iou_threshold=iou_threshold)

        # Tespitleri görüntü üzerine çiz
        result_image = draw_detections(image_np, detections)

        # Görüntüyü JPEG formatına encode et
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, image_quality]
        _, buffer = cv2.imencode('.jpg', result_image, encode_params)

        # Buffer'ı BytesIO nesnesine dönüştür
        io_buf = io.BytesIO(buffer.tobytes())

        # Görüntüyü HTTP yanıtı olarak döndür
        return StreamingResponse(
            io_buf,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"inline; filename=detected_{file.filename}",
                "X-Detections-Count": str(len(detections)),
                "X-White-Pawns": str(sum(1 for d in detections if d["class"] == "white-pawn")),
                "X-Black-Pawns": str(sum(1 for d in detections if d["class"] == "black-pawn"))
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"\nVisual Detection Error:")
        print(f"File: {file.filename}")
        print(f"Error: {str(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")

        raise HTTPException(status_code=500, detail=f"Visual detection error: {str(e)}")
