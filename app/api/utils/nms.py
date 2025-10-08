import numpy as np
from typing import List, Dict, Any


def calculate_iou(box1: Dict[str, int], box2: Dict[str, int]) -> float:
    # Kesişim alanının koordinatlarını hesapla
    x1 = max(box1["x1"], box2["x1"])
    y1 = max(box1["y1"], box2["y1"])
    x2 = min(box1["x2"], box2["x2"])
    y2 = min(box1["y2"], box2["y2"])

    # Kesişim alanı
    if x2 <= x1 or y2 <= y1:
        intersection = 0.0
    else:
        intersection = (x2 - x1) * (y2 - y1)

    # Her kutunun alanını hesapla
    area1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])

    # Birleşim alanı
    union = area1 + area2 - intersection

    # IoU hesapla
    # IoU = Kesişim / Birleşim
    if union <= 0:
        return 0.0

    return intersection / union


def apply_nms(detections: List[Dict[str, Any]],
              iou_threshold: float = 0.5,
              confidence_threshold: float = 0.0) -> List[Dict[str, Any]]:
    if not detections:
        return []

    # Önce güven skoruna göre filtrele
    filtered_detections = [det for det in detections
                           if det["confidence"] >= confidence_threshold]

    if not filtered_detections:
        return []

    # Sınıf bazında NMS uygula
    classes = set(det["class"] for det in filtered_detections)
    final_detections = []

    for class_name in classes:
        # Bu sınıfa ait tespitleri al
        class_detections = [det for det in filtered_detections
                            if det["class"] == class_name]

        # Güven skoruna göre azalan sırada sırala
        class_detections.sort(key=lambda x: x["confidence"], reverse=True)

        # NMS uygula
        class_nms_detections = apply_nms_single_class(class_detections, iou_threshold)
        final_detections.extend(class_nms_detections)

    # Güven skoruna göre tekrar sırala
    final_detections.sort(key=lambda x: x["confidence"], reverse=True)

    return final_detections


def apply_nms_single_class(detections: List[Dict[str, Any]],
                           iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
    if not detections:
        return []

    selected_detections = []
    remaining_detections = detections.copy()

    while remaining_detections:
        # En yüksek güven skoruna sahip tespiti seç
        best_detection = remaining_detections.pop(0)
        selected_detections.append(best_detection)

        # Bu tespitin box bilgileri
        best_box = best_detection["bbox"]

        # Kalan tespitlerden çakışanları çıkar
        filtered_remaining = []
        for det in remaining_detections:
            det_box = det["bbox"]
            iou = calculate_iou(best_box, det_box)

            # IoU eşiğinden düşükse koru
            if iou < iou_threshold:
                filtered_remaining.append(det)

        remaining_detections = filtered_remaining

    return selected_detections
