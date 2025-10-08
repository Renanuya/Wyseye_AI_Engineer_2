import torch
import numpy as np
from typing import List, Dict, Any, Tuple


def postprocess_predictions(predictions: List[Dict[str, torch.Tensor]],
                            original_sizes: List[Tuple[int, int]],
                            confidence_threshold: float = 0.5) -> List[List[Dict[str, Any]]]:
    all_detections = []

    for pred, (orig_h, orig_w) in zip(predictions, original_sizes):
        detections = []

        # Prediction'dan bilgileri çıkar
        boxes = pred['boxes'].cpu().numpy()  # [N, 4] (x1, y1, x2, y2)
        labels = pred['labels'].cpu().numpy()  # [N]
        scores = pred['scores'].cpu().numpy()  # [N]

        # Güven eşiğini uygula
        valid_mask = scores >= confidence_threshold
        boxes = boxes[valid_mask]
        labels = labels[valid_mask]
        scores = scores[valid_mask]

        # Her tespit için API formatında dict oluştur
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box

            # Koordinatları integer'a çevir ve sınırlar içinde tut
            x1 = max(0, min(int(x1), orig_w))
            y1 = max(0, min(int(y1), orig_h))
            x2 = max(0, min(int(x2), orig_w))
            y2 = max(0, min(int(y2), orig_h))

            # Label'ı class name'e çevir
            class_name = get_class_name(int(label))

            detection = {
                "class": class_name,
                "confidence": float(score),
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            }
            detections.append(detection)

        all_detections.append(detections)

    return all_detections


def postprocess_single_prediction(prediction: Dict[str, torch.Tensor],
                                  original_size: Tuple[int, int],
                                  confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:

    detections = []
    orig_h, orig_w = original_size

    # Prediction'dan bilgileri çıkar
    boxes = prediction['boxes'].cpu().numpy()  # [N, 4]
    labels = prediction['labels'].cpu().numpy()  # [N]
    scores = prediction['scores'].cpu().numpy()  # [N]

    # Güven eşiğini uygula
    valid_mask = scores >= confidence_threshold
    boxes = boxes[valid_mask]
    labels = labels[valid_mask]
    scores = scores[valid_mask]

    # Her tespit için API formatında dict oluştur
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box

        # Koordinatları integer'a çevir ve sınırlar içinde tut
        x1 = max(0, min(int(x1), orig_w))
        y1 = max(0, min(int(y1), orig_h))
        x2 = max(0, min(int(x2), orig_w))
        y2 = max(0, min(int(y2), orig_h))

        # Label'ı class name'e çevir
        class_name = get_class_name(int(label))

        detection = {
            "class": class_name,
            "confidence": float(score),
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            }
        }
        detections.append(detection)

    return detections


def get_detection_summary(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not detections:
        return {
            "total_detections": 0,
            "classes": {},
            "confidence_stats": {}
        }

    # Sınıf bazında sayıları hesapla
    class_counts = {}
    confidences = []

    for detection in detections:
        class_name = detection["class"]
        confidence = detection["confidence"]

        # Sınıf sayısını güncelle
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        confidences.append(confidence)

    # İstatistikleri hesapla
    confidence_stats = {}
    if confidences:
        confidence_stats = {
            "min": round(min(confidences), 3),
            "max": round(max(confidences), 3),
            "avg": round(sum(confidences) / len(confidences), 3)
        }

    return {
        "total_detections": len(detections),
        "classes": class_counts,
        "confidence_stats": confidence_stats
    }


def get_class_name(label_id: int) -> str:
    class_mapping = {
        0: "background",  # Background class (genellikle kullanılmaz)
        1: "black-pawn",
        2: "white-pawn"
    }

    return class_mapping.get(label_id, f"unknown-{label_id}")
