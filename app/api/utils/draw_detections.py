import cv2
import numpy as np
from typing import List, Dict, Any, Tuple


def draw_detections(image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:

    # Orijinal görüntüyü bozmamak için kopyasını oluştur
    img_copy = image.copy()

    # Her sınıf için renkleri tanımla (BGR formatında)
    colors = {
        "white-pawn": (255, 255, 255),  # Beyaz
        "black-pawn": (0, 0, 0),        # Siyah
        "background": (128, 128, 128)    # Gri
    }

    for det in detections:
        class_name = det["class"]
        confidence = det["confidence"]
        bbox = det["bbox"]

        # Bounding box'ın köşe koordinatları
        x1, y1 = bbox["x1"], bbox["y1"]
        x2, y2 = bbox["x2"], bbox["y2"]

        # Bu sınıf için rengi al, bulunamazsa varsayılan yeşil kullan
        color = colors.get(class_name, (0, 255, 0))

        # Bounding box'ı çiz
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 3)

        # Etiket başlığı (sınıf adı ve güven skoru)
        label = f"{class_name.replace('-', ' ').title()}: {confidence:.2f}"

        # Etiket başlığı boyutları
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        # Etiket arka plan kutusunu çiz
        cv2.rectangle(
            img_copy,
            (x1, y1 - text_height - 10),  # Arka plan kutusunun sol üst köşesi
            (x1 + text_width, y1),        # Arka plan kutusunun sağ alt köşesi
            color,                        # Bounding box ile aynı renk
            -1                            # -1 = Dolu kutu (içi dolu)
        )

        # Metin rengini belirle (siyah veya beyaz, arka plan rengine bağlı olarak)
        text_color = (0, 0, 0) if class_name == "white-pawn" else (255, 255, 255)

        cv2.putText(
            img_copy,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2
        )

    # Tespitlerin sayısını hesapla
    white_count = sum(1 for d in detections if d["class"] == "white-pawn")
    black_count = sum(1 for d in detections if d["class"] == "black-pawn")

    # İstatistik metinlerini hazırla
    stats_text = [
        f"Total: {len(detections)}",
        f"White Pawns: {white_count}",
        f"Black Pawns: {black_count}"
    ]

    # İstatistikleri görüntünün sol üst köşesine yaz
    y_offset = 30
    for text in stats_text:
        # Arka plan kutusu boyutunu hesapla
        (stat_width, stat_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
        )

        # Arka plan kutusunu çiz
        cv2.rectangle(
            img_copy,
            (5, y_offset - stat_height - 5),
            (15 + stat_width, y_offset + 5),
            (0, 0, 0),  # Siyah arka plan
            -1
        )

        # İstatistik metnini yaz
        cv2.putText(
            img_copy,
            text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),  # Yeşil metin
            2
        )
        y_offset += 35

    return img_copy


def draw_single_detection(image: np.ndarray,
                          detection: Dict[str, Any],
                          color: Tuple[int, int, int] = None) -> np.ndarray:
    img_copy = image.copy()

    class_name = detection["class"]
    confidence = detection["confidence"]
    bbox = detection["bbox"]

    x1, y1 = bbox["x1"], bbox["y1"]
    x2, y2 = bbox["x2"], bbox["y2"]

    if color is None:
        colors = {
            "white-pawn": (255, 255, 255),
            "black-pawn": (0, 0, 0),
        }
        color = colors.get(class_name, (0, 255, 0))

    # Bounding box çiz
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 3)

    # Label
    label = f"{class_name.replace('-', ' ').title()}: {confidence:.2f}"

    # Label arka planı ve metin
    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )

    cv2.rectangle(
        img_copy,
        (x1, y1 - text_height - 10),
        (x1 + text_width, y1),
        color,
        -1
    )

    text_color = (0, 0, 0) if class_name == "white-pawn" else (255, 255, 255)
    cv2.putText(
        img_copy,
        label,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        text_color,
        2
    )

    return img_copy


def draw_detection_grid(image: np.ndarray,
                        detections: List[Dict[str, Any]],
                        grid_size: int = 50,
                        grid_color: Tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:

    img_copy = image.copy()
    h, w = img_copy.shape[:2]

    for x in range(0, w, grid_size):
        cv2.line(img_copy, (x, 0), (x, h), grid_color, 1)

    for y in range(0, h, grid_size):
        cv2.line(img_copy, (0, y), (w, y), grid_color, 1)

    return draw_detections(img_copy, detections)


def highlight_detection_centers(image: np.ndarray,
                                detections: List[Dict[str, Any]],
                                radius: int = 5) -> np.ndarray:

    img_copy = draw_detections(image, detections)

    colors = {
        "white-pawn": (255, 255, 255),
        "black-pawn": (0, 0, 0),
    }

    for detection in detections:
        bbox = detection["bbox"]
        class_name = detection["class"]

        # Merkez koordinatını hesapla
        center_x = (bbox["x1"] + bbox["x2"]) // 2
        center_y = (bbox["y1"] + bbox["y2"]) // 2

        color = colors.get(class_name, (0, 255, 0))

        # Merkez dairesi
        cv2.circle(img_copy, (center_x, center_y), radius, color, -1)

        # Merkez çarpısı (kontrastlı renk)
        cross_color = (255, 255, 255) if class_name == "black-pawn" else (0, 0, 0)
        cv2.line(img_copy, (center_x-3, center_y), (center_x+3, center_y), cross_color, 2)
        cv2.line(img_copy, (center_x, center_y-3), (center_x, center_y+3), cross_color, 2)

    return img_copy


def create_detection_heatmap(image_shape: Tuple[int, int],
                             detections: List[Dict[str, Any]],
                             blur_radius: int = 50) -> np.ndarray:

    h, w = image_shape
    heatmap = np.zeros((h, w), dtype=np.float32)

    for detection in detections:
        bbox = detection["bbox"]
        confidence = detection["confidence"]

        # Merkez koordinatını hesapla
        center_x = (bbox["x1"] + bbox["x2"]) // 2
        center_y = (bbox["y1"] + bbox["y2"]) // 2

        # Güven skoruna göre ağırlık
        if 0 <= center_y < h and 0 <= center_x < w:
            heatmap[center_y, center_x] += confidence

    # Gaussian blur uygula
    if blur_radius > 0:
        heatmap = cv2.GaussianBlur(heatmap, (blur_radius*2+1, blur_radius*2+1), 0)

    # Normalize et
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Colormap uygula
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

    return heatmap_colored
