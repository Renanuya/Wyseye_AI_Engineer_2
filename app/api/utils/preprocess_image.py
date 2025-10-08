import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from typing import Tuple


def preprocess_image(image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:

    # Orijinal boyutları kaydet
    original_height, original_width = image.shape[:2]

    # BGR'den RGB'ye çevir (PyTorch modelleri RGB bekler)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_tensor = to_tensor(rgb_image)

    return image_tensor, (original_height, original_width)


def preprocess_image_batch(images: list) -> Tuple[list, list]:

    preprocessed_tensors = []
    original_sizes = []

    for image in images:
        tensor, size = preprocess_image(image)
        preprocessed_tensors.append(tensor)
        original_sizes.append(size)

    return preprocessed_tensors, original_sizes


def resize_image_keeping_aspect_ratio(image: np.ndarray, max_size: int = 1333, min_size: int = 800) -> np.ndarray:
    height, width = image.shape[:2]

    # En küçük kenarı min_size yapmak için scale faktörünü hesapla
    min_original_size = float(min(height, width))
    max_original_size = float(max(height, width))

    scale = min_size / min_original_size

    # Max size kontrolü
    if max_original_size * scale > max_size:
        scale = max_size / max_original_size

    # Yeni boyutları hesapla
    new_height = int(height * scale)
    new_width = int(width * scale)

    # Yeniden boyutlandır
    resized = cv2.resize(image, (new_width, new_height))

    return resized


def preprocess_image_with_resize(image: np.ndarray,
                                 max_size: int = 1333,
                                 min_size: int = 800) -> Tuple[torch.Tensor, Tuple[int, int], float]:
    # Orijinal boyutları kaydet
    original_height, original_width = image.shape[:2]

    # Scale faktörünü hesapla
    min_original_size = float(min(original_height, original_width))
    max_original_size = float(max(original_height, original_width))

    scale = min_size / min_original_size
    if max_original_size * scale > max_size:
        scale = max_size / max_original_size

    # Görüntüyü yeniden boyutlandır
    resized = resize_image_keeping_aspect_ratio(image, max_size, min_size)

    # Tensor'a çevir
    tensor, _ = preprocess_image(resized)

    return tensor, (original_height, original_width), scale


def normalize_image(image_tensor: torch.Tensor,
                    mean: list = [0.485, 0.456, 0.406],
                    std: list = [0.229, 0.224, 0.225]) -> torch.Tensor:
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    normalized = (image_tensor - mean) / std
    return normalized
