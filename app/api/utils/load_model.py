import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Optional
import os

# Model bilgilerini saklar
# Bu sayede model bir kez yüklenir ve tüm istekler tarafından kullanılır
model: Optional[torch.nn.Module] = None
device: torch.device = None
model_info = {
    "num_classes": 3,  # background + black-pawn + white-pawn
    "class_names": {0: "background", 1: "black-pawn", 2: "white-pawn"}
}


def get_model_architecture(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=None)  # No pretrained weights
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_model(model_path: str, device_name: str = "auto"):
    try:
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return None, None, False

        print(f"Loading model from: {model_path}")

        if device_name == "auto":
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_name)

        print(f"Using device: {device}")

        model = get_model_architecture(model_info["num_classes"])

        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()

        print(f"Model loaded successfully!")
        print(f"Device: {device}")
        print(f"Number of classes: {model_info['num_classes']}")
        print(f"Class names: {model_info['class_names']}")

        return model, device, True

    except FileNotFoundError:
        print(f"Model file cannot be found: {model_path}")
        return None, None, False
    except Exception as e:
        print(f"Error occurred while loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, False


def initialize_model():

    global model, device

    # Model dosyasının yolu - scripts klasöründeki model
    model_path = r"C:\Users\renan\OneDrive\Masaüstü\intern_ai_2\checkpoints\best.pth"

    print(f"Looking for model at: {model_path}")

    model, device, success = load_model(model_path)

    if not success:
        print("Model failed to load. API will not work properly.")
        raise RuntimeError("Model could not be loaded. Please check the model path and file.")
    else:
        print("Model initialization complete!")


def get_model():
    return model, device, model_info


def predict(image_tensor):
    global model, device

    if model is None:
        raise RuntimeError("Model not loaded. Call initialize_model() first.")

    if isinstance(image_tensor, list):
        image_tensor = [img.to(device) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)

    return predictions
