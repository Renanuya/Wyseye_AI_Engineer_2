import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_convert
from torchvision.transforms import functional as TF

from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")


# ---- Dataset ----
class CocoPawnDataset(torch.utils.data.Dataset):
    """
    COCO formatında sadece siyah/beyaz piyonları döndüren dataset.
    Diğer kategorileri tamamen yok sayar.
    Boş görüntülerde (hedef yoksa) boş tensör döner (torchvision için geçerli).
    """

    def __init__(self, img_dir: Path, ann_path: Path, keep_id_to_label={5: 1, 11: 2}, is_train=True):
        self.img_dir = Path(img_dir)
        self.coco = COCO(str(ann_path))
        self.img_ids = list(self.coco.imgs.keys())
        self.keep_id_to_label = keep_id_to_label  # {5:1, 11:2}
        self.is_train = is_train

        # Sadece bu kategorilerdeki anotasyonları çekmek için id listesi:
        self.keep_cat_ids = list(self.keep_id_to_label.keys())

        # Hız için: img_id -> ann listesi
        self.img_to_anns = defaultdict(list)
        for ann in self.coco.anns.values():
            if ann.get("iscrowd", 0) == 0 and ann["category_id"] in self.keep_cat_ids:
                self.img_to_anns[ann["image_id"]].append(ann)

        # Görüntü yollarını hazırla
        self.id_to_path = {}
        for img_id, meta in self.coco.imgs.items():
            file_name = meta["file_name"]
            self.id_to_path[img_id] = self.img_dir / file_name

    def __len__(self):
        return len(self.img_ids)

    def _hflip(self, image, target):
        w, _ = image.size
        image = TF.hflip(image)
        boxes = target["boxes"].clone()
        # x1,y1,x2,y2 -> yansıma: x -> w - x
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        target["boxes"] = boxes
        return image, target

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        path = self.id_to_path[img_id]
        image = Image.open(path).convert("RGB")

        anns = self.img_to_anns.get(img_id, [])
        boxes_xywh = []
        labels = []

        # COCO bbox: [x, y, w, h] -> XYXY'ye çevireceğiz
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes_xywh.append([x, y, w, h])
            labels.append(self.keep_id_to_label[ann["category_id"]])

        if len(boxes_xywh) > 0:
            boxes_xywh = torch.tensor(boxes_xywh, dtype=torch.float32)
            boxes_xyxy = box_convert(boxes_xywh, in_fmt="xywh", out_fmt="xyxy")
            # Görüntü boyutuna clamp
            W, H = image.size
            boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clamp(0, W)
            boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clamp(0, H)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_xyxy = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_xyxy,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
        }

        # Basit augment (sadece train için yatay çevirme %50)
        if self.is_train and np.random.rand() < 0.5:
            image, target = self._hflip(image, target)

        image = TF.to_tensor(image)  # [0,1] float32, CxHxW

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ---- Model ----
def get_model(num_classes: int):
    """
    num_classes: background dahil toplam sınıf sayısı.
    Bizde: background(0) + black-pawn(1) + white-pawn(2) = 3
    """
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ---- Train/Eval helpers ----
@torch.no_grad()
def evaluate_val_loss(model, data_loader, device):
    model.train(False)
    total = 0.0
    n = 0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # Faster R-CNN eval'de forward(loss) çalıştırmak için train modunda gerekir,
        # ama grad gerekmediği için context'i no_grad tutuyoruz ve geçici train(True) açıyoruz.
        model.train(True)
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values()).item()
        model.train(False)
        total += loss
        n += 1
    return total / max(n, 1)


def train(args):
    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "valid"
    test_dir = data_root / "test"

    ann_train = train_dir / "_annotations.coco.json"
    ann_val = val_dir / "_annotations.coco.json"
    ann_test = test_dir / "_annotations.coco.json"  # eval-only için kullanılabilir

    keep = {5: 1, 11: 2}  # COCO id -> label id (1..num_classes-1)

    # Datasets
    ds_train = CocoPawnDataset(train_dir, ann_train, keep_id_to_label=keep, is_train=True)
    ds_val = CocoPawnDataset(val_dir,   ann_val,   keep_id_to_label=keep, is_train=False)
    # test opsiyonel, eval-only'de val kullanıyoruz, istersen benzerce ekleyebilirsin.

    # Loaders
    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = 3  # background + 2 piyon
    model = get_model(num_classes).to(device)

    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt)

    if args.eval_only:
        val_loss = evaluate_val_loss(model, val_loader, device)
        print(f"[EVAL-ONLY] Validation loss: {val_loss:.4f}")
        return

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args.reduce_lr_patience, factor=args.reduce_lr_factor
    )

    best_val = float("inf")
    patience = args.patience
    wait = 0

    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, args.epochs + 1):
        model.train(True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        running = 0.0
        steps = 0

        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += loss.item()
            steps += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running / max(steps, 1)
        val_loss = evaluate_val_loss(model, val_loader, device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={current_lr:.2e}")

        # Checkpoint
        out_dir = Path("checkpoints")
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / "last.pth")

        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), out_dir / "best.pth")
            print(f"  → New best (val_loss {best_val:.4f}). Saved checkpoints/best.pth")
        else:
            wait += 1
            print(f"  → No improvement ({wait}/{patience})")

        if wait >= patience:
            print("Early stopping triggered.")
            break

    if args.plot_history:
        try:
            epochs = range(1, len(history["train_loss"]) + 1)
            plt.figure()
            plt.plot(epochs, history["train_loss"], label="train_loss")
            plt.plot(epochs, history["val_loss"], label="val_loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            Path("plots").mkdir(exist_ok=True, parents=True)
            plt.savefig("plots/training_curves.png", dpi=150, bbox_inches="tight")
            print("Saved plots/training_curves.png")
        except Exception as e:
            print(f"Plot error: {e}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True,
                    help="COCO dizin kökü (train/valid/test alt klasörleriyle).")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--eval-only", action="store_true",
                    help="Yalnızca valid set üzerinde loss hesapla (eğitim yok).")
    ap.add_argument("--checkpoint", type=str, default="",
                    help="Eğitime kaldığın yerden devam etmek için .pth dosyası yolu.")
    ap.add_argument("--patience", type=int, default=7,
                    help="Erken durdurma sabrı (iyileşme olmazsa).")
    ap.add_argument("--min-delta", type=float, default=1e-3,
                    help="İyileşme eşiği (val loss için).")
    ap.add_argument("--reduce-lr-patience", type=int, default=3)
    ap.add_argument("--reduce-lr-factor", type=float, default=0.5)
    ap.add_argument("--plot-history", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
