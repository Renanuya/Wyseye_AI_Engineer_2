import os
import json
import time
import argparse
from pathlib import Path
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


class EarlyStopping:

    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_map, model):
        if self.best_score is None:
            self.best_score = val_map
            self.save_checkpoint(model)
        elif val_map > self.best_score + self.min_delta:
            self.best_score = val_map
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    print(f"Early stopping triggered. Restored best weights (mAP: {self.best_score:.4f})")

    def save_checkpoint(self, model):
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()


class TrainingHistory:

    def __init__(self):
        self.epochs = []
        self.train_losses = []
        self.val_maps = []
        self.val_ap50s = []
        self.learning_rates = []

    def update(self, epoch, train_loss, val_map, val_ap50, lr):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_maps.append(val_map)
        self.val_ap50s.append(val_ap50)
        self.learning_rates.append(lr)

    def plot_metrics(self, save_path=None):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curve
        ax1.plot(self.epochs, self.train_losses, 'b-', label='Train Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # mAP curve
        ax2.plot(self.epochs, self.val_maps, 'r-', label='Validation mAP')
        ax2.set_title('Validation mAP')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mAP')
        ax2.legend()
        ax2.grid(True)

        # AP50 curve
        ax3.plot(self.epochs, self.val_ap50s, 'g-', label='Validation AP50')
        ax3.set_title('Validation AP50')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('AP50')
        ax3.legend()
        ax3.grid(True)

        # Learning rate
        ax4.plot(self.epochs, self.learning_rates, 'm-', label='Learning Rate')
        ax4.set_title('Learning Rate')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('LR')
        ax4.legend()
        ax4.grid(True)
        ax4.set_yscale('log')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class COCODetectionDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_file):
        self.img_dir = Path(img_dir)
        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()

        # Sadece black-pawn (id:5) ve white-pawn (id:11) kategorilerini kullan
        target_cat_ids = [5, 11]  # black-pawn ve white-pawn
        self.cat_ids_sorted = target_cat_ids

        # black-pawn: 1, white-pawn: 2
        self.catid2label = {5: 1, 11: 2}
        self.label2catid = {1: 5, 2: 11}

        filtered = []
        for img_id in self.img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=False)
            if len(ann_ids) > 0:
                filtered.append(img_id)
        if len(filtered) > 0:
            self.img_ids = filtered

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_dir / img_info["file_name"]
        img = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in anns:
            # Sadece black-pawn (5) ve white-pawn (11) kategorilerini kabul et
            if ann["category_id"] not in [5, 11]:
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])  # xyxy
            labels.append(self.catid2label[ann["category_id"]])
            areas.append(ann.get("area", w * h))
            iscrowd.append(0)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64),
        }

        img = to_tensor(img)
        return img, target


def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    return list(imgs), list(targets)


def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights="COCO_V1")  # backbone + FPN pretrained
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def coco_evaluate(model, data_loader, device, label2catid, max_images=None, measure_latency=False):
    model.eval()
    coco_gt = data_loader.dataset.coco
    results = []
    n_imgs = 0

    if measure_latency:
        warmup = min(20, len(data_loader))
        with torch.no_grad():
            for i, (imgs, targets) in enumerate(data_loader):
                imgs = [img.to(device) for img in imgs]
                _ = model(imgs)
                if i + 1 >= warmup:
                    break

    # Zaman ölçümü
    total_time = 0.0
    total_count = 0

    with torch.no_grad():
        for imgs, targets in tqdm(data_loader, desc="Eval"):
            imgs = [img.to(device) for img in imgs]

            if measure_latency:
                torch.cuda.synchronize(device) if device.type == "cuda" else None
                t0 = time.perf_counter()

            outputs = model(imgs)

            if measure_latency:
                torch.cuda.synchronize(device) if device.type == "cuda" else None
                total_time += (time.perf_counter() - t0)
                total_count += len(imgs)

            for img, target, output in zip(imgs, targets, outputs):
                image_id = int(target["image_id"].item())
                boxes = output["boxes"].detach().cpu().tolist()
                scores = output["scores"].detach().cpu().tolist()
                labels = output["labels"].detach().cpu().tolist()

                # COCO formatına çevir (xyxy -> xywh)
                for b, s, l in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = b
                    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
                    cat_id = label2catid.get(int(l), None)
                    if cat_id is None:
                        continue
                    results.append({
                        "image_id": image_id,
                        "category_id": int(cat_id),
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(s),
                    })
            n_imgs += len(imgs)
            if max_images and n_imgs >= max_images:
                break

    if len(results) == 0:
        print("No detections to evaluate.")
        return {"mAP": 0.0, "AP50": 0.0, "latency_ms_per_image": None}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    AP = float(coco_eval.stats[0])
    AP50 = float(coco_eval.stats[1])

    latency_ms = None
    if measure_latency and total_count > 0:
        latency_ms = (total_time / total_count) * 1000.0

    return {"mAP": AP, "AP50": AP50, "latency_ms_per_image": latency_ms}


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None, log_interval=50):
    model.train()
    lr = optimizer.param_groups[0]["lr"]
    running_loss = 0.0
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(data_loader, desc=f"Train epoch {epoch}")
    for i, (imgs, targets) in enumerate(progress_bar):
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_value = loss.item()
        running_loss += loss_value
        total_loss += loss_value
        num_batches += 1

        if (i + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{lr:.6f}'
            })
            running_loss = 0.0

    epoch_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return epoch_loss


def save_checkpoint(model, path):
    state = model.state_dict()
    torch.save(state, path)


def model_size_mb(path):
    size_bytes = os.path.getsize(path)
    return size_bytes / (1024 * 1024)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True, help="Dataset root (içinde train/valid/test var)")
    p.add_argument("--train-split", type=str, default="train")
    p.add_argument("--val-split", type=str, default="valid")
    p.add_argument("--epochs", type=int, default=50, help="Maximum number of epochs")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--checkpoint", type=str, default="fasterrcnn_best.pth")

    # Early stopping parameters
    p.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    p.add_argument("--min-delta", type=float, default=0.001, help="Minimum change in mAP to qualify as improvement")

    # Advanced training parameters
    p.add_argument("--reduce-lr-patience", type=int, default=3, help="ReduceLROnPlateau patience")
    p.add_argument("--reduce-lr-factor", type=float, default=0.5, help="ReduceLROnPlateau factor")
    p.add_argument("--plot-history", action="store_true", help="Plot training history at the end")

    return p.parse_args()


def make_loader(root, split, batch_size, num_workers, shuffle):
    img_dir = Path(root) / split
    ann_file = img_dir / "_annotations.coco.json"
    ds = COCODetectionDataset(img_dir, ann_file)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                    num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    return ds, dl


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds, train_dl = make_loader(args.data_root, args.train_split, args.batch_size, args.num_workers, shuffle=True)
    val_ds, val_dl = make_loader(args.data_root, args.val_split, 1, args.num_workers, shuffle=False)

    num_classes = 3  # background + black-pawn + white-pawn
    print("Classes:", num_classes - 1, "Target categories: black-pawn (5), white-pawn (11)")

    model = get_model(num_classes).to(device)

    if args.eval_only:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(args.checkpoint)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        metrics = coco_evaluate(model, val_dl, device, train_ds.label2catid, measure_latency=True)
        print("Eval-only:", metrics)
        return

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)

    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=args.reduce_lr_factor,
        patience=args.reduce_lr_patience, verbose=True, min_lr=1e-7
    )

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # Early stopping ve training history
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    history = TrainingHistory()

    print(f"Starting training with early stopping (patience={args.patience})")
    print(f"Maximum epochs: {args.epochs}")
    print("-" * 60)

    best_map = -1.0
    for epoch in range(1, args.epochs + 1):
        # Training
        epoch_loss = train_one_epoch(model, optimizer, train_dl, device, epoch, scaler=scaler)

        # Validation
        metrics = coco_evaluate(model, val_dl, device, train_ds.label2catid, measure_latency=True)
        current_lr = optimizer.param_groups[0]['lr']

        # Update learning rate scheduler
        lr_sched.step(metrics["mAP"])

        # Update history
        history.update(epoch, epoch_loss, metrics["mAP"], metrics["AP50"], current_lr)

        # Print epoch results
        print(f"[epoch {epoch:3d}] Loss: {epoch_loss:.4f} | mAP: {metrics['mAP']:.4f} | AP50: {metrics['AP50']:.4f} | LR: {current_lr:.6f} | Latency: {metrics['latency_ms_per_image']:.1f}ms")

        # Save best model
        if metrics["mAP"] > best_map:
            save_checkpoint(model, args.checkpoint)
            best_map = metrics["mAP"]
            print(f"  ✓ New best mAP! Saved to {args.checkpoint}")

        # Early stopping check
        early_stopping(metrics["mAP"], model)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"Best mAP achieved: {early_stopping.best_score:.4f}")
            break

    # Final results
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Best mAP: {early_stopping.best_score:.4f}")
    print(f"Total epochs: {epoch}")

    # Model size
    if os.path.exists(args.checkpoint):
        mb = model_size_mb(args.checkpoint)
        print(f"Model size: {mb:.2f} MB")

    # Plot training history
    if args.plot_history:
        history.plot_metrics("training_history.png")

    # Save training history
    history_file = args.checkpoint.replace('.pth', '_history.json')
    with open(history_file, 'w') as f:
        json.dump({
            'epochs': history.epochs,
            'train_losses': history.train_losses,
            'val_maps': history.val_maps,
            'val_ap50s': history.val_ap50s,
            'learning_rates': history.learning_rates,
            'best_map': float(early_stopping.best_score),
            'final_epoch': epoch
        }, f, indent=2)
    print(f"Training history saved to {history_file}")


if __name__ == "__main__":
    main()
