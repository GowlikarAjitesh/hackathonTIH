import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import tifffile as tiff
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_BASE = "/home/vishnu/Ajitesh/ajitesh/dataset_after_train_test_split"
CHECKPOINT_DIR = "/home/vishnu/Ajitesh/ajitesh/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 11          # 0=background + 10 classes
IMG_SIZE    = 512
BATCH_SIZE  = 4           # reduce to 2 if OOM
EPOCHS      = 50
LR          = 1e-4
NUM_WORKERS = 4

# All your village dataset folders
VILLAGE_FOLDERS = [
    "37458_fattu_bhila_ortho_3857(6)",
    "37774_bagga ortho_3857(7)_patches",
    "patch_dataNADALA_ORTHO(8)_patches",
    "PINDORI_MAYA_SINGH_TUGALWAL_28456_ortho(10)",
    "TIMMOWAL_37695_ORI(9)",
]

# Class names for logging
CLASS_NAMES = {
    0:  "background",
    1:  "buildings",
    2:  "roads",
    3:  "road_center",
    4:  "water",
    5:  "water_line",
    6:  "water_point",
    7:  "utility",
    8:  "utility_poly",
    9:  "railway",
    10: "bridge",
}

# ─────────────────────────────────────────────
# CLASS WEIGHTS (handle imbalance)
# background is very common → low weight
# rare classes → high weight
# ─────────────────────────────────────────────
CLASS_WEIGHTS = torch.tensor([
    0.05,  # 0  background
    2.0,   # 1  buildings
    3.0,   # 2  roads
    3.0,   # 3  road_center
    2.5,   # 4  water
    3.5,   # 5  water_line
    3.5,   # 6  water_point
    3.5,   # 7  utility
    3.0,   # 8  utility_poly
    4.0,   # 9  railway
    4.0,   # 10 bridge
], dtype=torch.float32)


# ─────────────────────────────────────────────
# AUGMENTATIONS
# ─────────────────────────────────────────────
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.4),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class PatchDataset(Dataset):
    def __init__(self, village_folder, split, transforms=None):
        self.img_dir  = os.path.join(village_folder, split, "images")
        self.mask_dir = os.path.join(village_folder, split, "masks")
        self.transforms = transforms

        if not os.path.exists(self.img_dir):
            self.images = []
            return

        self.images = [
            f for f in os.listdir(self.img_dir)
            if f.endswith(".tif")
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name  = self.images[idx]
        mask_name = img_name.replace("img_", "mask_")

        img  = tiff.imread(os.path.join(self.img_dir,  img_name))
        mask = tiff.imread(os.path.join(self.mask_dir, mask_name))

        # ── Fix image shape → HWC, 3 channels ──
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.ndim == 3 and img.shape[0] in [3, 4]:
            img = np.transpose(img, (1, 2, 0))
        img = img[:, :, :3]

        # ── Normalize to uint8 if needed ──
        if img.dtype != np.uint8:
            img = img.astype(np.float32)
            p2, p98 = np.percentile(img, 2), np.percentile(img, 98)
            img = np.clip((img - p2) / (p98 - p2 + 1e-6), 0, 1)
            img = (img * 255).astype(np.uint8)

        # ── Resize ──
        img  = self._resize_img(img,  IMG_SIZE, IMG_SIZE)
        mask = self._resize_mask(mask, IMG_SIZE, IMG_SIZE)

        # ── Clamp mask to valid class range ──
        mask = np.clip(mask, 0, NUM_CLASSES - 1).astype(np.int64)

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask.astype(np.uint8))
            img  = augmented["image"]           # tensor (3, H, W)
            mask = augmented["mask"].long()     # tensor (H, W)
        else:
            img  = torch.tensor(img).permute(2, 0, 1).float() / 255.0
            mask = torch.tensor(mask).long()

        return img, mask

    def _resize_img(self, img, h, w):
        import cv2
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    def _resize_mask(self, mask, h, w):
        import cv2
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def build_dataloaders():
    train_sets, val_sets, test_sets = [], [], []

    for folder_name in VILLAGE_FOLDERS:
        folder = os.path.join(DATASET_BASE, folder_name)
        if not os.path.exists(folder):
            print(f"⚠️  Folder not found: {folder_name}")
            continue

        train_ds = PatchDataset(folder, "train", train_transforms)
        val_ds   = PatchDataset(folder, "val",   val_transforms)
        test_ds  = PatchDataset(folder, "test",  val_transforms)

        print(f"  {folder_name[:35]:35s} → train:{len(train_ds):4d}  val:{len(val_ds):4d}  test:{len(test_ds):4d}")

        if len(train_ds) > 0: train_sets.append(train_ds)
        if len(val_ds)   > 0: val_sets.append(val_ds)
        if len(test_ds)  > 0: test_sets.append(test_ds)

    train_loader = DataLoader(
        ConcatDataset(train_sets),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        ConcatDataset(val_sets),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        ConcatDataset(test_sets),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    total_train = sum(len(d) for d in train_sets)
    total_val   = sum(len(d) for d in val_sets)
    total_test  = sum(len(d) for d in test_sets)
    print(f"\n  Total → train:{total_train}  val:{total_val}  test:{total_test}")

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def build_model():
    model = deeplabv3_resnet101(weights="DEFAULT")

    # Replace classifier head for our number of classes
    model.classifier    = DeepLabHead(2048, NUM_CLASSES)
    model.aux_classifier = None  # disable aux loss to save VRAM

    return model.to(DEVICE)


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
class Metrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.long
        )

    def update(self, pred, target):
        pred   = torch.argmax(pred, dim=1).cpu().view(-1)
        target = target.cpu().view(-1)
        mask   = (target >= 0) & (target < self.num_classes)
        idx    = self.num_classes * target[mask] + pred[mask]
        self.confusion += torch.bincount(
            idx, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def iou_per_class(self):
        iou = []
        for c in range(self.num_classes):
            tp = self.confusion[c, c].item()
            fp = self.confusion[:, c].sum().item() - tp
            fn = self.confusion[c, :].sum().item() - tp
            denom = tp + fp + fn
            iou.append(tp / denom if denom > 0 else float("nan"))
        return iou

    def mean_iou(self):
        iou = self.iou_per_class()
        # Only average over classes that actually appear
        valid = [v for v in iou if not np.isnan(v)]
        return np.mean(valid) if valid else 0.0

    def pixel_accuracy(self):
        correct = self.confusion.diagonal().sum().item()
        total   = self.confusion.sum().item()
        return correct / total if total > 0 else 0.0

    def dice_per_class(self):
        dice = []
        for c in range(self.num_classes):
            tp = self.confusion[c, c].item()
            fp = self.confusion[:, c].sum().item() - tp
            fn = self.confusion[c, :].sum().item() - tp
            denom = 2 * tp + fp + fn
            dice.append(2 * tp / denom if denom > 0 else float("nan"))
        return dice

    def mean_dice(self):
        dice  = self.dice_per_class()
        valid = [v for v in dice[1:] if not np.isnan(v)]  # skip background
        return np.mean(valid) if valid else 0.0


# ─────────────────────────────────────────────
# COMBINED LOSS: CrossEntropy + Dice
# ─────────────────────────────────────────────
class CombinedLoss(nn.Module):
    def __init__(self, weights, ce_weight=0.6, dice_weight=0.4):
        super().__init__()
        self.ce        = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
        self.ce_w      = ce_weight
        self.dice_w    = dice_weight
        self.n_classes = len(weights)

    def dice_loss(self, pred, target):
        pred   = F.softmax(pred, dim=1)
        target_oh = F.one_hot(target, self.n_classes).permute(0, 3, 1, 2).float()
        dims   = (0, 2, 3)
        inter  = (pred * target_oh).sum(dim=dims)
        denom  = pred.sum(dim=dims) + target_oh.sum(dim=dims)
        dice   = 1 - (2 * inter + 1e-6) / (denom + 1e-6)
        return dice[1:].mean()   # skip background

    def forward(self, pred, target):
        return (self.ce_w   * self.ce(pred, target) +
                self.dice_w * self.dice_loss(pred, target))


# ─────────────────────────────────────────────
# TRAIN ONE EPOCH
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, loss_fn, scaler):
    model.train()
    total_loss = 0.0
    metrics    = Metrics(NUM_CLASSES)

    for imgs, masks in tqdm(loader, desc="  Train", leave=False):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out  = model(imgs)["out"]
            loss = loss_fn(out, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        metrics.update(out, masks)

    return total_loss / len(loader), metrics


# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    metrics    = Metrics(NUM_CLASSES)

    for imgs, masks in tqdm(loader, desc="  Eval ", leave=False):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        with torch.cuda.amp.autocast():
            out  = model(imgs)["out"]
            loss = loss_fn(out, masks)

        total_loss += loss.item()
        metrics.update(out, masks)

    return total_loss / len(loader), metrics


# ─────────────────────────────────────────────
# PRINT PER CLASS METRICS
# ─────────────────────────────────────────────
def print_class_metrics(metrics, prefix=""):
    iou_list  = metrics.iou_per_class()
    dice_list = metrics.dice_per_class()
    print(f"\n  {prefix} Per-class metrics:")
    print(f"  {'Class':15s} {'IoU':>8s} {'Dice':>8s}")
    print(f"  {'-'*35}")
    for c in range(NUM_CLASSES):
        iou  = f"{iou_list[c]:.4f}"  if not np.isnan(iou_list[c])  else "  n/a  "
        dice = f"{dice_list[c]:.4f}" if not np.isnan(dice_list[c]) else "  n/a  "
        print(f"  {CLASS_NAMES[c]:15s} {iou:>8s} {dice:>8s}")
    print(f"  {'-'*35}")
    print(f"  {'mIoU':15s} {metrics.mean_iou():>8.4f}")
    print(f"  {'mDice':15s} {metrics.mean_dice():>8.4f}")
    print(f"  {'Pixel Acc':15s} {metrics.pixel_accuracy():>8.4f}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print(f"Device : {DEVICE}")
    print(f"Classes: {NUM_CLASSES}")
    print(f"Epochs : {EPOCHS}\n")

    # Dataloaders
    print("Building dataloaders...")
    train_loader, val_loader, test_loader = build_dataloaders()

    # Model
    print("\nBuilding DeepLabV3+ (ResNet-101 backbone)...")
    model = build_model()
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total parameters: {total_params:.1f}M")

    # Loss, optimizer, scheduler
    loss_fn   = CombinedLoss(CLASS_WEIGHTS)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = torch.cuda.amp.GradScaler()

    best_miou     = 0.0
    best_ckpt     = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    history       = {"train_loss": [], "val_loss": [], "val_miou": [], "val_dice": []}

    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}  lr={scheduler.get_last_lr()[0]:.6f}")

        train_loss, train_m = train_epoch(model, train_loader, optimizer, loss_fn, scaler)
        val_loss,   val_m   = evaluate(model, val_loader, loss_fn)

        scheduler.step()

        val_miou = val_m.mean_iou()
        val_dice = val_m.mean_dice()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_miou)
        history["val_dice"].append(val_dice)

        print(f"  Train loss : {train_loss:.4f}")
        print(f"  Val   loss : {val_loss:.4f}")
        print(f"  Val   mIoU : {val_miou:.4f}")
        print(f"  Val   mDice: {val_dice:.4f}")

        # Print per-class every 5 epochs
        if epoch % 5 == 0:
            print_class_metrics(val_m, prefix="Val")

        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "best_miou":  best_miou,
                "num_classes": NUM_CLASSES,
            }, best_ckpt)
            print(f"  ✅ Best model saved  (mIoU={best_miou:.4f})")

        # Save latest checkpoint every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}.pth")
            torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt_path)
            print(f"  💾 Checkpoint saved: epoch_{epoch:03d}.pth")

    # ── FINAL TEST ──
    print("\n" + "="*50)
    print("Loading best model for final test evaluation...")
    checkpoint = torch.load(best_ckpt)
    model.load_state_dict(checkpoint["model"])

    test_loss, test_m = evaluate(model, test_loader, loss_fn)
    print(f"\nFINAL TEST RESULTS (best model from epoch {checkpoint['epoch']})")
    print(f"  Test loss : {test_loss:.4f}")
    print_class_metrics(test_m, prefix="Test")

    # Save training history
    import json
    history_path = os.path.join(CHECKPOINT_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved: {history_path}")
    print(f"Best checkpoint saved : {best_ckpt}")


if __name__ == "__main__":
    main()