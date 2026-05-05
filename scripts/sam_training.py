import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import tifffile as tiff
from segment_anything import sam_model_registry

DATASET_DIR    = r"/home/cs24m119/hackathon/ajitesh/dataset_after_train_test_split/37458_fattu_bhila_ortho_3857(6)"
SAM_CHECKPOINT = "/home/cs24m119/hackathon/ajitesh/sam_vit_b.pth"

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE   = 1024
EPOCHS     = 20
BATCH_SIZE = 1
NUM_CLASSES = 5   # 0=bg, 1=buildings, 2=roads, 3=unused, 4=water


class SegDataset(Dataset):
    def __init__(self, root, split):
        self.img_dir  = os.path.join(root, split, "images")
        self.mask_dir = os.path.join(root, split, "masks")
        self.images   = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        img  = tiff.imread(os.path.join(self.img_dir, name))
        mask = tiff.imread(os.path.join(self.mask_dir, name.replace("img_", "mask_")))

        # Channel fix
        if len(img.shape) == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] > 3:
            img = img[:, :, :3]

        img = img.astype(np.float32) / 255.0

        # ✅ Keep multiclass labels (0,1,2,4) — do NOT binarize
        mask = mask.astype(np.int64)

        img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
        img = F.interpolate(img, size=(IMG_SIZE, IMG_SIZE))[0]

        mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float()
        mask = F.interpolate(mask, size=(IMG_SIZE, IMG_SIZE), mode='nearest')[0, 0].long()

        return img.float(), mask


class SAM_Model(nn.Module):
    def __init__(self):
        super().__init__()
        sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
        self.encoder = sam.image_encoder

        for p in self.encoder.parameters():
            p.requires_grad = False

        # ✅ Output NUM_CLASSES channels, not 1
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, NUM_CLASSES, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        return x  # (B, NUM_CLASSES, H, W)


def dice_per_class(pred, target, num_classes=NUM_CLASSES):
    pred   = torch.argmax(pred, dim=1)
    scores = []
    for c in range(1, num_classes):   # skip background
        p = (pred == c).float()
        t = (target == c).float()
        inter = (p * t).sum()
        scores.append((2*inter) / (p.sum() + t.sum() + 1e-6))
    return torch.stack(scores).mean()

def iou_per_class(pred, target, num_classes=NUM_CLASSES):
    pred   = torch.argmax(pred, dim=1)
    scores = []
    for c in range(1, num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        inter = (p * t).sum()
        union = p.sum() + t.sum() - inter
        scores.append(inter / (union + 1e-6))
    return torch.stack(scores).mean()

def accuracy(pred, target):
    pred = torch.argmax(pred, dim=1)
    return (pred == target).float().mean()


def train():
    train_ds = SegDataset(DATASET_DIR, "train")
    val_ds   = SegDataset(DATASET_DIR, "val")
    test_ds  = SegDataset(DATASET_DIR, "test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, num_workers=4)

    model = SAM_Model().to(DEVICE)

    # ✅ Class weights to handle imbalance within patches
    class_weights = torch.tensor([0.1, 2.0, 3.0, 1.0, 2.5]).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

    opt    = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    best_dice = 0.0

    for ep in range(EPOCHS):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()

            with torch.cuda.amp.autocast():
                pred = model(x)
                loss = loss_fn(pred, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()

        # Validation
        model.eval()
        d, i, a = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                d += dice_per_class(pred, y).item()
                i += iou_per_class(pred, y).item()
                a += accuracy(pred, y).item()

        n = len(val_loader)
        val_dice = d / n
        print(f"\nEpoch {ep+1}/{EPOCHS}")
        print(f"  Loss: {total_loss/len(train_loader):.4f}")
        print(f"  Val  → Dice:{val_dice:.4f}  IoU:{i/n:.4f}  Acc:{a/n:.4f}")

        # ✅ Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✅ Best model saved (dice={best_dice:.4f})")

    # Test
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    d, i, a = 0, 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            d += dice_per_class(pred, y).item()
            i += iou_per_class(pred, y).item()
            a += accuracy(pred, y).item()

    n = len(test_loader)
    print("\n🔥 FINAL TEST RESULTS")
    print(f"  Dice:{d/n:.4f}  IoU:{i/n:.4f}  Acc:{a/n:.4f}")

if __name__ == "__main__":
    train()