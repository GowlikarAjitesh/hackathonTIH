# train_sam_final_eta.py

import os
import time
from datetime import timedelta

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import tifffile as tiff

from segment_anything import sam_model_registry

# ================= PATHS ================= #
DATASET_DIR = "/home/vishnu/Ajitesh/ajitesh/dataset_after_train_test_split"
SAM_CHECKPOINT = "/home/vishnu/Ajitesh/ajitesh/sam_vit_b.pth"

# ================= CONFIG ================= #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 1024
EPOCHS = 10
BATCH_SIZE = 2

# ================= GPU INFO ================= #
print("\n🖥️ DEVICE INFO")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("Running on CPU")

# ================= DATASET ================= #
class SegDataset(Dataset):
    def __init__(self, root_dir, split):
        self.image_paths = []
        self.mask_paths = []

        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)

            if not os.path.isdir(folder_path):
                continue

            img_dir = os.path.join(folder_path, split, "images")
            mask_dir = os.path.join(folder_path, split, "masks")

            if not os.path.exists(img_dir):
                continue

            for img_name in os.listdir(img_dir):
                img_path = os.path.join(img_dir, img_name)
                mask_name = img_name.replace("img", "mask")
                mask_path = os.path.join(mask_dir, mask_name)

                if os.path.exists(mask_path):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)

        self.image_paths.sort()
        self.mask_paths.sort()

        print(f"✅ Loaded {len(self.image_paths)} samples for {split}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = tiff.imread(self.image_paths[idx])
        mask = tiff.imread(self.mask_paths[idx])

        if len(img.shape) == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] > 3:
            img = img[:, :, :3]

        img = img.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        img = torch.tensor(img).permute(2,0,1).unsqueeze(0)
        img = F.interpolate(img, size=(IMG_SIZE, IMG_SIZE))[0]

        mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
        mask = F.interpolate(mask, size=(IMG_SIZE, IMG_SIZE))[0]

        return img.float(), mask.float()

# ================= MODEL ================= #
class SAM_Model(nn.Module):
    def __init__(self):
        super().__init__()

        sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
        self.encoder = sam.image_encoder

        for name, p in self.encoder.named_parameters():
            if "blocks.10" in name or "blocks.11" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode='bilinear')
        return x

# ================= METRICS ================= #
def dice(pred, target):
    pred = torch.sigmoid(pred) > 0.5
    inter = (pred & target.bool()).float().sum()
    return (2 * inter) / (pred.float().sum() + target.sum() + 1e-6)

def iou(pred, target):
    pred = torch.sigmoid(pred) > 0.5
    inter = (pred & target.bool()).float().sum()
    union = pred.float().sum() + target.sum() - inter
    return inter / (union + 1e-6)

def accuracy(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    return (pred == target).float().mean()

# ================= TRAIN ================= #
def train():

    start_time = time.time()

    train_ds = SegDataset(DATASET_DIR, "train")
    val_ds   = SegDataset(DATASET_DIR, "val")
    test_ds  = SegDataset(DATASET_DIR, "test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = SAM_Model().to(DEVICE)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()

    total_batches = len(train_loader)

    for ep in range(EPOCHS):

        epoch_start = time.time()
        model.train()
        total_loss = 0

        for batch_idx, (x, y) in enumerate(train_loader):

            batch_start = time.time()

            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                pred = model(x)
                loss = loss_fn(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # ===== LIVE ETA ===== #
            elapsed = time.time() - start_time
            avg_batch_time = elapsed / ((ep * total_batches) + batch_idx + 1)

            remaining_batches = (EPOCHS * total_batches) - ((ep * total_batches) + batch_idx + 1)
            remaining_time = avg_batch_time * remaining_batches

            if batch_idx % 10 == 0:
                print(f"[Epoch {ep+1}/{EPOCHS}] Batch {batch_idx}/{total_batches} "
                      f"| Loss: {loss.item():.4f} "
                      f"| ⏳ Remaining: {str(timedelta(seconds=int(remaining_time)))}")

        # ===== VALIDATION ===== #
        model.eval()
        d, i, a = 0, 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)

                d += dice(pred, y).item()
                i += iou(pred, y).item()
                a += accuracy(pred, y).item()

        n = len(val_loader)

        print(f"\n🔥 Epoch {ep+1}/{EPOCHS}")
        print(f"📉 Loss: {total_loss/len(train_loader):.4f}")
        print(f"📊 Val → Dice:{d/n:.4f} IoU:{i/n:.4f} Acc:{a/n:.4f}")

    print("\n✅ TRAINING COMPLETE")

# ================= RUN ================= #
if __name__ == "__main__":
    train()