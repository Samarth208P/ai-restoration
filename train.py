import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "./Restormer")
try:
    from basicsr.models.archs.restormer_arch import Restormer
except ImportError:
    print("Error: Could not import Restormer. Please make sure you have run:")
    print("git clone https://github.com/swz30/Restormer.git")
    print("cd Restormer && python setup.py develop --no_cuda_ext")
    sys.exit(1)

class RestorationDataset(Dataset):
    def __init__(self, noisy_dir, gt_dir, files, augment=False):
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir
        self.files = files
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        noisy = np.load(os.path.join(self.noisy_dir, filename)).astype(np.float32)
        gt = np.load(os.path.join(self.gt_dir, filename)).astype(np.float32)

        noisy = torch.from_numpy(noisy).unsqueeze(0)
        gt = torch.from_numpy(gt).unsqueeze(0)

        noisy = torch.clamp(noisy, 0.0, 1.0)
        gt = torch.clamp(gt, 0.0, 1.0)

        # 128x128 → 256x256
        noisy = F.interpolate(noisy.unsqueeze(0), size=(256, 256), mode="bicubic", align_corners=False).squeeze(0)

        if self.augment:
            if random.random() < 0.5:
                noisy = torch.flip(noisy, dims=[2])
                gt = torch.flip(gt, dims=[2])
            if random.random() < 0.5:
                noisy = torch.flip(noisy, dims=[1])
                gt = torch.flip(gt, dims=[1])

        return noisy, gt

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff ** 2 + self.eps ** 2)
        return loss.mean()

def ssim_loss(pred, target):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(target, 3, 1, 1)

    sigma_x = (F.avg_pool2d(pred ** 2, 3, 1, 1) - mu_x ** 2)
    sigma_y = (F.avg_pool2d(target ** 2, 3, 1, 1) - mu_y ** 2)
    sigma_xy = (F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y)

    numerator = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2))
    denominator = ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

    ssim = numerator / (denominator + 1e-8)
    return 1 - ssim.mean()

def main():
    parser = argparse.ArgumentParser(description="Train Image Restoration Model")
    parser.add_argument("--noisy_dir", type=str, required=True, help="Directory containing noisy test images (.npy)")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing ground truth images (.npy)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset splitting
    gt_files = sorted([f for f in os.listdir(args.gt_dir) if f.endswith(".npy") and not f.startswith("._")])
    noisy_files = sorted([f for f in os.listdir(args.noisy_dir) if f.endswith(".npy") and not f.startswith("._")])
    common_files = sorted(set(gt_files).intersection(noisy_files))
    
    assert len(common_files) > 0, "No common files found between GT and Noisy directories"

    random.seed(42)
    files = common_files.copy()
    random.shuffle(files)
    split = int(0.9 * len(files))
    train_files = files[:split]
    val_files = files[split:]

    train_dataset = RestorationDataset(args.noisy_dir, args.gt_dir, train_files, augment=True)
    val_dataset = RestorationDataset(args.noisy_dir, args.gt_dir, val_files, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    model = Restormer(
        inp_channels=1, out_channels=1, dim=16, num_blocks=[1, 1, 2, 2],
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.0, bias=False,
        LayerNorm_type="WithBias", dual_pixel_task=False
    ).to(device)

    criterion = CharbonnierLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (noisy, gt) in enumerate(train_loader):
            noisy, gt = noisy.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                restored = model(noisy)
                restored = torch.clamp(restored, 0.0, 1.0)
                loss_pixel = criterion(restored, gt)
                loss_ssim = ssim_loss(restored, gt)
                loss = loss_pixel + 0.1 * loss_ssim

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{args.epochs} | Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.5f}")

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, gt in val_loader:
                noisy, gt = noisy.to(device), gt.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    restored = model(noisy)
                    restored = torch.clamp(restored, 0.0, 1.0)
                    loss_pixel = criterion(restored, gt)
                    loss_ssim = ssim_loss(restored, gt)
                    loss = loss_pixel + 0.1 * loss_ssim
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step()

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.8f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_restormer.pth")
            print("✅ Best model saved!")
        print("=" * 60)

if __name__ == "__main__":
    main()
