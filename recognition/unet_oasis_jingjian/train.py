# train.py
"""
Training script for UNetSmall.
Example:
 python train.py --data-root /home/groups/comp3710/OASIS --epochs 5 --batch-size 4 --dry-run
"""
import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from dataset import SimpleSliceDataset
from modules import UNetSmall

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sigmoid(x):
    return torch.sigmoid(x)

def dice_coeff_from_logits(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    inter = (preds * target).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2*inter + eps) / (union + eps)
    return dice.mean().item()

def dice_loss_from_logits(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(1,2,3))
    union = probs.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    loss = 1 - ((2*inter + eps) / (union + eps))
    return loss.mean()

def train_epoch(model, loader, optim, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    n = 0
    for x,y in tqdm(loader, desc="train", leave=False):
        x = x.to(device); y = y.to(device)
        optim.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y) + dice_loss_from_logits(logits, y)
        loss.backward()
        optim.step()
        total_loss += loss.item() * x.size(0)
        total_dice += dice_coeff_from_logits(logits, y) * x.size(0)
        n += x.size(0)
    return total_loss / n, total_dice / n

def val_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    n = 0
    with torch.no_grad():
        for x,y in tqdm(loader, desc="val", leave=False):
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y) + dice_loss_from_logits(logits, y)
            total_loss += loss.item() * x.size(0)
            total_dice += dice_coeff_from_logits(logits, y) * x.size(0)
            n += x.size(0)
    return total_loss / n, total_dice / n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--save-dir', type=str, default='./checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    results_dir = os.path.join(args.save_dir, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = SimpleSliceDataset(args.data_root, early_stop=args.dry_run)
    n = len(ds)
    if n < 2:
        raise RuntimeError("Dataset too small or not found. Check data_root images/ and masks/")

    train_n = int(0.8 * n)
    val_n = n - train_n
    train_ds, val_ds = random_split(ds, [train_n, val_n])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = UNetSmall(in_ch=1, out_ch=1).to(device)
    optim = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optim, step_size=10, gamma=0.5)
    bce = nn.BCEWithLogitsLoss()

    best_val_dice = -1.0
    train_losses, val_losses, train_dices, val_dices = [], [], [], []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = train_epoch(model, train_loader, optim, bce, device)
        val_loss, val_dice = val_epoch(model, val_loader, bce, device)
        scheduler.step()
        train_losses.append(train_loss); val_losses.append(val_loss)
        train_dices.append(train_dice); val_dices.append(val_dice)

        print(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f}, train_dice={train_dice:.4f} | val_loss={val_loss:.4f}, val_dice={val_dice:.4f}")

        # save best by val_dice
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optim.state_dict(),
                'val_dice': val_dice
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best.pth'))

        # optionally quick exit for dry-run after 1 epoch
        if args.dry_run:
            break

    # Save final plots
    try:
        plt.figure(); plt.plot(train_losses, label='train_loss'); plt.plot(val_losses, label='val_loss'); plt.legend(); plt.savefig(os.path.join(results_dir, 'loss.png')); plt.close()
        plt.figure(); plt.plot(train_dices, label='train_dice'); plt.plot(val_dices, label='val_dice'); plt.legend(); plt.savefig(os.path.join(results_dir, 'dice.png')); plt.close()
    except Exception as e:
        print("Warning: failed to save plots:", e)

    print("Training complete. Best val dice:", best_val_dice)
    print("Checkpoints saved to:", args.save_dir)
    print("Plots saved to:", results_dir)

if __name__ == "__main__":
    main()
