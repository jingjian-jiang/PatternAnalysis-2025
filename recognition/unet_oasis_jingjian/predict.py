# predict.py
"""
Inference script that loads checkpoints/best.pth and generates visualizations.
Example:
 python predict.py --checkpoint checkpoints/best.pth --data-root /home/groups/comp3710/OASIS --outdir ./predictions --num 5
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from dataset import SimpleSliceDataset
from modules import UNetSmall

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help="path to checkpoint (best.pth)")
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--outdir', default='./predictions')
    parser.add_argument('--num', type=int, default=5)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    ensure_dir(args.outdir)

    ds = SimpleSliceDataset(args.data_root, early_stop=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    # load model
    model = UNetSmall(in_ch=1, out_ch=1).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    with torch.no_grad():
        for i, (x,y) in enumerate(loader):
            if i >= args.num:
                break
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0,0]
            inp = x.cpu().numpy()[0,0]
            tgt = y.numpy()[0,0]
            pred = (probs > 0.5).astype(float)
            fig, axes = plt.subplots(1,3,figsize=(9,3))
            axes[0].imshow(inp, cmap='gray'); axes[0].set_title('input'); axes[0].axis('off')
            axes[1].imshow(tgt, cmap='gray'); axes[1].set_title('target'); axes[1].axis('off')
            axes[2].imshow(pred, cmap='gray'); axes[2].set_title('pred'); axes[2].axis('off')
            plt.tight_layout()
            out_path = os.path.join(args.outdir, f'pred_{i}.png')
            fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            print("Saved:", out_path)

if __name__ == "__main__":
    main()
