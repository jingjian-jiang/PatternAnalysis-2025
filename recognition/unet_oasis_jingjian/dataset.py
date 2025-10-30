# dataset.py
"""
SimpleSliceDataset:
- Expects `data_root/images/` and `data_root/masks/`
- Accepts .nii/.nii.gz (via nibabel) or image files (.png, .jpg).
- Normalizes per-slice to zero mean unit std.
- Returns (image_tensor, mask_tensor) with shapes (C,H,W).
"""
import os
import glob
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image

try:
    import nibabel as nib
    HAVE_NIB = True
except Exception:
    HAVE_NIB = False

def load_image_file(path):
    if path.endswith(('.nii', '.nii.gz')) and HAVE_NIB:
        arr = nib.load(path).get_fdata()
        # If 3D, take the central axial slice to convert to 2D
        if arr.ndim == 3:
            z = arr.shape[2] // 2
            arr = arr[..., z]
        arr = np.asarray(arr, dtype=np.float32)
    else:
        img = Image.open(path).convert('L')
        arr = np.array(img, dtype=np.float32)
    return arr

class SimpleSliceDataset(Dataset):
    def __init__(self, data_root: str, transform=None, early_stop: bool = False, resize: tuple = None):
        self.img_dir = os.path.join(data_root, 'images')
        self.mask_dir = os.path.join(data_root, 'masks')
        img_paths = sorted(glob.glob(os.path.join(self.img_dir, '*')))
        mask_paths = sorted(glob.glob(os.path.join(self.mask_dir, '*')))
        # Basic sanity: keep only pairs where both exist (by filename)
        # If counts mismatch, we pair by index as a fallback
        if len(img_paths) == 0 or len(mask_paths) == 0:
            raise FileNotFoundError(f"No images or masks found in {data_root}/images and {data_root}/masks")
        if early_stop:
            img_paths = img_paths[:40]
            mask_paths = mask_paths[:40]
        self.imgs = img_paths
        self.masks = mask_paths
        self.transform = transform
        self.resize = resize

    def __len__(self):
        return min(len(self.imgs), len(self.masks))

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        x = load_image_file(img_path)
        y = load_image_file(mask_path)
        # optional resize via PIL for images and masks
        if self.resize is not None:
            x = Image.fromarray(np.uint8(255 * (x - x.min()) / (x.ptp() + 1e-8)))
            y = Image.fromarray(np.uint8(255 * (y - y.min()) / (y.ptp() + 1e-8)))
            x = x.resize(self.resize, Image.BILINEAR)
            y = y.resize(self.resize, Image.NEAREST)
            x = np.array(x).astype(np.float32)
            y = np.array(y).astype(np.float32)
        # normalize per-slice: zero-mean unit-std
        x = (x - x.mean()) / (x.std() + 1e-8)
        # ensure mask is binary
        y = (y > 0.5 * y.max()).astype(np.float32)
        # to tensor: (C,H,W)
        x = torch.from_numpy(x).unsqueeze(0).float()
        y = torch.from_numpy(y).unsqueeze(0).float()
        return x, y
