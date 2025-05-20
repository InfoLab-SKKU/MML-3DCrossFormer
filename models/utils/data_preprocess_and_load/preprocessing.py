import os
import torch
import numpy as np
from monai.transforms import LoadImage
from multiprocessing import Process, Queue
import torch.nn.functional as F


def center_crop_or_pad(tensor, target_shape=(96, 96, 96)):
    """
    Center-crop or pad a 3D tensor (C, D, H, W) to `target_shape`.
    """
    # tensor: (C, D, H, W)
    _, D, H, W = tensor.shape
    pads = []  # for F.pad: (W_left, W_right, H_left, H_right, D_left, D_right)
    slices = []
    for cur, tgt in zip((D, H, W), target_shape):
        if cur >= tgt:
            start = (cur - tgt) // 2
            slices.append(slice(start, start + tgt))
            pads.append((0, 0))
        else:
            pad_total = tgt - cur
            before = pad_total // 2
            after = pad_total - before
            slices.append(slice(0, cur))
            pads.append((before, after))
    # crop
    cropped = tensor[:, slices[0], slices[1], slices[2]]
    # prepare pad tuple in reverse order
    pad_cfg = [p for dims in reversed(pads) for p in dims]
    padded = F.pad(cropped, pad_cfg, "constant", 0)
    return padded


def read_data_adni(filename, load_root, save_root, scaling_method='z-norm'):
    """
    Preprocess a single ADNI T1-weighted NIfTI file:
    - Load, normalize (z-norm or minmax) within brain mask
    - Crop/pad to 96^3
    - Save as float16 .pt
    """
    subj = os.path.splitext(os.path.splitext(filename)[0])[0]
    print(f"Processing subject {subj}", flush=True)
    in_path = os.path.join(load_root, filename)
    out_dir = save_root
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{subj}.pt")
    if os.path.exists(out_path):
        return

    try:
        img, _ = LoadImage()(in_path)
    except Exception as e:
        print(f"Failed to load {filename}: {e}")
        return

    volume = img.numpy() if hasattr(img, 'numpy') else img
    mask = volume > 0

    if scaling_method == 'z-norm':
        mean = volume[mask].mean()
        std = volume[mask].std()
        volume = (volume - mean) / (std + 1e-8)
    else:
        mn = volume[mask].min()
        mx = volume[mask].max()
        volume = (volume - mn) / (mx - mn + 1e-8)

    volume[~mask] = 0
    tensor = torch.from_numpy(volume).unsqueeze(0).float()
    tensor = center_crop_or_pad(tensor, target_shape=(96, 96, 96))
    torch.save(tensor.type(torch.float16), out_path)


def main():
    load_root = '/path/to/adni_nifti'
    save_root = '/path/to/preprocessed_adni'
    scaling_method = 'z-norm'  # or 'minmax'

    files = [f for f in os.listdir(load_root) if f.endswith('.nii') or f.endswith('.nii.gz')]
    for filename in sorted(files):
        read_data_adni(filename, load_root, save_root, scaling_method)


if __name__ == '__main__':
    main()
