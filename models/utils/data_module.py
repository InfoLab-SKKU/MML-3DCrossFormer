#!/usr/bin/env python3
"""
Comprehensive ADNI MRI Preprocessing Pipeline

This script performs advanced preprocessing for ADNI T1-weighted MRI scans:

Features:
  - Input directory scanning for .nii/.nii.gz files
  - Optional BIDS-style folder hierarchy support
  - Intensity normalization within brain mask: z-score or min-max
  - Bias field correction using N4 algorithm (MONAI)
  - Center-cropping or padding to target volume size
  - Optional histogram equalization
  - Optional Gaussian smoothing filter
  - Data augmentation hooks example (flip, rotate)
  - Multiprocessing with progress bar
  - Detailed logging (file and console)
  - Per-subject metadata JSON output
  - Summary CSV report with processing metrics

Usage:
    python preprocess_adni.py \
        --input_dir /data/adni_nifti \
        --output_dir /data/adni_preprocessed \
        --scaling z-norm \
        --bias_correct \
        --equalize \
        --smooth 0.5 \
        --augment \
        --target_shape 96 96 96 \
        --workers 8 \
        --report summary.csv
"""
import os
import sys
import json
import argparse
import logging
import csv
import time
from multiprocessing import Pool, Manager
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import LoadImage, N4BiasFieldCorrection
from monai.networks.layers import GaussianFilter
from skimage import exposure
from tqdm import tqdm

# -----------------------
# Utility Functions
# -----------------------
def setup_logging(log_file: str = None):
    """
    Configure logging to console and file (if provided).
    """
    log_format = "%(asctime)s %(levelname)s:%(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=[])
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        fh = logging.FileHandler(log_file)
        handlers.append(fh)
    for h in handlers:
        logging.getLogger().addHandler(h)


def center_crop_or_pad(volume: np.ndarray, target_shape=(96,96,96)) -> np.ndarray:
    """Center-crop or pad a 3D NumPy array to target_shape."""
    D, H, W = volume.shape
    output = np.zeros(target_shape, dtype=volume.dtype)
    # Determine cropping and padding indices
    for axis, (cur, tgt) in enumerate(zip((D, H, W), target_shape)):
        # compute start indices
        if cur >= tgt:
            start_cur = (cur - tgt) // 2
            end_cur = start_cur + tgt
            slice_src = slice(start_cur, end_cur)
            slice_dst = slice(0, tgt)
        else:
            # pad
            pad_before = (tgt - cur) // 2
            pad_after = tgt - cur - pad_before
            slice_src = slice(0, cur)
            slice_dst = slice(pad_before, pad_before + cur)
        # store per-axis
        if axis == 0:
            src0, dst0 = slice_src, slice_dst
        elif axis == 1:
            src1, dst1 = slice_src, slice_dst
        else:
            src2, dst2 = slice_src, slice_dst
    cropped = volume[src0, src1, src2]
    output[dst0, dst1, dst2] = cropped
    return output


def histogram_equalization(volume: np.ndarray) -> np.ndarray:
    """Apply histogram equalization within brain mask."""
    mask = volume > 0
    eq = exposure.equalize_adapthist(volume, clip_limit=0.03)
    volume[mask] = eq[mask]
    return volume


def extract_metadata(header) -> dict:
    """Parse NIfTI header or loaded metadata for JSON output."""
    meta = {}
    if hasattr(header, 'get'):  # MONAI metadata
        meta['affine'] = header.get('affine').tolist() if header.get('affine') is not None else None
        meta['original_shape'] = header.get('spatial_shape')
    else:
        try:
            meta['shape'] = header.get_data_shape()
        except:
            meta['shape'] = None
    return meta


def preprocess_subject(filename: str,
                       input_dir: str,
                       output_dir: str,
                       scaling: str,
                       bias_correct: bool,
                       equalize: bool,
                       smooth_sigma: float,
                       augment: bool,
                       target_shape: tuple) -> dict:
    """
    Process one subject: load, preprocess, save tensor & metadata.
    Returns status dict with timing and errors.
    """
    logger = logging.getLogger('preprocess_subject')
    start = time.time()
    subj_id = os.path.splitext(os.path.splitext(filename)[0])[0]
    in_path = os.path.join(input_dir, filename)
    out_tensor = os.path.join(output_dir, f"{subj_id}.pt")
    out_meta = os.path.join(output_dir, f"{subj_id}_meta.json")
    status = {'subject': subj_id, 'status': 'skipped', 'error': '', 'duration': 0.0}
    try:
        img, meta = LoadImage(image_only=False)(in_path)
        vol = img.numpy() if hasattr(img, 'numpy') else img
        # optional bias field correction
        if bias_correct:
            img_bc = N4BiasFieldCorrection()(img.unsqueeze(0).unsqueeze(0))
            vol = img_bc.squeeze().numpy()
        # brain mask
        mask = vol > 0
        # normalization
        if scaling == 'z-norm':
            m = vol[mask].mean(); s = vol[mask].std()
            vol = (vol - m)/(s + 1e-8)
        else:
            mn = vol[mask].min(); mx = vol[mask].max()
            vol = (vol - mn)/(mx - mn + 1e-8)
        vol[~mask] = 0
        # histogram equalization
        if equalize:
            vol = histogram_equalization(vol)
        # center crop/pad
        vol = center_crop_or_pad(vol, target_shape)
        # smoothing
        if smooth_sigma > 0:
            tensor = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float()
            tensor = GaussianFilter(sigma=smooth_sigma)(tensor)
            vol = tensor.squeeze().numpy()
        # data augmentation example
        if augment:
            # simple flip augmentation
            vol = np.flip(vol, axis=2).copy()
        # save tensor
        tensor = torch.from_numpy(vol).unsqueeze(0).float().half()
        os.makedirs(output_dir, exist_ok=True)
        torch.save(tensor, out_tensor)
        # save metadata
        meta_info = extract_metadata(meta)
        meta_info.update({'scaling': scaling, 'bias_correct': bias_correct,
                          'equalize': equalize, 'smooth_sigma': smooth_sigma,
                          'augment': augment, 'target_shape': target_shape})
        with open(out_meta, 'w') as fj:
            json.dump(meta_info, fj, indent=2)
        status['status'] = 'success'
    except Exception as e:
        logger.error(f"Error {subj_id}: {e}")
        status['status'] = 'failed'
        status['error'] = str(e)
    finally:
        status['duration'] = round(time.time() - start, 3)
    return status

# -----------------------
# Main Entry
# -----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Comprehensive ADNI MRI Preprocessing")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to ADNI NIfTI files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory for output tensors and metadata')
    parser.add_argument('--scaling', choices=['z-norm','minmax'], default='z-norm', help='Normalization method')
    parser.add_argument('--bias_correct', action='store_true', help='Apply N4 bias field correction')
    parser.add_argument('--equalize', action='store_true', help='Apply histogram equalization')
    parser.add_argument('--smooth', type=float, default=0.0, help='Gaussian smoothing sigma')
    parser.add_argument('--augment', action='store_true', help='Perform simple flip augmentation')
    parser.add_argument('--target_shape', nargs=3, type=int, default=[96,96,96], help='Output shape (D H W)')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--report', type=str, default='summary.csv', help='CSV summary report filename')
    parser.add_argument('--log_file', type=str, default=None, help='Optional log file path')
    args = parser.parse_args()

    setup_logging(args.log_file)
    logger = logging.getLogger('main')
    logger.info(f"Starting preprocessing with {args.workers} workers...")

    # gather files
    files = [f for f in os.listdir(args.input_dir) if f.endswith(('.nii','.nii.gz'))]
    files.sort()
    logger.info(f"Found {len(files)} input volumes in {args.input_dir}")

    # parallel processing
    pool = Pool(processes=args.workers)
    func = partial(preprocess_subject,
                   input_dir=args.input_dir,
                   output_dir=args.output_dir,
                   scaling=args.scaling,
                   bias_correct=args.bias_correct,
                   equalize=args.equalize,
                   smooth_sigma=args.smooth,
                   augment=args.augment,
                   target_shape=tuple(args.target_shape))
    results = []
    for res in tqdm(pool.imap(func, files), total=len(files)):
        results.append(res)
    pool.close(); pool.join()

    # write summary report
    report_path = os.path.join(args.output_dir, args.report)
    with open(report_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['subject','status','error','duration'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    logger.info(f"Preprocessing complete. Summary saved to {report_path}")
