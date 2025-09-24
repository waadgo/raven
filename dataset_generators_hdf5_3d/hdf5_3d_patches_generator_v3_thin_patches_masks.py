#!/usr/bin/env python
import os
import time
import csv
import h5py
import numpy as np
import argparse
import nibabel as nib
from nibabel import processing as nibp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
import gc
import torch
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# -------------------------------------------------------------------------
# 0) Visualization function (called only in main thread)
# -------------------------------------------------------------------------
def generate_snapshot_from_patches(patches, subject_basename, perm, snapshot_dir):
    """
    Generates and saves a PNG snapshot (grid of up to 16 patches) from the given patches.
    This function is called in the main thread.
    """
    n_patches = patches.shape[0]
    n_samples = min(n_patches, 16)
    sample_indices = np.random.choice(n_patches, size=n_samples, replace=False)
    sampled_patches = patches[sample_indices]
    patch_slices = [p[:, :, p.shape[2] // 2] for p in sampled_patches]
    grid_cols = math.ceil(math.sqrt(n_samples))
    grid_rows = math.ceil(n_samples / grid_cols)
    fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 2, grid_rows * 2))
    if grid_rows * grid_cols > 1:
        axs = np.array(axs).flatten()
    else:
        axs = [axs]
    for i in range(len(axs)):
        if i < n_samples:
            axs[i].imshow(patch_slices[i], cmap='gray')
            axs[i].axis('off')
        else:
            axs[i].axis('off')
    plt.tight_layout()
    perm_suffix = "".join(str(x) for x in perm)
    snap_fname = os.path.join(snapshot_dir, subject_basename + f"_perm{perm_suffix}.png")
    plt.savefig(snap_fname)
    plt.close(fig)
    print(f"Saved snapshot for {subject_basename} (perm={perm_suffix}) -> {snap_fname}")

# -------------------------------------------------------------------------
# 1) Robust rescaling functions (unchanged)
# -------------------------------------------------------------------------
def getscale(data: np.ndarray, dst_min: float, dst_max: float, f_low: float = 0.0, f_high: float = 0.995):
    src_min = np.min(data)
    src_max = np.max(data)
    if src_min < 0.0:
        print("WARNING: Input image has value(s) below 0.0!")
    histosize = 1000
    bin_size = (src_max - src_min) / histosize
    hist, _ = np.histogram(data, bins=histosize)
    cs = np.concatenate(([0], np.cumsum(hist)))
    voxnum = data.size
    nth_low = int(f_low * voxnum)
    idx_low = np.where(cs < nth_low)[0]
    idx_low = idx_low[-1] + 1 if len(idx_low) > 0 else 0
    robust_min = src_min + idx_low * bin_size
    nth_high = voxnum - int((1.0 - f_high) * voxnum)
    idx_high = np.where(cs >= nth_high)[0]
    idx_high = idx_high[0] - 1 if len(idx_high) > 0 else histosize - 1
    robust_max = src_min + idx_high * bin_size
    scale = 1.0 if robust_max == robust_min else (dst_max - dst_min) / (robust_max - robust_min)
    print(f"Robust rescale: robust_min = {robust_min:.2f}, robust_max = {robust_max:.2f}, scale = {scale:.4f}")
    return robust_min, scale

def scalecrop(data: np.ndarray, dst_min: float, dst_max: float, src_min: float, scale: float):
    data_new = dst_min + scale * (data - src_min)
    return np.clip(data_new, dst_min, dst_max)

# -------------------------------------------------------------------------
# 2) Modified patch extraction: center crop for larger dims, sliding for smallest dim
# -------------------------------------------------------------------------
def extract_3d_patches(volume: np.ndarray, mask: np.ndarray,
                       patch_size=(64, 64, 64), stride=None) -> np.ndarray:
    """
    Extracts 3D patches using a center crop for dimensions where the patch size is
    larger than the smallest patch dimension (which is used as stride) and sliding window
    for the dimension(s) equal to the smallest patch dimension.
    
    If the volume is smaller than patch_size along any axis, symmetric zero-padding is applied.
    
    For example, for a volume of 256x256x256 and patch_size (192,192,32):
      - Let S = min(192,192,32) = 32.
      - For axis 0 and 1 (192 > 32), only one patch is extracted, centered.
      - For axis 2 (32 == S), sliding is applied with stride 32.
      This yields 1*1*8 = 8 patches.
    """
    if stride is None:
        S = min(patch_size)
        stride = (S, S, S)
    else:
        S = min(patch_size)
        stride = (S, S, S)
    # Pad volume and mask if needed so each dim >= patch_size.
    pad_width = []
    for i in range(3):
        D = volume.shape[i]
        P = patch_size[i]
        if D < P:
            needed = P - D
            pad_left = needed // 2
            pad_right = needed - pad_left
        else:
            pad_left = 0
            pad_right = 0
        pad_width.append((pad_left, pad_right))
    if any(p != (0, 0) for p in pad_width):
        volume = np.pad(volume, pad_width, mode='constant', constant_values=0)
        mask = np.pad(mask, pad_width, mode='constant', constant_values=0)
    # For each axis, determine patch start positions:
    indices = []
    for i in range(3):
        P = patch_size[i]
        D = volume.shape[i]
        if P > S:
            # Only one patch: center crop.
            start = (D - P) // 2
            indices.append([start])
        else:
            # Sliding window positions.
            positions = list(range(0, D - P + 1, S))
            indices.append(positions)
    patches = []
    # Iterate over all combinations of start positions.
    for x in indices[0]:
        for y in indices[1]:
            for z in indices[2]:
                patch_vol = volume[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]]
                patch_mask = mask[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]]
                frac_fg = patch_mask.sum() / patch_mask.size
                # Apply random drop for patches with <20% foreground.
                if frac_fg < 0.2 and np.random.rand() < 0.5:
                    continue
                patches.append(patch_vol)
    if len(patches) == 0:
        return np.empty((0,) + patch_size, dtype=volume.dtype)
    return np.stack(patches, axis=0)

# -------------------------------------------------------------------------
# 3) CSV splitting utilities (unchanged)
# -------------------------------------------------------------------------
def get_splits(csv_file, total_splits):
    with open(csv_file, 'r') as f:
        row_count = sum(1 for _ in f)
    chunk_size = row_count // total_splits
    splits = [i * chunk_size for i in range(total_splits)]
    splits.append(row_count)
    return splits

def is_idx_within_split(idx, current_split, split_indices, total_splits):
    return split_indices[current_split] <= idx < split_indices[current_split + 1]

def get_new_fname(base_name, current_split, total_splits):
    if total_splits > 1:
        name, ext = os.path.splitext(base_name)
        return f"{name}_{current_split}{ext}"
    return base_name

# -------------------------------------------------------------------------
# 4) load_image_masked with memory mapping (unchanged)
# -------------------------------------------------------------------------
def load_image_masked(img_filename, im_threshold=15):
    orig = nib.load(img_filename, mmap=True)
    shape = orig.shape
    zoom = orig.header.get_zooms()
    print("Reshaping to standard RAS orientation...")
    orig = nibp.conform(orig, out_shape=shape, voxel_size=zoom, order=3, cval=0.0,
                        orientation='RAS', out_class=None)
    if len(zoom) == 4:
        zoom = zoom[:-1]
    orig_data = orig.get_fdata().astype(np.float32)
    src_min, scale = getscale(orig_data, 0, 255, f_low=0.0, f_high=0.995)
    orig_data = scalecrop(orig_data, 0, 255, src_min, scale)
    orig_data = orig_data.astype(np.uint8)
    mask = (orig_data > im_threshold).astype(np.uint8)
    return zoom, orig_data, mask

# -------------------------------------------------------------------------
# 5) Flush buffered data to HDF5 (unchanged)
# -------------------------------------------------------------------------
def flush_to_disk(hf, ds_patches, ds_zooms_x, ds_zooms_y, ds_zooms_z, ds_subjects,
                  patch_buffer, zoomx_buffer, zoomy_buffer, zoomz_buffer, subj_buffer, current_size):
    if not patch_buffer:
        return current_size
    arr = np.concatenate(patch_buffer, axis=0)  # shape = (B, px, py, pz)
    B = arr.shape[0]
    new_size = current_size + B
    ds_patches.resize((new_size,) + ds_patches.shape[1:])
    ds_zooms_x.resize((new_size,))
    ds_zooms_y.resize((new_size,))
    ds_zooms_z.resize((new_size,))
    ds_subjects.resize((new_size,))
    ds_patches[current_size:new_size, ...] = arr
    ds_zooms_x[current_size:new_size] = zoomx_buffer
    ds_zooms_y[current_size:new_size] = zoomy_buffer
    ds_zooms_z[current_size:new_size] = zoomz_buffer
    ds_subjects[current_size:new_size] = subj_buffer
    print(f"Flushed {B} patches to disk; total patches so far: {new_size}")
    patch_buffer.clear()
    zoomx_buffer.clear()
    zoomy_buffer.clear()
    zoomz_buffer.clear()
    subj_buffer.clear()
    return new_size

# -------------------------------------------------------------------------
# 6) Worker function (snapshot generation is not done here)
# -------------------------------------------------------------------------
def process_subject_and_extract(subject_path, im_threshold, patch_size, stride, selected_perms):
    try:
        zoom, volume, mask = load_image_masked(subject_path, im_threshold=im_threshold)
        subject_basename = os.path.basename(subject_path)
        perm = random.choice(selected_perms)
        volume_perm = np.transpose(volume, perm)
        mask_perm   = np.transpose(mask, perm)
        zoom_perm   = [zoom[perm[0]], zoom[perm[1]], zoom[perm[2]]]
        patches = extract_3d_patches(volume_perm, mask_perm, patch_size=patch_size, stride=stride)
        n_patches = patches.shape[0]
        if n_patches == 0:
            return None
        zoomx_vals = [zoom_perm[0]] * n_patches
        zoomy_vals = [zoom_perm[1]] * n_patches
        zoomz_vals = [zoom_perm[2]] * n_patches
        subj_vals  = [(subject_basename + f"_p{''.join(str(x) for x in perm)}").encode("ascii", "ignore")] * n_patches
        # Return patch data and metadata, plus subject_basename and chosen permutation for snapshot.
        return (patches, zoomx_vals, zoomy_vals, zoomz_vals, subj_vals, subject_basename, perm)
    except Exception as e:
        print(f"[ERROR] While processing {subject_path}: {e}")
        return None

# -------------------------------------------------------------------------
# 7) Main dataset class with parallel option and main-thread snapshot generation
# -------------------------------------------------------------------------
class PopulationDataset:
    def __init__(self, params):
        self.csv_file = params["csv_file"]
        self.dataset_name = params["dataset_name"]
        self.dataset_path = params["dataset_path"]
        self.im_threshold = params["im_threshold"]
        self.total_splits = params["total_splits"]
        self.buffer_size  = params.get("buffer_size", 200)
        self.num_workers  = params.get("num_workers", 1)
    
    def create_hdf5_dataset(self, patch_size=(64,64,64), stride=None):
        # Use stride equal to the smallest dimension of the patch.
        if stride is None:
            S = min(patch_size)
            stride = (S, S, S)
        else:
            S = min(patch_size)
            stride = (S, S, S)
        start_time = time.time()
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path, exist_ok=True)
            print("Created folder " + self.dataset_path)
        snapshot_dir = os.path.join(self.dataset_path, "snapshots", "snaps")
        os.makedirs(snapshot_dir, exist_ok=True)
        selected_perms = [(0,1,2), (1,0,2), (2,1,0)]
        split_indices = get_splits(self.csv_file, self.total_splits)
        with open(self.csv_file, 'r') as file:
            csv_rows = list(csv.reader(file))
        for current_split in range(self.total_splits):
            current_hdf5_fname = os.path.join(self.dataset_path, self.dataset_name)
            current_hdf5_fname = get_new_fname(current_hdf5_fname, current_split, self.total_splits)
            print(f"\n=== Creating HDF5 for split {current_split+1}/{self.total_splits} ===")
            print(f"Output file: {current_hdf5_fname}")
            with h5py.File(current_hdf5_fname, "w") as hf:
                ds_patches = hf.create_dataset(
                    "orig_dataset_imgs",
                    shape=(0, patch_size[0], patch_size[1], patch_size[2]),
                    maxshape=(None, patch_size[0], patch_size[1], patch_size[2]),
                    dtype=np.uint8,
                    compression='gzip'
                )
                ds_zooms_x = hf.create_dataset("orig_zooms_x", shape=(0,), maxshape=(None,), dtype=np.float32, compression='gzip')
                ds_zooms_y = hf.create_dataset("orig_zooms_y", shape=(0,), maxshape=(None,), dtype=np.float32, compression='gzip')
                ds_zooms_z = hf.create_dataset("orig_zooms_z", shape=(0,), maxshape=(None,), dtype=np.float32, compression='gzip')
                dt = h5py.string_dtype('ascii')
                ds_subj = hf.create_dataset("subject", shape=(0,), maxshape=(None,), dtype=dt, compression='gzip')
                patch_buffer = []
                zoomx_buffer = []
                zoomy_buffer = []
                zoomz_buffer = []
                subj_buffer  = []
                current_size = 0
                split_start = split_indices[current_split]
                split_end   = split_indices[current_split+1]
                if self.num_workers > 1:
                    ExecutorClass = ThreadPoolExecutor  # or ProcessPoolExecutor if CPU-bound
                    tasks = []
                    with ExecutorClass(max_workers=self.num_workers) as executor:
                        for idx in range(split_start, split_end):
                            row = csv_rows[idx]
                            if not row:
                                continue
                            subject_path = row[0]
                            fut = executor.submit(
                                process_subject_and_extract,
                                subject_path,
                                self.im_threshold,
                                patch_size,
                                stride,
                                selected_perms
                            )
                            tasks.append(fut)
                        for fut in as_completed(tasks):
                            result = fut.result()
                            if result is None:
                                continue
                            patches, zx, zy, zz, subs, subject_basename, perm = result
                            # Generate snapshot in main thread.
                            generate_snapshot_from_patches(patches, subject_basename, perm, snapshot_dir)
                            patch_buffer.append(patches)
                            zoomx_buffer.extend(zx)
                            zoomy_buffer.extend(zy)
                            zoomz_buffer.extend(zz)
                            subj_buffer.extend(subs)
                            buf_len = sum(arr.shape[0] for arr in patch_buffer)
                            if buf_len >= self.buffer_size:
                                current_size = flush_to_disk(
                                    hf,
                                    ds_patches, ds_zooms_x, ds_zooms_y, ds_zooms_z, ds_subj,
                                    patch_buffer, zoomx_buffer, zoomy_buffer, zoomz_buffer, subj_buffer,
                                    current_size
                                )
                else:
                    for idx in range(split_start, split_end):
                        row = csv_rows[idx]
                        if not row:
                            continue
                        subject_path = row[0]
                        result = process_subject_and_extract(
                            subject_path,
                            self.im_threshold,
                            patch_size,
                            stride,
                            selected_perms
                        )
                        if result is None:
                            continue
                        patches, zx, zy, zz, subs, subject_basename, perm = result
                        generate_snapshot_from_patches(patches, subject_basename, perm, snapshot_dir)
                        patch_buffer.append(patches)
                        zoomx_buffer.extend(zx)
                        zoomy_buffer.extend(zy)
                        zoomz_buffer.extend(zz)
                        subj_buffer.extend(subs)
                        buf_len = sum(arr.shape[0] for arr in patch_buffer)
                        if buf_len >= self.buffer_size:
                            current_size = flush_to_disk(
                                hf,
                                ds_patches, ds_zooms_x, ds_zooms_y, ds_zooms_z, ds_subj,
                                patch_buffer, zoomx_buffer, zoomy_buffer, zoomz_buffer, subj_buffer,
                                current_size
                            )
                current_size = flush_to_disk(
                    hf,
                    ds_patches, ds_zooms_x, ds_zooms_y, ds_zooms_z, ds_subj,
                    patch_buffer, zoomx_buffer, zoomy_buffer, zoomz_buffer, subj_buffer,
                    current_size
                )
            print(f"[SPLIT {current_split}] Done writing: {current_hdf5_fname}")
        elapsed = time.time() - start_time
        print(f"All splits processed in {elapsed:.1f} seconds total.")

# -------------------------------------------------------------------------
# 8) Main command-line entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='HDF5 Creation for 3D Patches with random axis permutation per subject. '
                    'Uses robust rescaling, center-cropped patch extraction (with sliding along the smallest dimension), '
                    'buffered HDF5 writes, memory‐mapped loads, and parallel processing. '
                    'Snapshots are saved only as PNG files.'
    )
    parser.add_argument('--hdf5_name', type=str, default=None,
                        help='Name of output HDF5 dataset (for single CSV mode).')
    parser.add_argument('--hdf5_path', type=str,
                        default="/data/dadmah/gonwal2/Documents/SuperResolution/datasets/RAVEN_LDM_3D_v4_aniso/",
                        help='Folder to save the HDF5 dataset(s).')
    parser.add_argument('--csv_file_images', type=str,
                        default="/data/dadmah/gonwal2/Documents/datasets/Datasets_raw/ALL_pp1_datasplit_csvs/",
                        help="Either a CSV file or a directory with _train/_val/_test CSVs.")
    parser.add_argument('--patch_size', nargs=3, type=int, default=[192,192,32],
                        help="Size of each 3D patch: [P1, P2, P3].")
    parser.add_argument('--stride', nargs=3, type=int, default=None,
                        help="(Ignored) Stride is set to the smallest dimension of the patch.")
    parser.add_argument('--intensity_threshold', type=float, default=15,
                        help="Foreground threshold for mask (voxels > threshold).")
    parser.add_argument('--total_splits', type=int, default=1,
                        help="Number of HDF5 files to produce from the CSV rows.")
    parser.add_argument('--suffix', type=str, default="v4_aniso",
                        help="Optional suffix for the output HDF5 filename.")
    parser.add_argument('--buffer_size', type=int, default=200,
                        help="Number of patches to buffer before flushing to HDF5.")
    parser.add_argument('--num_workers', type=int, default=24,
                        help="If >1, enable parallel loading & patch extraction with that many workers.")
    args = parser.parse_args()

    if os.path.isdir(args.csv_file_images):
        csv_dir = args.csv_file_images
        csv_files = [f for f in os.listdir(csv_dir)
                     if f.endswith('_train.csv') or f.endswith('_val.csv') or f.endswith('_test.csv')]
        if not csv_files:
            print("No CSV files with _train/_val/_test found in the directory.")
            exit(1)
        for csv_file in csv_files:
            full_csv_path = os.path.join(csv_dir, csv_file)
            if csv_file.endswith('_train.csv'):
                cat = 'train'
            elif csv_file.endswith('_val.csv'):
                cat = 'val'
            elif csv_file.endswith('_test.csv'):
                cat = 'test'
            else:
                cat = 'other'
            output_dir = os.path.join(args.hdf5_path, cat)
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(csv_file)[0]
            if args.suffix is not None:
                base_name = base_name + "_" + args.suffix
            dataset_name = base_name + '.hdf5'
            hdf5_full_path = os.path.join(output_dir, dataset_name)
            print(f"Processing CSV file: {full_csv_path} -> {hdf5_full_path}")
            info = {
                "dataset_name": dataset_name,
                "dataset_path": output_dir,
                "csv_file": full_csv_path,
                "im_threshold": args.intensity_threshold,
                "total_splits": args.total_splits,
                "buffer_size": args.buffer_size,
                "num_workers": args.num_workers
            }
            dataset_obj = PopulationDataset(info)
            dataset_obj.create_hdf5_dataset(
                patch_size=tuple(args.patch_size),
                stride=None  # stride is automatically set to the smallest patch dimension
            )
    else:
        output_dir = args.hdf5_path
        os.makedirs(output_dir, exist_ok=True)
        base_name, ext = os.path.splitext(args.hdf5_name)
        if args.suffix is not None:
            base_name = base_name + "_" + args.suffix
        dataset_name = base_name + ext
        hdf5_full_path = os.path.join(output_dir, dataset_name)
        if os.path.exists(hdf5_full_path):
            print(f"[SKIP] HDF5 file {hdf5_full_path} already exists.")
        else:
            info = {
                "dataset_name": dataset_name,
                "dataset_path": output_dir,
                "csv_file": args.csv_file_images,
                "im_threshold": args.intensity_threshold,
                "total_splits": args.total_splits,
                "buffer_size": args.buffer_size,
                "num_workers": args.num_workers
            }
            dataset_obj = PopulationDataset(info)
            dataset_obj.create_hdf5_dataset(
                patch_size=tuple(args.patch_size),
                stride=None
            )
