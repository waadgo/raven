import os
import time
import csv
import h5py
import numpy as np
import argparse
import nibabel as nib
from nibabel import processing as nibp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import math
from typing import Tuple
import torch
import itertools  # <-- New import for permutations

def volshow(volume, save_path="/data/dadmah/gonwal2/Documents/SuperResolution/networks/taming-transformers3d_local/snapshots", filename="volshow.png"):
    """
    Saves a 3D volume visualization as a 3x3 grid PNG file.
    
    The function accepts:
      - A 3D array of shape [H, W, D],
      - A 4D array of shape [B, H, W, D] (first sample used),
      - A 5D array of shape [B, C, H, W, D] (first sample, first channel used).
    
    For the selected 3D volume, it extracts slices at 25%, 50%, and 75% along each axis and 
    arranges them into a 3x3 grid:
      - Row 0: Slices along axis 0 (volume[slice, :, :])
      - Row 1: Slices along axis 1 (volume[:, slice, :])
      - Row 2: Slices along axis 2 (volume[:, :, slice])
    
    The resulting grid is saved as a PNG file in the specified directory.
    """
    # If input is a torch tensor, convert it to a NumPy array.
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()
    
    # Handle batched shapes:
    if volume.ndim == 5:
        # Expected shape [B, C, H, W, D]; take first sample and first channel.
        volume = volume[0, 0, :, :, :]
    elif volume.ndim == 4:
        # Expected shape [B, H, W, D]; take first sample.
        volume = volume[0, :, :, :]
    elif volume.ndim != 3:
        raise ValueError("Input volume must be 3D, 4D, or 5D.")
    
    # Now volume is a 3D array of shape [H, W, D]
    H, W, D = volume.shape

    # Define slice indices at 25%, 50%, and 75% for each axis.
    idx_H = [int(H * 0.25), int(H * 0.5), int(H * 0.75)]
    idx_W = [int(W * 0.25), int(W * 0.5), int(W * 0.75)]
    idx_D = [int(D * 0.25), int(D * 0.5), int(D * 0.75)]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    # Row 0: slices along axis 0
    for j, idx in enumerate(idx_H):
        img = volume[idx, :, :]
        axes[0, j].imshow(img, cmap='gray')
        axes[0, j].set_title(f"Axis 0, slice {idx}")
        axes[0, j].axis('off')
    
    # Row 1: slices along axis 1
    for j, idx in enumerate(idx_W):
        img = volume[:, idx, :]
        axes[1, j].imshow(img, cmap='gray')
        axes[1, j].set_title(f"Axis 1, slice {idx}")
        axes[1, j].axis('off')
    
    # Row 2: slices along axis 2
    for j, idx in enumerate(idx_D):
        img = volume[:, :, idx]
        axes[2, j].imshow(img, cmap='gray')
        axes[2, j].set_title(f"Axis 2, slice {idx}")
        axes[2, j].axis('off')
    
    plt.tight_layout()
    
    # Ensure the output directory exists.
    os.makedirs(save_path, exist_ok=True)
    out_filepath = os.path.join(save_path, filename)
    
    # Save the figure as a PNG file.
    plt.savefig(out_filepath)
    plt.close(fig)
    print(f"Saved volume visualization to {out_filepath}")


# ------------------------------------------------------------------------
# Robust rescaling functions using a histogram approach
# ------------------------------------------------------------------------
def getscale(
    data: np.ndarray,
    dst_min: float,
    dst_max: float,
    f_low: float = 0.0,
    f_high: float = 0.995
) -> Tuple[float, float]:
    """
    Compute a robust lower bound (src_min) and a scale factor so that a robust range
    of data is mapped to [dst_min, dst_max]. The method uses a histogram-based approach.
    
    Parameters:
        data : np.ndarray
            Input image data.
        dst_min : float
            Target minimum value.
        dst_max : float
            Target maximum value.
        f_low : float, optional
            Fraction to crop at the low end (default is 0.0).
        f_high : float, optional
            Fraction to crop at the high end (default is 0.995).
            
    Returns:
        src_min : float
            The robust lower bound computed from the histogram.
        scale : float
            The scale factor so that the robust range maps to [dst_min, dst_max].
    """
    src_min = np.min(data)
    src_max = np.max(data)
    if src_min < 0.0:
        print("WARNING: Input image has value(s) below 0.0!")
    
    # Compute histogram.
    histosize = 1000
    bin_size = (src_max - src_min) / histosize
    hist, _ = np.histogram(data, bins=histosize)
    cs = np.concatenate(([0], np.cumsum(hist)))
    voxnum = data.size
    
    # Compute robust lower bound.
    nth_low = int(f_low * voxnum)
    idx_low = np.where(cs < nth_low)[0]
    if len(idx_low) > 0:
        idx_low = idx_low[-1] + 1
    else:
        idx_low = 0
    robust_min = src_min + idx_low * bin_size
    
    # Compute robust upper bound.
    nth_high = voxnum - int((1.0 - f_high) * voxnum)
    idx_high = np.where(cs >= nth_high)[0]
    if len(idx_high) > 0:
        idx_high = idx_high[0] - 1
    else:
        idx_high = histosize - 1
    robust_max = src_min + idx_high * bin_size

    if robust_max == robust_min:
        scale = 1.0
    else:
        scale = (dst_max - dst_min) / (robust_max - robust_min)
    
    print(f"Robust rescale: robust_min = {robust_min:.2f}, robust_max = {robust_max:.2f}, scale = {scale:.4f}")
    return robust_min, scale

def scalecrop(
    data: np.ndarray,
    dst_min: float,
    dst_max: float,
    src_min: float,
    scale: float
) -> np.ndarray:
    """
    Apply scaling and cropping to map data to the [dst_min, dst_max] range.
    
    Parameters:
        data : np.ndarray
            Input image data.
        dst_min : float
            Target minimum intensity.
        dst_max : float
            Target maximum intensity.
        src_min : float
            Robust lower bound from the original data.
        scale : float
            Scale factor computed by getscale.
            
    Returns:
        np.ndarray:
            Scaled image data with intensities clipped to [dst_min, dst_max].
    """
    data_new = dst_min + scale * (data - src_min)
    data_new = np.clip(data_new, dst_min, dst_max)
    return data_new


# ------------------------------------------------------------------------
# 1) Extract 3D patches (without bounding box cropping)
#    and randomly remove 50% of patches that have <20% foreground.
# ------------------------------------------------------------------------
def extract_3d_patches(
    volume: np.ndarray,
    mask: np.ndarray,
    patch_size=(64, 64, 64),
    stride=(32, 32, 32)
) -> np.ndarray:
    """
    Given a 3D volume (e.g., a full image) and its corresponding mask (computed as intensity > 15),
    zero-pad them symmetrically so that (volume_dim + padding - patch_size) is divisible
    by the stride in each dimension. Then extract 3D patches using a sliding-window approach.
    
    Additionally, if a patch has less than 20% of its voxels marked as foreground,
    it is dropped with a probability of 50%.
    
    Args:
        volume (np.ndarray): 3D volume (Dx, Dy, Dz).
        mask (np.ndarray): 3D mask (same shape as volume), with 1 indicating voxels
                           where intensity > 15 and 0 otherwise.
        patch_size (tuple): Size (px, py, pz) of each patch.
        stride (tuple): Stride (sx, sy, sz) for the sliding window.
        
    Returns:
        np.ndarray: A 4D array of shape (N, px, py, pz) containing the extracted patches.
    """
    px, py, pz = patch_size
    sx, sy, sz = stride

    Dx, Dy, Dz = volume.shape
    if mask.shape != volume.shape:
        raise ValueError("Volume and mask must have the same shape.")

    # Compute symmetrical padding for a given dimension.
    def compute_padding(D, p, s):
        leftover = (D - p) % s
        needed = 0
        if D < p:
            needed = (p - D)
        if leftover != 0:
            needed = max(needed, s - leftover)
        pad_left = needed // 2
        pad_right = needed - pad_left
        return pad_left, pad_right

    pad_x_left, pad_x_right = compute_padding(Dx, px, sx)
    pad_y_left, pad_y_right = compute_padding(Dy, py, sy)
    pad_z_left, pad_z_right = compute_padding(Dz, pz, sz)

    # Zero-pad the volume and mask symmetrically.
    volume_padded = np.pad(
        volume,
        pad_width=((pad_x_left, pad_x_right),
                   (pad_y_left, pad_y_right),
                   (pad_z_left, pad_z_right)),
        mode='constant',
        constant_values=0
    )
    mask_padded = np.pad(
        mask,
        pad_width=((pad_x_left, pad_x_right),
                   (pad_y_left, pad_y_right),
                   (pad_z_left, pad_z_right)),
        mode='constant',
        constant_values=0
    )

    Nx, Ny, Nz = volume_padded.shape
    kept_patches = []

    # Slide over the padded volume.
    for x in range(0, Nx - px + 1, sx):
        for y in range(0, Ny - py + 1, sy):
            for z in range(0, Nz - pz + 1, sz):
                patch_vol = volume_padded[x:x+px, y:y+py, z:z+pz]
                patch_mask = mask_padded[x:x+px, y:y+py, z:z+pz]

                frac_fg = patch_mask.sum() / patch_mask.size
                # For patches with less than 20% foreground, randomly drop 50%.
                if frac_fg < 0.2 and np.random.rand() < 0.5:
                    continue
                kept_patches.append(patch_vol)

    if len(kept_patches) == 0:
        return np.empty((0, px, py, pz), dtype=volume.dtype)
    return np.stack(kept_patches, axis=0)


# ------------------------------------------------------------------------
# 2) Utility functions for CSV splitting and file naming
# ------------------------------------------------------------------------
def get_splits(csv_file, total_splits):
    with open(csv_file, 'r') as f:
        row_count = sum(1 for _ in f)
    chunk_size = row_count // total_splits
    splits = [i * chunk_size for i in range(total_splits)]
    splits.append(row_count)
    return splits

def is_idx_within_split(idx, current_split, split_indices, total_splits):
    start_idx = split_indices[current_split]
    end_idx = split_indices[current_split + 1]
    return (start_idx <= idx < end_idx)

def get_new_fname(base_name, current_split, total_splits):
    if total_splits > 1:
        name, ext = os.path.splitext(base_name)
        return f"{name}_{current_split}{ext}"
    else:
        return base_name


# ------------------------------------------------------------------------
# 3) Minimal placeholder for load_image_masked.
#    (Replace with your actual loading & thresholding code)
# ------------------------------------------------------------------------
def load_image_masked(img_filename, im_threshold=15):
    """
    Load an image file using nibabel, conform the image to standard RAS orientation,
    and robustly rescale intensities to the range [0, 255] using a histogram-based approach.
    Also creates a mask where intensities exceed im_threshold.
    """
    orig = nib.load(img_filename)
    shape = orig.shape
    zoom = orig.header.get_zooms()
    print("Reshaping to standard RAS orientation...")
    orig = nibp.conform(orig, out_shape=shape, voxel_size=zoom, order=3, cval=0.0,
                        orientation='RAS', out_class=None)
    if len(zoom) == 4:
        zoom = zoom[:-1]
    # Load as float for robust rescaling.
    orig_data = orig.get_fdata().astype(np.float32)
    
    # Perform robust rescaling to the range [0, 255].
    src_min, scale = getscale(orig_data, 0, 255, f_low=0.0, f_high=0.995)
    orig_data = scalecrop(orig_data, 0, 255, src_min, scale)
    # Convert back to uint8.
    orig_data = orig_data.astype(np.uint8)
    
    # Create a mask using the intensity threshold.
    mask = (orig_data > im_threshold).astype(np.uint8)
    return zoom, orig_data, mask


# ------------------------------------------------------------------------
# 4) Main Class: PopulationDataset
# ------------------------------------------------------------------------
class PopulationDataset:
    """
    Class to load all images listed in a CSV file and extract 3D patches from
    all permutations of the axes. The final stored patches have shape (P1,P2,P3).
    """
    def __init__(self, params):
        self.csv_file = params["csv_file"]
        self.dataset_name = params["dataset_name"]
        self.dataset_path = params["dataset_path"]
        self.im_threshold = params["im_threshold"]
        self.total_splits = params["total_splits"]

    def create_hdf5_dataset(self, patch_size=(64, 64, 64), stride=(32, 32, 32)):
        """
        Creates HDF5 dataset(s) of 3D patches by:
          1) Loading each volume & mask from a CSV file.
          2) For each volume, loop over all permutations of (0,1,2):
             - Transpose volume & mask to match that permutation
             - Reorder zoom to match the permuted axes
             - Extract patches (size (P1,P2,P3)) with stride (S1,S2,S3)
          3) Randomly discard 50% of patches with <20% foreground (done in extract_3d_patches).
          4) Concatenate all patches (they keep shape (P1,P2,P3)) and store in HDF5.
          5) Store voxel sizes in permuted order as well.
          6) Save a snapshot PNG grid for up to 16 patches per permutation for each subject.
        """
        start_d = time.time()

        # Make sure the dataset path exists
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path, exist_ok=True)
            print("Created folder " + self.dataset_path)

        # Create folder structure for snapshots
        snapshot_dir = os.path.join(self.dataset_path, "snapshots", "snaps")
        os.makedirs(snapshot_dir, exist_ok=True)

        # All permutations of the 3 axes
        perms = list(itertools.permutations([0, 1, 2]))

        # Get split indices for partitioning CSV rows (if desired)
        split_indices = get_splits(self.csv_file, self.total_splits)

        for current_split in range(self.total_splits):
            all_patches = np.empty((0, *patch_size), dtype=np.uint8)
            all_zooms_x = []
            all_zooms_y = []
            all_zooms_z = []
            subjects = []

            with open(self.csv_file, mode='r') as file:
                csv_reader = csv.reader(file)
                idx = 0
                print(
                    f'Processing split {current_split + 1} of {self.total_splits} '
                    f'[{split_indices[current_split]} to {split_indices[current_split + 1] - 1}] ...'
                )

                for row in csv_reader:
                    if is_idx_within_split(idx, current_split, split_indices, self.total_splits):
                        current_subject_fpath = row[0]
                        try:
                            start = time.time()
                            print(f"Volume #{idx+1} | Loading {current_subject_fpath} ...")
                            # Load image (robustly rescaled) and compute its mask
                            zoom, volume, mask = load_image_masked(current_subject_fpath,
                                                                   im_threshold=self.im_threshold)
                            subject_basename = os.path.basename(current_subject_fpath)

                            # Loop over each permutation of axes
                            for perm in perms:
                                # Reorder volume & mask
                                volume_perm = np.transpose(volume, perm)
                                mask_perm   = np.transpose(mask, perm)

                                # Reorder voxel sizes to match the perm axes
                                zoom_perm = [zoom[perm[0]], zoom[perm[1]], zoom[perm[2]]]

                                # Extract patches with shape (P1,P2,P3)
                                patches = extract_3d_patches(
                                    volume_perm,
                                    mask_perm,
                                    patch_size=patch_size,
                                    stride=stride
                                )
                                n_patches = patches.shape[0]
                                if n_patches == 0:
                                    continue

                                # For snapshot, randomly sample up to 16 patches
                                n_samples = min(n_patches, 16)
                                sample_indices = np.random.choice(n_patches, size=n_samples, replace=False)
                                sampled_patches = patches[sample_indices]
                                # For visualization, take the middle slice along the 3rd dimension
                                patch_slices = [p[:, :, p.shape[2] // 2] for p in sampled_patches]

                                # Build a figure
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

                                # Create a unique snapshot name for each permutation
                                perm_suffix = "".join(str(x) for x in perm)
                                snap_fname = os.path.join(
                                    snapshot_dir,
                                    subject_basename + f"_perm{perm_suffix}.png"
                                )
                                plt.savefig(snap_fname)
                                plt.close(fig)
                                print(f"Saved snapshot for {subject_basename} (perm={perm_suffix}) -> {snap_fname}")

                                # Append to the global arrays
                                all_patches = np.concatenate([all_patches, patches], axis=0)
                                all_zooms_x += [zoom_perm[0]] * n_patches
                                all_zooms_y += [zoom_perm[1]] * n_patches
                                all_zooms_z += [zoom_perm[2]] * n_patches

                                # Mark subject with permutation
                                subject_name_with_perm = (subject_basename + f"_p{perm_suffix}").encode("ascii","ignore")
                                subjects += [subject_name_with_perm] * n_patches

                            end = time.time() - start
                            print(f" -> Extracted patches (all perms) in {end:.3f}s")

                        except Exception as e:
                            print(f"Volume #{idx+1} -> Failed reading data. Error: {e}")
                            pass
                    idx += 1

            # Determine output file name based on splits
            current_hdf5_fname = os.path.join(self.dataset_path, self.dataset_name)
            current_hdf5_fname = get_new_fname(current_hdf5_fname, current_split, self.total_splits)
            
            print(f'Saving dataset into: {current_hdf5_fname} ...')
            with h5py.File(current_hdf5_fname, "w") as hf:
                hf.create_dataset('orig_dataset_imgs', data=all_patches, compression='gzip')
                hf.create_dataset("orig_zooms_x", data=np.array(all_zooms_x), compression='gzip')
                hf.create_dataset("orig_zooms_y", data=np.array(all_zooms_y), compression='gzip')
                hf.create_dataset("orig_zooms_z", data=np.array(all_zooms_z), compression='gzip')
                dt = h5py.special_dtype(vlen=str)
                hf.create_dataset("subject", data=subjects, dtype=dt, compression="gzip")

            end_d = time.time() - start_d
            print(f"Successfully written {current_hdf5_fname} in {end_d:.3f} seconds.")


# ------------------------------------------------------------------------
# 5) Command-line / main entry
# ------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='HDF5 Creation for 3D Patches (Multi-Axis Permutations), Symmetric Padding, Random Filtering, Robust Rescaling, and Snapshot Visualization'
    )
    parser.add_argument('--hdf5_name', type=str, default=None,
                        help='Name of output HDF5 dataset (used if a single CSV file is provided).')
    parser.add_argument('--hdf5_path', type=str,
                        default="/data/dadmah/gonwal2/Documents/SuperResolution/datasets/RAVEN_LDM_3D_v4_aniso/",
                        help='Folder to save the HDF5 dataset(s).')
    parser.add_argument('--csv_file_images', type=str,
                        default="/data/dadmah/gonwal2/Documents/datasets/Datasets_raw/ALL_pp1_datasplit_csvs/",
                        help="Either a CSV file listing subjects or a directory containing CSV files "
                             "with names ending in _train.csv, _val.csv, or _test.csv.")
    parser.add_argument('--patch_size', nargs=3, type=int, default=[192, 192, 32],
                        help="Size of the 3D patch: [P1, P2, P3].")
    parser.add_argument('--stride', nargs=3, type=int, default=[192, 192, 32],
                        help="Stride of the sliding window: [S1, S2, S3].")
    parser.add_argument('--intensity_threshold', type=float, default=15,
                        help="Intensity threshold for masking the images (foreground > threshold).")
    parser.add_argument('--total_splits', type=int, default=1,
                        help="Number of HDF5 files to split into (for memory constraints).")
    parser.add_argument('--suffix', type=str, default="v4_aniso",
                        help="Optional suffix to add to the output HDF5 filename. If provided, an underscore "
                             "and the suffix will be appended before the .hdf5 extension.")

    args = parser.parse_args()

    # If a directory is provided for CSV files:
    if os.path.isdir(args.csv_file_images):
        csv_dir = args.csv_file_images
        csv_files = [f for f in os.listdir(csv_dir)
                     if f.endswith('_train.csv') or f.endswith('_val.csv') or f.endswith('_test.csv')]
        if not csv_files:
            print("No CSV files with the required pattern (_train.csv, _val.csv, _test.csv) were found in the directory.")
            exit(1)
        for csv_file in csv_files:
            full_csv_path = os.path.join(csv_dir, csv_file)
            # Determine subfolder based on file ending.
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
            # Create the HDF5 file name based on the CSV filename.
            base_name = os.path.splitext(csv_file)[0]
            if args.suffix is not None:
                base_name = base_name + "_" + args.suffix
            dataset_name = base_name + '.hdf5'
            hdf5_full_path = os.path.join(output_dir, dataset_name)
            print(f"Processing CSV file: {full_csv_path} into folder: {output_dir}")
            info = {
                "dataset_name": dataset_name,
                "dataset_path": output_dir,
                "csv_file": full_csv_path,
                "im_threshold": args.intensity_threshold,
                "total_splits": args.total_splits
            }
            dataset_obj = PopulationDataset(info)
            dataset_obj.create_hdf5_dataset(
                patch_size=tuple(args.patch_size),
                stride=tuple(args.stride)
            )
    else:
        # Single CSV file mode.
        output_dir = args.hdf5_path
        os.makedirs(output_dir, exist_ok=True)
        # If a suffix is provided, insert it into the provided hdf5_name.
        base_name, ext = os.path.splitext(args.hdf5_name)
        if args.suffix is not None:
            base_name = base_name + "_" + args.suffix
        dataset_name = base_name + ext
        hdf5_full_path = os.path.join(output_dir, dataset_name)
        if os.path.exists(hdf5_full_path):
            print(f"HDF5 file {hdf5_full_path} already exists. Skipping creation.")
        else:
            info = {
                "dataset_name": dataset_name,
                "dataset_path": output_dir,
                "csv_file": args.csv_file_images,
                "im_threshold": args.intensity_threshold,
                "total_splits": args.total_splits
            }
            dataset_obj = PopulationDataset(info)
            dataset_obj.create_hdf5_dataset(
                patch_size=tuple(args.patch_size),
                stride=tuple(args.stride)
            )
