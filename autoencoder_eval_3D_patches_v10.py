# -*- coding: utf-8 -*-
import os

# Set KMP_DUPLICATE_LIB_OK early if needed for matplotlib/torch issues
# WARNING: This can mask underlying library conflicts. Use with caution.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset # Import Dataset base class
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
import inspect # Needed for checking model signature
from omegaconf import OmegaConf # Needed by loading wizard likely (pip install omegaconf)
import math # For pi in Hann window

# --- Taming Transformer imports ---
# (Keep existing imports)
try:
    # Make sure the taming library is installed and accessible
    from taming.models.autoencoders import AutoencoderKL3D, VQModel3D, AutoencoderKL3DFiLM_BiCond
except ImportError as e:
    print(f"ERROR: Could not import Taming models: {e}")
    print("Please ensure 'taming' library (e.g., taming-transformers-rom1504) is installed and accessible.")
    sys.exit(1)

# --- Local imports ---
# (Keep existing imports, ensuring they exist and are correct)
try:
    # Ensure these helper scripts are in the Python path or the same directory
    from data_loader.load_neuroimaging_data_final import load_and_rescale_image_sitk_v3, sitk2sitk, OrigDataThickPatches
    from utils_v2 import gzip_this, is_anisotropic, arguments_setup_ae, filename_wizard, model_loading_wizard_ae_v2, load_model_from_ckpt
except ImportError as e:
    print(f"ERROR: Could not import local modules/functions: {e}")
    print("Check paths for 'data_loader/load_neuroimaging_data_final.py' and 'utils_v2.py'. Ensure they are accessible.")
    print("Also ensure all functions they rely on (e.g., conform_itk) are available.")
    sys.exit(1)
except FileNotFoundError as e:
    print(f"ERROR: Could not find local module file: {e}")
    print("Check paths for 'data_loader/load_neuroimaging_data_final.py' and 'utils_v2.py'. Ensure they are accessible.")
    sys.exit(1)


# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger("eval_ae_3d")

# --- Change CWD (Optional but often helpful) ---
try:
    # Get the directory where the script is located
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    logger.info(f"Changing working directory to: {dname}")
    os.chdir(dname)
except NameError: # __file__ might not be defined if run interactively
    logger.warning("__file__ not defined. Keeping current working directory.")
    dname = os.getcwd()

HELPTEXT = "3D Patch-based Autoencoder Evaluation Script with FiLM support, selectable patch merging, and optional multi-orientation averaging."

# --- Helper Function (Window Weights - No changes needed here) ---
def get_window_weights(patch_shape, window_type='triangular'):
    """
    Computes a 3D weighting mask for a given patch shape using the specified window type.
    These windows often lead to smoother merging than Gaussian windows,
    especially if the stride is approximately half the patch size.

    Args:
        patch_shape (tuple): Patch dimensions (D, H, W).
        window_type (str): Type of window ('triangular' or 'hann'). Default: 'triangular'.

    Returns:
        torch.Tensor: A 3D tensor (D, H, W) containing the window weights.
    """
    if not isinstance(patch_shape, (tuple, list)) or len(patch_shape) != 3:
        raise ValueError(f"patch_shape must be a sequence of 3 dimensions, got {patch_shape}")
    if window_type not in ['triangular', 'hann']:
        raise ValueError(f"Unsupported window_type '{window_type}'. Choose 'triangular' or 'hann'.")

    dims = patch_shape # Should be (Depth, Height, Width)
    weights_1d = []

    for i, dim_size in enumerate(dims):
        if dim_size <= 1: # Handle single-voxel dimension
            weight_1d = torch.ones(dim_size, dtype=torch.float32)
        else:
            coords = torch.arange(dim_size, dtype=torch.float32)
            if window_type == 'triangular':
                # Linear decay from center (1.0) to edges (0.0)
                center = (dim_size - 1.0) / 2.0
                dist_from_center = torch.abs(coords - center)
                half_size = dim_size / 2.0
                weight_1d = torch.clamp(1.0 - dist_from_center / half_size, min=0.0)
            elif window_type == 'hann':
                # Cosine-based window, zero at edges, one at center
                weight_1d = 0.5 * (1.0 - torch.cos(2.0 * math.pi * coords / (dim_size - 1.0)))

        weights_1d.append(weight_1d)

    # Combine 1D weights using outer product simulation: w(d,h,w) = w(d) * w(h) * w(w)
    # Expand dimensions for broadcasting: (D,1,1) * (1,H,1) * (1,1,W) -> (D,H,W)
    weight_3d = weights_1d[0][:, None, None] * weights_1d[1][None, :, None] * weights_1d[2][None, None, :]

    # Normalize to have a max of 1 (important for consistent blending)
    max_val = torch.max(weight_3d)
    if max_val > 1e-8: # Avoid division by zero
        weight_3d = weight_3d / max_val
    else:
        logger.warning(f"Generated {window_type} weight mask has maximum value close to zero for shape {patch_shape}.")

    return weight_3d


# --- Argument Parsing ---
def options_parse():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description=HELPTEXT, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Basic I/O options
    parser.add_argument('--in_name', type=str, default="/data/dadmah/gonwal2/Documents/SuperResolution/LUSENBRINK_001_T1w_pp0_downsample-z3.nii.gz",
                        help="Path to the single input 3D volume (REQUIRED if --csv_file not used).")
    parser.add_argument('--csv_file', type=str, default=None,
                        help="CSV file with one volume path per line (optional, overrides --in_name if provided). NOTE: CSV processing is NOT implemented.")
    parser.add_argument('--out_name', type=str, default="/data/dadmah/gonwal2/Documents/SuperResolution/Lusenbrink_smore_z2/LUSENBRINK_001_T1w_pp0_downsample-z3/LUSENBRINK_001_T1w_pp0_downsample-z3_RAVEN.nii.gz",
                        help="Path for the single output volume. If None or using CSV, derived from input name/dest_dir.")
    parser.add_argument('--iname_new', type=str, default=None,
                        help="If --save_new_input set, the preprocessed image is saved here. If None, derived.")
    parser.add_argument('--dest_dir', type=str, default=None,
                        help="Output directory for batch processing (from CSV) or if --out_name is relative.")
    parser.add_argument('--suffix', type=str, default=None,
                        help="Suffix to append to output filenames if --out_name not set.")
    parser.add_argument('--ext', default=None,
                        help="Output file extension (e.g., '.nii.gz'). If None, input extension is used.")
    parser.add_argument('--save_new_input', action='store_true', default=True,
                        help="Save the preprocessed (resampled before normalization) volume if set.")
    parser.add_argument('--skip_existing', action='store_true', default=False,
                        help="Skip processing if the output file already exists.")

    # Preprocessing options
    parser.add_argument('--intensity_range_mode', type=int, default=2, choices=[0, 1, 2],
                        help="Output intensity range: 0=[0-255], 1=[0-1], 2=Original Min-Max.")
    parser.add_argument('--robust_rescale_input', action='store_true', default=True,
                        help="Use robust histogram rescaling during preprocessing if set (requires implementation in load_and_rescale_image_sitk_v2).")
    parser.add_argument('--uf_w', type=float, default=1, help="Upscaling factor in width (X) dimension.")
    parser.add_argument('--uf_h', type=float, default=1, help="Upscaling factor in height (Y) dimension.")
    parser.add_argument('--uf_z', type=float, default=3, help="Upscaling factor in depth (Z) dimension.")
    parser.add_argument('--order', type=int, default=1,
                        help="Order of interpolation for resampling (0=Nearest, 1=Linear, 3=Cubic B-Spline). Default: 1, otherwise there is a high risk of overshoot/undershoot artifacts")
    parser.add_argument('--use_scipy', action='store_true', default=False,
                        help="Use scipy.ndimage zoom for resampling instead of SimpleITK if set (requires implementation in load_and_rescale_image_sitk_v2).")

    # Network / Inference options
    parser.add_argument('--name', type=str, default="raven_f2_nfg64_nfd64_dw1e-1_lpw2.0_l1w0.0_lrgtod0.5_3dlpips_AEv5_gan_finetune",
                        help="Model identifier (base name for checkpoint/config).",
                        # List of known/tested model configurations
                        choices=['klgan_neuro_3D_deg2gt', 'vqgan_neuro_3D_deg2gt',
                                 "klgan_neuro_3D_deg2gt_v2", "klgan_neuro_3D_deg2gt_v3",
                                 "klgan_neuro_3D_deg2gt_v4", "klgan_neuro_3D_deg2gt_v2p1",
                                 "klgan_neuro_3D_deg2gt_v2p1_scratch", "klgan_neuro_3D_deg2gt_v3_attn",
                                 "klgan_neuro_3D_deg2gt_v5", 'klgan_neuro_3D_deg2gt_v6_aniso',
                                 "klgan_neuro_3D_deg2gt_v6_3_xl2", "klgan_neuro_3D_deg2gt_v3_3_small",
                                 "klgan_neuro_3D_deg2gt_v3p4_aw1e4_GtoDLR100",
                                 "klgan_neuro_3D_deg2gt_v3p4_aw1e4_GtoDLR010",
                                 "klgan_neuro_3D_deg2gt_v3p4_aw1e4_GtoDLR001",
                                 "klgan_neuro_3D_deg2gt_v3p4_aw1e3_GtoDLR100",
                                 "klgan_neuro_3D_deg2gt_v3p4_aw1e3_GtoDLR010",
                                 "klgan_neuro_3D_deg2gt_v3p4_aw1e3_GtoDLR001",
                                 "klgan_neuro_3D_deg2gt_v3p4_aw1e2_GtoDLR100",
                                 "klgan_neuro_3D_deg2gt_v3p4_aw1e2_GtoDLR010",
                                 "klgan_neuro_3D_deg2gt_v3p4_aw1e2_GtoDLR001",
                                 "klgan_neuro_3D_deg2gt_v3p4_aw1e1_GtoDLR100",
                                 "klgan_neuro_3D_deg2gt_v3p4_aw1e1_GtoDLR010",
                                 "klgan_neuro_3D_deg2gt_v3p4_aw1e1_GtoDLR001",
                                 "custom_klgan_neuro_3D_deg2gt_v3p4_aw1e-1_GtoDLR1e0_FiLM_aniso",
                                 "custom_klgan_neuro_3D_deg2gt_v3p4_aw1e-1_GtoDLR1e0_aniso",
                                 "raven_nfg32_nfd64_dw5e0_lpw2.0_l1w0.0_lrgtod0.05",
                                 "raven_nfg32_nfd64_dw5e0_lpw2.0_l1w0.0_lrgtod0.5",
                                 "raven_nfg64_nfd64_dw5e0_lpw2.0_l1w0.0_lrgtod0.05",
                                 "raven_nfg64_nfd64_dw5e0_lpw2.0_l1w0.0_lrgtod0.5",
                                 "raven_nfg94_nfd64_dw5e0_lpw2.0_l1w0.0_lrgtod0.05",
                                 "raven_nfg94_nfd64_dw5e0_lpw2.0_l1w0.0_lrgtod0.5",
                                 "raven_nfg64_nfd64_dw5e0_lpw2.0_l1w0.0_lrgtod0.5_finetune",
                                 "raven_nfg96_nfd64_dw5e0_lpw2.0_l1w0.0_lrgtod0.5",
                                 "raven_nfg64_nfd64_dw5e0_lpw2.0_l1w0.0_lrgtod0.05_3dlpips",
                                 "raven_nodegrade_nfg64_nfd64_dw5e0_lpw2.0_l1w0.0_lrgtod0.05_3dlpips",
                                 "raven_f2_nfg64_nfd64_dw5e0_lpw2.0_l1w0.0_lrgtod0.05_3dlpips_v2",
                                 "raven_f2_nfg64_nfd64_dw5e0_lpw2.0_l1w0.0_lrgtod0.05_3dlpips_AEv5",
                                 "raven_f2_nodegrade_nfg64_nfd64_dw5e0_lpw2.0_l1w0.0_lrgtod0.05_3dlpips_AEv5",
                                 "raven_f2_nodegrade_nfg64_nfd64_dw5e0_lpw2.0_l1w0.0_lrgtod0.5_3dlpips_AEv5",
                                 "raven_f2_nfg64_nfd64_dw1e-1_lpw2.0_l1w0.0_lrgtod0.5_3dlpips_AEv5_gan_finetune",
                                 "raven_f2_nfg64_nfd64_dw1e-1_adw0.3_lpw1.0_l1w1.0_lrgtod0.5_frg1_frd1_3dlpips_AEv5",
                                 "raven_f2_nfg64_nfd64_dw1e-1_adw0.3_lpw1.0_l1w1.0_lrgtod0.5_frg1_frd1_3dlpips_AEv6_consistency"
                                 ],)
    parser.add_argument('--model_path', type=str, default=None,
                        help="Optional directory path containing model checkpoints.")
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help="Use automatic mixed precision (AMP) for inference if set.")
    parser.add_argument('--use_ema', action='store_true', default=False,
                        help="Attempt to use EMA model weights (Requires proper EMA implementation/loading in model_loading_wizard_ae_v2).")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help="Disable CUDA even if available.")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size for processing patches.")

    # Patch parameters - Expect comma-separated strings
    parser.add_argument('--max_patch_size', type=str, default="160,32,160",
                        help="Input patch size (D,H,W) as comma-separated integers (e.g., '128,128,32'). Must be divisible by network downsampling factor.")
    parser.add_argument('--stride', type=str, default="80,16,80", # Default stride is half patch size for good soft merging
                        help="Stride for patch extraction (D,H,W) as comma-separated integers. For seamless 'soft' merging with Hann/Triangular windows, set stride <= patch_size / 2 (IDEAL: stride = patch_size / 2).")
    parser.add_argument('--merging_method', type=str, default="soft", choices=["average", "soft"],
                        help="Patch merging method: 'average' (uniform weight=1) or 'soft' (weighted using selected window).")
    parser.add_argument('--window_type', type=str, default="triangular", choices=["triangular", "hann"],
                        help="Window function type to use for 'soft' merging.")

    # Multi-orientation and post-processing
    parser.add_argument('--multi_orientation_avg', action='store_true', default=False,
                        help="If set, process the volume in three orientations (original D,H,W; rotated H,D,W; rotated W,H,D) and average the results.")
    # --- MODIFICATION START ---
    parser.add_argument('--aggregate_on_gpu', action='store_true', default=False,
                        help="For multi-orientation, aggregate results on the GPU. Default behavior is to use the CPU to conserve VRAM, which is slower but safer for large volumes.")
    # --- MODIFICATION END ---
    parser.add_argument('--net2out_rescaling', default="clamp", choices=["clamp", "linear"],
                        help="The raw output of the network is expected to be within [-1, 1], but in reality it can have any values slightly beyond this range. The output needs to be rescaled to the original intensity range and values outside this range need to be handled. ")
    parser.add_argument('--histogram_matching_mode', type=str, default='none',
                        choices=['nonlinear', 'linear', 'none'],
                        help="""Method for matching the output image's histogram to the interpolated input's.
                                 'nonlinear': Use piecewise non-linear matching (default).
                                 'linear': Match the mean and standard deviation.
                                 'none': Do not perform any histogram matching.""")


    # Timing output
    parser.add_argument('--proc_time_txt', type=str, default=None,
                        help="Full .txt path to store model processing time (Total - SavingPreprocessed).")
    parser.add_argument('--trilinear_proc_time_txt', type=str, default=None,
                        help="Full .txt path to store preprocessing time (Resampling + SavingPreprocessed).")

    args = parser.parse_args()

    # --- Argument Validation and Post-Processing ---
    if not args.csv_file and not args.in_name:
        parser.error("Either --in_name or --csv_file must be provided.")
    if args.csv_file:
        logger.error("--csv_file mode is not implemented in this version. Use --in_name for single file processing.")
        sys.exit(1)

    # Parse tuple arguments from strings
    try:
        args.max_patch_size = tuple(map(int, args.max_patch_size.split(',')))
        if len(args.max_patch_size) != 3: raise ValueError("Must have 3 dimensions (D, H, W)")
    except Exception as e:
        parser.error(f"--max_patch_size must be 3 comma-separated integers (e.g., '128,128,32'): {e}")
    try:
        args.stride = tuple(map(int, args.stride.split(',')))
        if len(args.stride) != 3: raise ValueError("Must have 3 dimensions (D, H, W)")
    except Exception as e:
        parser.error(f"--stride must be 3 comma-separated integers (e.g., '96,96,32'): {e}")

    # --- Validate patch/stride dimensions and provide feedback ---
    for i in range(3):
        patch_dim = args.max_patch_size[i]
        stride_dim = args.stride[i]
        if patch_dim <= 0 or stride_dim <= 0:
            parser.error(f"Patch size and stride dimensions must be positive. Got patch={args.max_patch_size}, stride={args.stride}")
        if stride_dim > patch_dim:
            logger.warning(f"WARNING: Stride dim {i} ({stride_dim}) > Patch size dim {i} ({patch_dim}). This WILL leave gaps in the reconstruction!")
        # Check for ideal overlap specifically for soft merging
        elif args.merging_method == "soft":
            ideal_stride = patch_dim / 2.0
            if abs(stride_dim - ideal_stride) < 1e-6: # Check if equal using tolerance
                logger.info(f"Stride dim {i} ({stride_dim}) == Patch size dim {i} / 2 ({ideal_stride:.1f}). Ideal overlap for seamless soft ({args.window_type}) merging.")
            elif stride_dim > ideal_stride:
                logger.warning(f"WARNING: Soft ({args.window_type}) merging selected, but Stride dim {i} ({stride_dim}) > Patch size dim {i} / 2 ({ideal_stride:.1f}). "
                               f"The sum of overlapping weights may not be constant, potentially causing **SEAM ARTIFACTS**. Consider stride <= patch_size / 2 (e.g., {int(math.floor(ideal_stride))}) for smoother results.")
            else: # stride_dim < ideal_stride
                logger.info(f"Stride dim {i} ({stride_dim}) < Patch size dim {i} / 2 ({ideal_stride:.1f}). "
                            f"Provides more overlap than strictly necessary for {args.window_type} window, which is generally fine but less efficient.")
        # Check for average merging (no specific overlap needed beyond stride <= patch_size)
        elif args.merging_method == "average" and stride_dim < patch_dim:
            logger.info(f"Average merging: Stride dim {i} ({stride_dim}) <= Patch size dim {i} ({patch_dim}). Overlap exists.")


    # Call the arguments setup helper (e.g., to determine output filenames)
    try:
        logger.info("Running arguments_setup_ae from utils...")
        args = arguments_setup_ae(args) # This might modify args like suffix, out_name etc.
        logger.info("Finished arguments_setup_ae.")
    except NameError:
        logger.error("Function 'arguments_setup_ae' not found in utils. Ensure it's defined/imported correctly.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during arguments_setup_ae: {e}")
        sys.exit(1)

    # Ensure CUDA is set up
    if not args.no_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        try:
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            logger.warning(f"Could not get CUDA device name: {e}. Still attempting to use CUDA.")
    else:
        args.device = torch.device("cpu")
        logger.info("Using CPU.")
        if not args.no_cuda and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU.")

    return args

# --- Patch Inference Function ---
def run_autoencoder_3d(img_filename: str,
                       orig_data_normalized_dhw: torch.Tensor, # Assuming np.ndarray is converted to tensor before this func
                       params_model: dict,
                       model: nn.Module,
                       logger: logging.Logger,
                       args: argparse.Namespace, # Assuming args is from argparse
                       is_film_model: bool,
                       source_zooms_list_whd: list | None,
                       target_zooms_list_whd: list | None,
                       patch_shape_dhw: tuple,
                       stride_dhw: tuple):
    """
    Processes the entire 3D volume with patches using sliding window, supporting FiLM models.
    Includes robust weighted averaging and improved AMP handling.

    Args:
        img_filename (str): Input filename (for dataset logging).
        orig_data_normalized_dhw (torch.Tensor): Normalized input data (shape D, H, W or Z, Y, X for the current orientation).
                                                 Assumed to be a PyTorch tensor on the correct device or CPU.
        params_model (dict): Dictionary with model parameters like device, batch_size.
        model (torch.nn.Module): The loaded autoencoder model.
        logger (logging.Logger): Logger instance.
        args (argparse.Namespace): Parsed command-line arguments (used for device, batch_size, merging, window, amp).
        is_film_model (bool): Flag indicating if the model uses FiLM.
        source_zooms_list_whd (list or None): Calculated source voxel size [W, H, D] corresponding to the CURRENT orientation.
        target_zooms_list_whd (list or None): Calculated target voxel size [W, H, D] corresponding to the CURRENT orientation.
        patch_shape_dhw (tuple): Patch size (D, H, W) for this processing run.
        stride_dhw (tuple): Stride (D, H, W) for this processing run.

    Returns:
        torch.Tensor: The reconstructed output volume tensor on the specified device.
    """
    # Assumes input numpy array is D, H, W (according to the current orientation)
    orig_D, orig_H, orig_W = orig_data_normalized_dhw.shape
    logger.info(f"Processing volume shape D,H,W: {orig_D}x{orig_H}x{orig_W}")
    logger.info(f"Using patch size D,H,W: {patch_shape_dhw}")
    logger.info(f"Using stride D,H,W: {stride_dhw}")

    # Create dataset (handles padding internally based on patch_shape and stride)
    try:
        dataset = OrigDataThickPatches(img_filename, orig_data_normalized_dhw, # Pass tensor directly
                                       max_patch_size=patch_shape_dhw,
                                       stride=stride_dhw,
                                       transforms=None)
    except Exception as e:
        logger.error(f"Error creating OrigDataThickPatches dataset: {e}")
        raise

    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0, pin_memory=(args.device.type == 'cuda'))

    device = args.device
    patch_size_d, patch_size_h, patch_size_w = patch_shape_dhw

    padded_shape = dataset.volume.shape # Should be (padded_D, padded_H, padded_W)
    logger.info(f"Padded volume shape D,H,W for processing: {padded_shape}")
    output_accum = torch.zeros(padded_shape, dtype=torch.float32, device=device)
    count_accum = torch.zeros(padded_shape, dtype=torch.float32, device=device)

    weight_mask = None
    if args.merging_method == "soft":
        logger.info(f"Using soft ({args.window_type} weighted) patch merging.")
        weight_mask = get_window_weights(patch_shape_dhw, window_type=args.window_type).to(device)
    else: # average merging
        logger.info("Using average (uniform weight=1) patch merging.")
        weight_mask = torch.ones(patch_shape_dhw, dtype=torch.float32, device=device)

    src_z_tensor = None
    tgt_z_tensor = None
    if is_film_model:
        if source_zooms_list_whd is not None:
            if len(source_zooms_list_whd) != 3:
                logger.error(f"Invalid source zooms list provided for FiLM (length != 3): {source_zooms_list_whd}")
                raise ValueError("Invalid source zoom information for FiLM model.")
            src_z_tensor = torch.tensor(source_zooms_list_whd, dtype=torch.float32, device=device).unsqueeze(0)
            logger.info(f"Using source voxel size [W,H,D] (tensor shape {src_z_tensor.shape}): {source_zooms_list_whd}")
        else:
            logger.error("FiLM model requires source zoom information, but it was None.")
            raise ValueError("Missing source zoom information for FiLM model.")

        if target_zooms_list_whd is not None:
            if len(target_zooms_list_whd) != 3:
                logger.error(f"Invalid target zooms list provided for FiLM (length != 3): {target_zooms_list_whd}")
                raise ValueError("Invalid target zoom information for FiLM model.")
            tgt_z_tensor = torch.tensor(target_zooms_list_whd, dtype=torch.float32, device=device).unsqueeze(0)
            logger.info(f"Using target voxel size [W,H,D] (tensor shape {tgt_z_tensor.shape}): {target_zooms_list_whd}")
        else:
            logger.error("FiLM model requires target zoom information, but it was None.")
            raise ValueError("Missing target zoom information for FiLM model.")

    model_to_inspect = model.module if hasattr(model, 'module') else model
    needs_source_cond = False
    needs_target_cond = False
    try:
        if not hasattr(model_to_inspect, 'forward'):
            raise AttributeError("Loaded model object has no 'forward' method.")
        sig = inspect.signature(model_to_inspect.forward)
        needs_source_cond = "source_cond_input" in sig.parameters
        needs_target_cond = "target_cond_input" in sig.parameters
    except AttributeError as e:
        logger.warning(f"Could not inspect model.forward signature: {e}. Assuming FiLM needs based on is_film_model flag.")
        needs_source_cond = is_film_model
        needs_target_cond = is_film_model
    except Exception as e:
        logger.error(f"Unexpected error during model signature inspection: {e}")
        raise

    # --- Inference Loop ---
    model.eval()
    # Variable to ensure one-time logging for AMP dtype
    amp_dtype_logged = False
    with torch.no_grad():
        pbar = tqdm(total=len(data_loader), desc="Processing patches", file=sys.stdout, ncols=100, leave=False)
        for i, batch in enumerate(data_loader):
            try:
                patches = batch["patch"].to(device) # Expected shape [B, 1, pD, pH, pW], should be float32 from DataLoader
                d_batch = batch["indices"][0].int()
                h_batch = batch["indices"][1].int()
                w_batch = batch["indices"][2].int()
                current_batch_size = patches.shape[0]

                forward_kwargs = {}
                if is_film_model:
                    if needs_source_cond and src_z_tensor is not None:
                        forward_kwargs["source_cond_input"] = src_z_tensor.expand(current_batch_size, -1)
                    if needs_target_cond and tgt_z_tensor is not None:
                        forward_kwargs["target_cond_input"] = tgt_z_tensor.expand(current_batch_size, -1)

                if args.use_amp:
                    amp_dtype_to_use = None
                    if args.device.type == 'cuda':
                        if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                            amp_dtype_to_use = torch.bfloat16
                            if not amp_dtype_logged: logger.info("AMP on CUDA: bfloat16 is supported and will be used.")
                        else:
                            amp_dtype_to_use = torch.float16
                            if not amp_dtype_logged: logger.info("AMP on CUDA: bfloat16 not supported, falling back to float16. This might differ from your bf16 training.")
                    elif args.device.type == 'cpu':
                        amp_dtype_to_use = torch.bfloat16 # PyTorch >=1.10 for bfloat16 CPU AMP
                        if not amp_dtype_logged: logger.info("AMP on CPU: Attempting to use bfloat16. Behavior depends on PyTorch version and backend.")

                    if not amp_dtype_logged: # Ensure this logs only once per run_autoencoder_3d call
                        amp_dtype_logged = True

                    with torch.amp.autocast(device_type=args.device.type, dtype=amp_dtype_to_use, enabled=True):
                        # `patches` (assumed float32 here) will be handled by autocast for model ops.
                        # No explicit patches.half() or patches.bfloat16() here.

                        # Optional: Check for NaNs in input patches before model call
                        # if torch.isnan(patches).any():
                        #     logger.warning(f"NaNs detected in input patches *before* model call in AMP context (batch {i}).")

                        preds, *_ = model(patches, **forward_kwargs)

                        # Optional: Check for NaNs in predictions *inside* AMP context (preds will be lower precision here)
                        # if torch.isnan(preds).any():
                        #     logger.warning(f"NaNs detected in model output (preds) *inside* AMP context (batch {i}). Patch indices D:{d_batch[0]}, H:{h_batch[0]}, W:{w_batch[0]}")
                else:
                    preds, *_ = model(patches, **forward_kwargs) # Standard float32 forward pass

                preds = preds.float() # Cast output to float32 for accumulation (CRITICAL)

                if torch.isnan(preds).any():
                    logger.error(f"NaNs detected in predictions *after* casting to float32 (batch {i}). This means NaNs were generated by the model. Patch indices D:{d_batch[0]}, H:{h_batch[0]}, W:{w_batch[0]}")
                    # Optionally, raise an error or skip this batch's accumulation
                    # For now, we continue and accumulate potential NaNs, which will propagate.
                    # Consider: continue

                if preds.shape[0] != current_batch_size or preds.shape[1] != 1 or preds.shape[2:] != patch_shape_dhw:
                    logger.warning(f"Batch {i}: Unexpected model output shape {preds.shape}. Expected ({current_batch_size}, 1, {patch_shape_dhw}). Skipping accumulation for this batch.")
                    pbar.update(1)
                    continue

                for b in range(current_batch_size):
                    d_start, h_start, w_start = int(d_batch[b]), int(h_batch[b]), int(w_batch[b])
                    slice_d = slice(d_start, d_start + patch_size_d)
                    slice_h = slice(h_start, h_start + patch_size_h)
                    slice_w = slice(w_start, w_start + patch_size_w)
                    patch_pred = preds[b, 0]
                    output_accum[slice_d, slice_h, slice_w] += patch_pred * weight_mask
                    count_accum[slice_d, slice_h, slice_w] += weight_mask

            except Exception as e:
                logger.exception(f"Error during processing batch {i}: {e}")
                try:
                    logger.error(f"Problematic Patch Indices (start D,H,W): D={d_batch.tolist()}, H={h_batch.tolist()}, W={w_batch.tolist()}")
                except NameError:
                    logger.error("Could not retrieve problematic patch indices.")
                logger.warning(f"Skipping batch {i} due to error. Final result might be incomplete/incorrect.")
                # If a critical error like NaN propagation is a concern, you might re-raise here:
                # raise e
                pbar.update(1)
                continue

            # --- MODIFICATION: Update pbar with GPU memory stats ---
            if args.device.type == 'cuda':
                # Get memory in GB
                allocated_mem = torch.cuda.memory_allocated(0) / (1024**3)
                reserved_mem = torch.cuda.memory_reserved(0) / (1024**3)
                # Create a postfix dictionary and update the bar
                pbar.set_postfix(GPU_GB=f'{allocated_mem:.2f}A/{reserved_mem:.2f}R', refresh=False)
            # --- END MODIFICATION ---

            pbar.update(1)
        pbar.close()

    logger.info("Finalizing output volume by dividing accumulated values by accumulated weights...")
    border_d, border_h, border_w = patch_size_d // 2, patch_size_h // 2, patch_size_w // 2
    if padded_shape[0] > border_d * 2 + 1 and padded_shape[1] > border_h * 2 + 1 and padded_shape[2] > border_w * 2 + 1:
        center_counts = count_accum[border_d:-border_d, border_h:-border_h, border_w:-border_w]
        if center_counts.numel() > 0:
            min_count, max_count = torch.min(center_counts), torch.max(center_counts)
            if max_count > 1e-6:
                ratio = max_count / torch.clamp(min_count, min=1e-8)
                logger.info(f"Weight sum (count_accum) in central region - Min: {min_count:.4f}, Max: {max_count:.4f}, Ratio Max/Min: {ratio:.4f}")
                if ratio > 1.05:
                    logger.warning("Significant variation detected in the sum of weights within the central region. This might indicate non-ideal stride/patch/window settings leading to potential merging artifacts.")
            else:
                logger.warning("Weight sum (count_accum) is near zero in the central region. Check stride, patch size, and padding.")
        else:
            logger.warning("Central region for weight sum diagnostics is empty.")
    else:
        logger.warning("Volume (padded) too small to define a central region for weight sum diagnostics.")

    epsilon = 1e-8
    output_volume = output_accum / torch.clamp(count_accum, min=epsilon)
    output_volume[count_accum < epsilon] = 0.0 # Ensure regions with no contribution are zero

    # Check for NaNs in the final aggregated volume before cropping
    if torch.isnan(output_volume).any():
        num_nans = torch.isnan(output_volume).sum().item()
        logger.error(f"NaNs detected in the reconstructed output_volume *before* cropping. Count: {num_nans}/{output_volume.numel()}")
        # Depending on policy, you might want to raise an error here or attempt to fill NaNs
        # output_volume = torch.nan_to_num(output_volume, nan=0.0) # Example: replace NaNs with 0
        # logger.warning("NaNs were replaced with 0.0 in the output volume.")


    logger.info("Weighted averaging division complete.")
    output_volume = output_volume[:orig_D, :orig_H, :orig_W]
    logger.info(f"Cropped final volume back to original shape for this orientation: {output_volume.shape}")

    logger.info("Patch processing and merging complete for this orientation.")
    return output_volume

# --- Main Execution Logic ---
def main():
    """Main function to orchestrate loading, processing, and saving."""
    args = options_parse()
    logger.info(f"Starting evaluation with args:\n{vars(args)}")
    start_total = time.time()

    # --- Input File Handling ---
    if args.csv_file:
        logger.error("--csv_file mode is not implemented. Use --in_name instead.")
        sys.exit(1)

    img_filename = args.in_name
    if not img_filename or not Path(img_filename).exists():
        logger.error(f"Input file does not exist: {img_filename}")
        sys.exit(1)
    img_filename = str(Path(img_filename).resolve()) # Use absolute path

    # --- Output Filename Setup (Handled by arguments_setup_ae now) ---
    # Ensure output directory exists based on the potentially modified args.out_name
    out_path = Path(args.out_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output will be saved to: {args.out_name}")

    # Setup preprocessed output name if needed (also potentially modified by arguments_setup_ae)
    if args.save_new_input and args.iname_new:
          iname_new_path = Path(args.iname_new)
          iname_new_path.parent.mkdir(parents=True, exist_ok=True)
          logger.info(f"Preprocessed input will be saved to: {args.iname_new}")
    elif args.save_new_input:
          logger.warning("--save_new_input is True but --iname_new was not set or generated by arguments_setup_ae. Cannot save preprocessed input.")

    # --- Skip if Output Exists ---
    if args.skip_existing and Path(args.out_name).exists():
        logger.info(f"{args.out_name} already exists, skipping.")
        return # Exit cleanly

    # --- Preprocessing: Load and Rescale ---
    load_start = time.time()
    try:
        # This function should return numpy array (D,H,W), SITK image (Z,Y,X),
        # original SITK image (Z,Y,X), original max/min, and zooms [W,H,D]
        (
            interp_v8_np,      # Numpy array (D,H,W) after resampling & intensity scaling (if any)
            interp_sitk_v8,      # SimpleITK image corresponding to interp_v8_np (Z,Y,X)
            interp_sitk_native,  # SimpleITK image after resampling but *before* potential numpy intensity scaling (Z,Y,X)
            max_orig, min_orig,  # Min/Max intensity from the *original* input image before any processing
            source_zooms_whd,    # Calculated source voxel size [W, H, D] of original input
            target_zooms_whd,      # Calculated target voxel size [W, H, D] after resampling
            max_orig_final, min_orig_final
        ) = load_and_rescale_image_sitk_v3(
            img_filename, args, interpol=args.order, logger=logger, is_eval=True,
            intensity_rescaling=args.robust_rescale_input, uf_h=args.uf_h, uf_w=args.uf_w,
            uf_z=args.uf_z, use_scipy=args.use_scipy,
        )
        # Ensure zooms are standard Python lists or None, not tensors/arrays
        if isinstance(source_zooms_whd, (np.ndarray, torch.Tensor)):
            source_zooms_whd = source_zooms_whd.tolist()
        if isinstance(target_zooms_whd, (np.ndarray, torch.Tensor)):
            target_zooms_whd = target_zooms_whd.tolist()

    except FileNotFoundError:
        logger.error(f"Input file not found during load_and_rescale: {img_filename}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error during preprocessing/resampling: {e}")
        sys.exit(1)

    preprocess_duration = time.time() - load_start
    logger.info(f"Preprocessing (load & rescale) took {preprocess_duration:.2f}s")
    logger.info(f"Original Source Zooms [W, H, D]: {source_zooms_whd}")
    logger.info(f"Original Target Zooms [W, H, D]: {target_zooms_whd}")

    # --- Save Preprocessed Input (if requested) ---
    if args.save_new_input and args.iname_new:
        try:
            logger.info(f"Saving preprocessed input to {args.iname_new}...")
            # interp_sitk_native is the volume *before* any numpy conversion/intensity scaling,
            # but *after* resampling to the target spacing/size. This seems appropriate to save.
            sitk.WriteImage(interp_sitk_native, args.iname_new)
            logger.info(f"Saved preprocessed input to {args.iname_new}.")
        except Exception as e:
            logger.error(f"Failed to save preprocessed image to {args.iname_new}: {e}")
            # Continue execution even if saving fails

    # --- Record Preprocessing Time (if requested) ---
    if args.trilinear_proc_time_txt:
        try:
            Path(args.trilinear_proc_time_txt).parent.mkdir(parents=True, exist_ok=True)
            with open(args.trilinear_proc_time_txt, "w") as f:
                f.write(f"{preprocess_duration:.4f}\n")
            logger.info(f"Saved preprocessing time to {args.trilinear_proc_time_txt}")
        except Exception as e:
            logger.error(f"Failed to write preprocessing time to {args.trilinear_proc_time_txt}: {e}")


    # --- Normalization [-1, 1] ---
    # Normalize the numpy array (D,H,W) for model input
    interp_min, interp_max = interp_v8_np.min(), interp_v8_np.max()
    logger.info(f"Preprocessed image intensity range before normalization: [{interp_min:.4f}, {interp_max:.4f}]")
    denom = interp_max - interp_min
    if denom < 1e-8:
        logger.warning("Input image appears to be constant after preprocessing. Normalizing to 0.")
        norm_np = np.zeros_like(interp_v8_np, dtype=np.float32)
    else:
        # Scale to [-1, 1]
        norm_np = ((interp_v8_np.astype(np.float32) - interp_min) / denom) * 2.0 - 1.0
        logger.info(f"Normalized image intensity range (used by model): [{norm_np.min():.4f}, {norm_np.max():.4f}]")

    # --- Load Model ---
    try:
        # Load the specified autoencoder model
        model, params_model, is_film = model_loading_wizard_ae_v2(args, logger)
        model.to(args.device) # Move model to the selected device (CPU or CUDA)
        logger.info(f"Model loaded successfully. FiLM model: {is_film}")

        # Inspect model signature once here (relevant for FiLM checks inside run_autoencoder_3d)
        model_to_inspect = model.module if hasattr(model, 'module') else model
        needs_source_cond = False
        needs_target_cond = False
        if is_film:
            try:
                if not hasattr(model_to_inspect, 'forward'):
                    raise AttributeError("Loaded model object has no 'forward' method.")
                sig = inspect.signature(model_to_inspect.forward)
                needs_source_cond = "source_cond_input" in sig.parameters
                needs_target_cond = "target_cond_input" in sig.parameters
                logger.info(f"Model forward signature inspection: Needs source_cond={needs_source_cond}, Needs target_cond={needs_target_cond}")
            except AttributeError as e:
                logger.warning(f"Could not inspect model.forward signature: {e}. Assuming FiLM needs based on is_film_model flag.")
                needs_source_cond = True
                needs_target_cond = True
            except Exception as e:
                logger.error(f"Unexpected error during model signature inspection: {e}")
                raise

    except Exception as e:
        logger.exception(f"Failed to load model '{args.name}': {e}")
        sys.exit(1)

    # --- Run Inference ---
    inf_start_time = time.time()
    out_tensor = None # Initialize output tensor

    try:
        if args.multi_orientation_avg:
            logger.info("--- Multi-Orientation Averaging Enabled ---")
            # Define axis permutations for data (D,H,W -> H,D,W -> W,H,D)
            # Data permutation means how to transpose the axes of the numpy array
            # Original: (0, 1, 2) -> D, H, W
            # Rotated 1: (1, 0, 2) -> H, D, W
            # Rotated 2: (2, 1, 0) -> W, H, D (Note: This assumes W becomes the new 'Depth')
            data_permutations = [(0, 1, 2), (1, 0, 2), (2, 1, 0)]

            # Define corresponding permutations for zoom lists [W, H, D]
            # Zoom permutation maps OLD indices [W=0, H=1, D=2] to NEW indices based on data perm
            # Data Perm (0,1,2) D,H,W -> Zooms (0,1,2) W,H,D (No change)
            # Data Perm (1,0,2) H,D,W -> New W is old W (idx 0), New H is old D (idx 2), New D is old H (idx 1) -> Zooms (0, 2, 1)
            # Data Perm (2,1,0) W,H,D -> New W is old D (idx 2), New H is old H (idx 1), New D is old W (idx 0) -> Zooms (2, 1, 0)
            zoom_permutations = [(0, 1, 2), (0, 2, 1), (2, 1, 0)]

            # Define inverse permutations to rotate results back to original (D, H, W)
            # This maps the axes of the *rotated* result back to the original order.
            # It's the argsort of the data permutation.
            inverse_data_permutations = [(0, 1, 2), (1, 0, 2), (2, 1, 0)] # argsort(0,1,2)=[0,1,2], argsort(1,0,2)=[1,0,2], argsort(2,1,0)=[2,1,0]

            # --- MODIFICATION START ---
            # Common loop for processing all orientations
            def process_orientations():
                for i, (data_perm, zoom_perm, inv_perm) in enumerate(zip(data_permutations, zoom_permutations, inverse_data_permutations)):
                    logger.info(f"\n--- Processing Orientation {i+1}/3 (Data Permutation: {data_perm}) ---")

                    # Rotate data using numpy transpose
                    rotated_data_np = np.transpose(norm_np, data_perm)
                    logger.info(f"Rotated data shape D',H',W': {rotated_data_np.shape}")

                    # Permute zooms for FiLM if needed
                    current_src_zooms = [source_zooms_whd[k] for k in zoom_perm] if source_zooms_whd is not None and is_film else None
                    current_tgt_zooms = [target_zooms_whd[k] for k in zoom_perm] if target_zooms_whd is not None and is_film else None

                    # Use the originally specified patch size and stride for the rotated data
                    current_patch_shape = args.max_patch_size
                    current_stride = args.stride

                    # Call the inference function, which returns tensor on args.device
                    out_tensor_rotated = run_autoencoder_3d(
                        img_filename=f"{img_filename}_orient{i}",
                        orig_data_normalized_dhw=torch.from_numpy(rotated_data_np.copy()),
                        params_model=params_model, model=model, logger=logger, args=args,
                        is_film_model=is_film, source_zooms_list_whd=current_src_zooms,
                        target_zooms_list_whd=current_tgt_zooms, patch_shape_dhw=current_patch_shape,
                        stride_dhw=current_stride
                    )
                    
                    yield out_tensor_rotated, inv_perm
            
            # Warning about patch size/stride
            logger.warning("Multi-orientation processing enabled. The SAME patch_size and stride "
                           f"({args.max_patch_size}, {args.stride}) will be used for all orientations. "
                           "This might be suboptimal if the network or data is highly anisotropic "
                           "and patch/stride were tuned only for the original orientation.")

            if args.aggregate_on_gpu:
                # --- Aggregate on GPU ---
                logger.info("Aggregating multi-orientation results on GPU (faster, uses more VRAM).")
                results_gpu = []
                for out_tensor_rotated, inv_perm in process_orientations():
                    # Rotate result back on the GPU using torch.permute
                    rotated_back_tensor = torch.permute(out_tensor_rotated, inv_perm)
                    results_gpu.append(rotated_back_tensor)
                    logger.info(f"Result for orientation stored on GPU. Shape: {rotated_back_tensor.shape}")
                
                logger.info("Averaging results from all orientations on GPU...")
                if results_gpu:
                    out_tensor = torch.stack(results_gpu).mean(dim=0).float()
                    logger.info(f"Final averaged tensor shape on GPU: {out_tensor.shape}")
                else:
                    raise RuntimeError("Multi-orientation processing yielded no results.")
                # Final result is moved to CPU after aggregation for post-processing
                out_tensor = out_tensor.cpu()

            else:
                # --- Aggregate on CPU (Default Behavior) ---
                logger.info("Aggregating multi-orientation results on CPU (slower, conserves VRAM).")
                results_cpu = []
                for out_tensor_rotated, inv_perm in process_orientations():
                    # Move tensor to CPU, convert to numpy, transpose, and convert back to tensor
                    rotated_back_np = np.transpose(out_tensor_rotated.cpu().numpy(), inv_perm).copy()
                    results_cpu.append(torch.from_numpy(rotated_back_np))
                    logger.info(f"Result for orientation rotated back and stored on CPU. Shape: {rotated_back_np.shape}")

                logger.info("Averaging results from all orientations on CPU...")
                if results_cpu:
                    out_tensor = torch.stack(results_cpu).mean(dim=0).float()
                    logger.info(f"Final averaged tensor shape on CPU: {out_tensor.shape}")
                else:
                    raise RuntimeError("Multi-orientation processing yielded no results.")
            # --- MODIFICATION END ---

        else: # Original single orientation processing
            logger.info("--- Single Orientation Processing ---")
            out_tensor = run_autoencoder_3d(
                img_filename=img_filename,
                orig_data_normalized_dhw=torch.from_numpy(norm_np.copy()), # Original normalized data
                params_model=params_model,
                model=model,
                logger=logger,
                args=args,
                is_film_model=is_film,
                source_zooms_list_whd=source_zooms_whd, # Original zooms
                target_zooms_list_whd=target_zooms_whd, # Original zooms
                patch_shape_dhw=args.max_patch_size,      # Original patch size
                stride_dhw=args.stride                   # Original stride
            )
            # Ensure tensor is on CPU for post-processing
            out_tensor = out_tensor.cpu()

    except Exception as e:
        # Catch errors during the inference stage (single or multi-orientation)
        logger.exception(f"Error during model inference: {e}")
        sys.exit(1)

    inf_duration = time.time() - inf_start_time
    logger.info(f"Total Inference time took {inf_duration:.2f}s")

    # --- Postprocessing Intensities ---
    # Ensure out_tensor is on CPU before numpy conversion (should be already)
    out_tensor = out_tensor.cpu()
    mn, mx = out_tensor.min(), out_tensor.max()
    logger.info(f"Network output tensor range before final scaling: [{mn:.4f}, {mx:.4f}]")

    out_np = None # Initialize final numpy array
    if (mx - mn).abs() < 1e-6:
        # Handle case where network output is constant
        logger.warning("Network output is constant. Creating zero array matching original preprocessed shape.")
        # interp_v8_np has the correct D,H,W shape of the *original* orientation
        out_np = np.zeros_like(interp_v8_np, dtype=np.float32)
    else:
        # Scale output to [0, 1] first
        logger.info(f"Scaling output tensor to [0, 1] range for final processing using method {args.net2out_rescaling}")
        if args.net2out_rescaling == "linear":
            scaled_01 = (out_tensor - mn) / (mx - mn)
        elif args.net2out_rescaling == "clamp":
            scaled_01 = torch.clamp(out_tensor, min=-1, max=1)
            scaled_01 = (scaled_01 + 1) / (2) # Scale to [0, 1] from [-1, 1]
        mode = int(args.intensity_range_mode)

        if mode == 0: # [0, 255] integer
            # Scale to 0-255, round, clamp, and convert to uint8
            out_np = (scaled_01 * 255.0).round().clamp(0, 255).byte().numpy()
            logger.info("Rescaled output to [0, 255] uint8.")
        elif mode == 2: # Original Min-Max based on the *initial* loaded volume
            # max_orig, min_orig came from the initial load_and_rescale_image_sitk_v2
            out_np_float = (scaled_01 * (max_orig_final - min_orig_final) + min_orig_final).numpy()
            logger.info(f"Rescaled output to original range [{min_orig_final:.4f}, {max_orig_final:.4f}]. Attempting to match original dtype.")
            try:
                # Get dtype from the SimpleITK image *after* resampling but *before* normalization
                # This represents the target dtype if we want to match the resampled input's type
                original_dtype_np = sitk.GetArrayViewFromImage(interp_sitk_v8).dtype
                logger.info(f"Attempting to cast to original resampled dtype: {original_dtype_np}")
                # Perform casting with checks for safety, e.g., check ranges if casting to int
                if np.issubdtype(original_dtype_np, np.integer):
                    # Clip to the valid range of the target integer type before casting
                    type_info = np.iinfo(original_dtype_np)
                    out_np_float = np.clip(np.round(out_np_float), type_info.min, type_info.max)
                out_np = out_np_float.astype(original_dtype_np)
                logger.info(f"Successfully cast output to original resampled dtype {out_np.dtype}.")
            except Exception as e:
                # Fallback if casting fails
                logger.warning(f"Could not determine or cast to original resampled dtype accurately, using preprocessed float dtype {interp_v8_np.dtype}. Error: {e}")
                out_np = out_np_float.astype(interp_v8_np.dtype) # Fallback to the float type used in processing
        else: # mode == 1, [0, 1] float
            # Keep as float32 in the [0, 1] range
            out_np = scaled_01.numpy().astype(np.float32)
            logger.info("Rescaled output to [0, 1] float32.")

    # --- Final Shape Check and Saving ---
    # interp_v8_np holds the shape D,H,W of the data as it was *before* normalization
    # This should be the target shape for our final out_np
    if out_np.shape != interp_v8_np.shape:
        logger.error(f"Final numpy array shape {out_np.shape} does not match expected shape {interp_v8_np.shape} (original D,H,W).")
        # Attempt common ZYX <-> DHW transpose fix if dimensions match inversely
        if len(out_np.shape) == 3 and out_np.shape[::-1] == interp_v8_np.shape:
            logger.warning("Attempting transpose (possible ZYX -> DHW mismatch) to match expected shape.")
            out_np = np.transpose(out_np, (2, 1, 0))
            if out_np.shape != interp_v8_np.shape:
                logger.error("Transpose did not fix shape mismatch. Aborting.")
                sys.exit(1)
            else:
                logger.info("Transpose successful, shape now matches.")
        else:
            logger.error("Shape mismatch cannot be fixed by simple transpose. Aborting.")
            sys.exit(1)

    # --- Create SimpleITK Image, Resample, and Match Histogram ---
    img_save_start = time.time()
    try:
        # Create the SimpleITK image from the final numpy array
        final_output_sitk = sitk.GetImageFromArray(out_np)
        final_output_sitk.CopyInformation(interp_sitk_v8)
        logger.info("Successfully created final SimpleITK image and copied geometry information.")

        # --- Resample to Native Interpolated Space (Mandatory) ---
        logger.info("--- Resampling final output to match interpolated native input space ---")
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(interp_sitk_native)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(float(out_np.min()))
        image_to_save = resampler.Execute(final_output_sitk)
        logger.info("Resampling complete.")

    except Exception as e:
        logger.exception(f"Failed during final image creation or resampling stage: {e}")
        sys.exit(1)

    # --- Intensity Histogram Matching (based on mode) ---
    if args.histogram_matching_mode == 'nonlinear':
        logger.info("--- Applying NON-LINEAR histogram matching ---")
        try:
            source_image = image_to_save
            reference_image = interp_sitk_native
            original_pixel_type = source_image.GetPixelID()
            source_float = sitk.Cast(source_image, sitk.sitkFloat32)
            reference_float = sitk.Cast(reference_image, sitk.sitkFloat32)
            matcher = sitk.HistogramMatchingImageFilter()
            matcher.SetNumberOfHistogramLevels(1024)
            matcher.SetNumberOfMatchPoints(7)
            matcher.ThresholdAtMeanIntensityOn()
            logger.info("Executing non-linear histogram matching...")
            matched_float = matcher.Execute(source_float, reference_float)
            logger.info("Casting result back to original pixel type...")
            image_to_save = sitk.Cast(matched_float, original_pixel_type)
        except Exception as e:
            logger.exception(f"Failed to perform non-linear histogram matching. Saving image without it. Error: {e}")

    elif args.histogram_matching_mode == 'linear':
        logger.info("--- Applying LINEAR histogram matching (Mean/StdDev) ---")
        try:
            source_image = image_to_save
            reference_image = interp_sitk_native
            original_pixel_type = source_image.GetPixelID()
            source_float = sitk.Cast(source_image, sitk.sitkFloat32)
            reference_float = sitk.Cast(reference_image, sitk.sitkFloat32)
            stats_filter = sitk.StatisticsImageFilter()
            stats_filter.Execute(source_float)
            mean_source, stddev_source = stats_filter.GetMean(), stats_filter.GetSigma()
            stats_filter.Execute(reference_float)
            mean_ref, stddev_ref = stats_filter.GetMean(), stats_filter.GetSigma()
            logger.info(f"Source: Mean={mean_source:.2f}, StdDev={stddev_source:.2f}")
            logger.info(f"Reference: Mean={mean_ref:.2f}, StdDev={stddev_ref:.2f}")
            logger.info("Executing linear intensity transformation...")
            if stddev_source > 1e-8:
                  normalized_source = (source_float - mean_source) / stddev_source
                  matched_float = normalized_source * stddev_ref + mean_ref
            else:
                  matched_float = source_float - mean_source + mean_ref
            logger.info("Casting result back to original pixel type...")
            image_to_save = sitk.Cast(matched_float, original_pixel_type)
        except Exception as e:
            logger.exception(f"Failed to perform linear histogram matching. Saving image without it. Error: {e}")
    else:
        logger.info("--- Histogram matching disabled by user ('none'). ---")

    # --- Write Output Image ---
    try:
        logger.info(f"Saving final output to {args.out_name}")
        sitk.WriteImage(image_to_save, args.out_name)
        img_save_duration = time.time() - img_save_start
        logger.info(f"Image saving took {img_save_duration:.2f}s")
    except Exception as e:
        logger.exception(f"Failed to write final output image to {args.out_name}: {e}")
        sys.exit(1)

    # --- Generate Final Summary ---
    logger.info("\n" + "="*80)
    logger.info("--- PROCESSING SUMMARY ---")

    # 1. Original Input Details
    try:
        original_summary_sitk = sitk.ReadImage(img_filename)
        original_dims = original_summary_sitk.GetSize()
        original_zooms_summary = ["{:.4f}".format(z) for z in original_summary_sitk.GetSpacing()]
        logger.info("\n--- [1] Original Input ---")
        logger.info(f"  File Path: {img_filename}")
        logger.info(f"  Zooms (W,H,D): {original_zooms_summary}")
        logger.info(f"  Dimensions (W,H,D): {original_dims}")
        logger.info(f"  Intensity Range: [{min_orig:.4f}, {max_orig:.4f}]")
    except Exception as e:
        logger.error(f"Could not generate summary for original input: {e}")

    # 2. Interpolated Input Details (if saved)
    if args.save_new_input and args.iname_new and Path(args.iname_new).exists():
        try:
            interp_dims = interp_sitk_native.GetSize()
            interp_zooms_summary = ["{:.4f}".format(z) for z in interp_sitk_native.GetSpacing()]
            interp_intensity_min, interp_intensity_max = interp_v8_np.min(), interp_v8_np.max()
            logger.info("\n--- [2] Interpolated Input (Upsampled) ---")
            logger.info(f"  File Path: {args.iname_new}")
            logger.info(f"  Zooms (W,H,D): {interp_zooms_summary}")
            logger.info(f"  Dimensions (W,H,D): {interp_dims}")
            logger.info(f"  Intensity Range (as processed): [{interp_intensity_min:.4f}, {interp_intensity_max:.4f}]")
        except Exception as e:
            logger.error(f"Could not generate summary for interpolated input: {e}")
    else:
        logger.info("\n--- [2] Interpolated Input (Upsampled) ---")
        logger.info("  Not saved (or --save_new_input was False).")

    # 3. Final Output Details
    try:
        final_dims = image_to_save.GetSize()
        final_zooms_summary = ["{:.4f}".format(z) for z in image_to_save.GetSpacing()]
        final_np_view = sitk.GetArrayViewFromImage(image_to_save)
        final_intensity_min, final_intensity_max = final_np_view.min(), final_np_view.max()
        logger.info("\n--- [3] Final Output ---")
        logger.info(f"  File Path: {args.out_name}")
        logger.info(f"  Zooms (W,H,D): {final_zooms_summary}")
        logger.info(f"  Dimensions (W,H,D): {final_dims}")
        logger.info(f"  Intensity Range: [{final_intensity_min:.4f}, {final_intensity_max:.4f}]")
    except Exception as e:
        logger.error(f"Could not generate summary for final output: {e}")

    logger.info("="*80 + "\n")

    # --- Final Timings & Logs ---
    total_duration = time.time() - start_total
    logger.info(f"All processing finished successfully in {total_duration:.2f}s")

    # Save inference time if requested
    if args.proc_time_txt:
        try:
            Path(args.proc_time_txt).parent.mkdir(parents=True, exist_ok=True)
            with open(args.proc_time_txt, "w") as f:
                f.write(f"{inf_duration:.4f}\n")
            logger.info(f"Saved inference time to {args.proc_time_txt}")
        except Exception as e:
            logger.error(f"Failed to write processing time to {args.proc_time_txt}: {e}")


# --- Wrapper & Main Execution ---
def main_wrapper():
    """Wraps the main function for robust error handling and timing."""
    script_start_time = time.time()
    try:
        main()
    except SystemExit as e:
        # Catch explicit sys.exit calls
        if e.code is not None and e.code != 0:
            logger.error(f"Script exiting with error code {e.code}.")
        else:
            logger.info(f"Script exiting with code {e.code or 0}.")
        sys.exit(e.code or 0) # Propagate the exit code
    except Exception as e:
        # Catch any unexpected errors
        logger.exception(f"An unexpected error occurred during script execution: {e}")
        sys.exit(1) # Exit with a non-zero code indicating an error
    finally:
        # Log total execution time regardless of success or failure
        script_end_time = time.time()
        logger.info(f"Total script execution finished in {script_end_time - script_start_time:.2f} seconds.")

if __name__ == "__main__":
    # Execute the main function through the wrapper
    main_wrapper()