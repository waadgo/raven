import importlib # <--- ADD THIS
import gzip
import os
from pathlib import Path
import sys
import numpy as np
import torch.backends.cudnn as cudnn
import yaml
import torch
import torch.nn as nn
import nibabel as nib
import glob
# Ensure main_hdf5 is accessible if instantiate_from_config is used elsewhere,
# otherwise, this import might not be needed directly in utils.py
# from main_hdf5 import instantiate_from_config
from omegaconf import OmegaConf
import logging

# Setup logger for utils module if not already configured by main script
# This helps if functions here are called independently or before main logger setup
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO) # Or desired level
    logger.propagate = False


# --- Taming Models Import Handling ---
try:
    # Ensure taming library is installed and accessible
    from taming.models.autoencoders import AutoencoderKL3D, VQModel3D, AutoencoderKL3DFiLM_BiCond
    MODELS_IMPORTED = True
    logger.debug("Successfully imported default model classes from taming.models.autoencoders.")
except ImportError:
    MODELS_IMPORTED = False
    logger.warning("Could not import default model classes from taming.models.autoencoders.")
    logger.warning("Model loading will rely solely on config target path and dynamic import.")
    # Define dummy placeholders if needed elsewhere, or handle errors robustly later
    AutoencoderKL3D, VQModel3D, AutoencoderKL3DFiLM_BiCond = None, None, None

def argparser_downscale_wizard(argparser):
    # CORRECTED: Use 'in_name' instead of 'iname'
    in_name = getattr(argparser, "in_name")
    header = nib.load(in_name).header
    zooms = header.get_zooms()
    z1, z2, z3 = zooms
    factor_h = getattr(argparser, "factor_h")
    factor_w = getattr(argparser, "factor_w")
    factor_z = getattr(argparser, "factor_z")
    z1 *= factor_h
    z2 *= factor_w
    z3 *= factor_z
    z1 = str(z1)[0:3]
    z2 = str(z2)[0:3]
    z3 = str(z3)[0:3]
    suffix = f'_smoothed_to_{z1}by{z2}by{z3}'

    fname = Path(in_name)
    basename = os.path.join(fname.parent, fname.stem)
    ext = fname.suffix
    if ext == ".gz":
        fname2 = Path(basename)
        basename = os.path.join(fname2.parent, fname2.stem)
        ext = fname2.suffix + ext

    # CORRECTED: Check for 'out_name' and set 'out_name'
    if getattr(argparser, "out_name") is None:
        oname = basename + suffix + ext
        setattr(argparser, "out_name", oname)
    return argparser


def gzip_this(in_file):
    """Compresses the input file using gzip and removes the original."""
    logger.debug(f"Gzipping file: {in_file}")
    try:
        with open(in_file, "rb") as f_in:
            in_data = f_in.read() # read the file as bytes

        out_gz = in_file + ".gz" # the name of the compressed file
        with gzip.open(out_gz, "wb") as gzf: # open the compressed file in write mode
            gzf.write(in_data) # write the data to the compressed file

        logger.debug(f"Successfully created gzipped file: {out_gz}")
        # If you want to delete the original file after the gzip is done:
        os.unlink(in_file)
        logger.debug(f"Removed original file: {in_file}")
    except Exception as e:
        logger.error(f"Error during gzipping or deleting {in_file}: {e}")
        # Decide if you want to raise the error or just log it
        # raise

def is_anisotropic(z1, z2, z3):
    """Checks if voxel dimensions are significantly anisotropic."""
    # find the largest value
    largest = max(z1, z2, z3)
    # determine which axis has the largest value
    if largest == z1:
        irr_pos = "Sagittal" # Assuming z1 is Sagittal axis dim
    elif largest == z2:
        irr_pos = "Coronal"  # Assuming z2 is Coronal axis dim
    else:
        irr_pos = "Axial"    # Assuming z3 is Axial axis dim
    # find the smallest value
    smallest = min(z1, z2, z3)
    # compare the largest and smallest values
    if largest >= 2 * smallest:
        logger.warning(f"Voxel size ({z1:.2f}, {z2:.2f}, {z3:.2f}) is anisotropic.")
        logger.warning(f"Voxel size is at least twice as large in the largest dimension ({irr_pos}) than in the smallest dimension.")
        # logger.warning("Will perform denoising only using the "+irr_pos+" plane.") # Commented out as it implies specific behavior not handled here
        return True, irr_pos
    else:
        logger.debug(f"Voxel size ({z1:.2f}, {z2:.2f}, {z3:.2f}) is not significantly anisotropic.")
        return False, None # Return None for position when not anisotropic


def arguments_setup(sel_option):
    """ Sets up default arguments, primarily for a diffusion model context? """
    # CORRECTED: Use 'in_name' instead of 'iname'
    in_name = getattr(sel_option, "in_name")
    model_name = "neuroldm_" # Hardcoded? Consider making this dynamic or an arg
    irm = getattr(sel_option, "intensity_range_mode")
    rri = getattr(sel_option, "robust_rescale_input")
    order = getattr(sel_option, "order")
    use_scipy = getattr(sel_option, "use_scipy")
    suffix_type = getattr(sel_option, "suffix_type") # Assumes this exists
    uf_h = getattr(sel_option, "uf_h")
    uf_w = getattr(sel_option, "uf_w")
    uf_z = getattr(sel_option, "uf_z")

    if getattr(sel_option, "ext", None) is None: # Safely get ext or None
        fname = Path(in_name)
        basename = os.path.join(fname.parent, fname.stem)
        ext = fname.suffix
        if ext.lower() == ".gz": # Use lower() for case-insensitivity
            fname2 = Path(basename)
            basename = os.path.join(fname2.parent, fname2.stem)
            ext = fname2.suffix + ext
        setattr(sel_option, "ext", ext)
    else:
        # Use provided extension
        ext = getattr(sel_option, "ext")
        fname = Path(in_name)
        basename = os.path.join(fname.parent, fname.stem)
        # Handle .nii.gz case for basename if extension is provided separately
        potential_ext = fname.suffix
        if potential_ext.lower() == ".gz":
            fname2 = Path(basename)
            basename = os.path.join(fname2.parent, fname2.stem)


    # Re-read order in case it was changed
    order = getattr(sel_option, "order")
    #=====Setting up the filename suffix =====
    if irm == 0:
        suffix_3 = "_irm0"
    elif irm == 1:
        suffix_3 = "_irm1"
    elif irm == 2:
        suffix_3 = "_irm2"
    else:
        suffix_3 = f"_irm{irm}" # Handle unexpected values

    suffix_4 = "_rri1" if rri else "_rri0"
    suffix_5 = f'_order{order}'
    suffix_6 = f'_scipy{int(use_scipy)}'
    suffix_7 = f'_UF{uf_h}x{uf_w}x{uf_z}'
    settings_suffix = suffix_3 + suffix_4 + suffix_5 + suffix_6 + suffix_7

    # Construct the full suffix string based on suffix_type
    if suffix_type == "detailed":
        # Suffix includes model name and detailed settings
        full_suffix = "_" + model_name + settings_suffix
    else:
        # Suffix only includes model name
        full_suffix = "_" + model_name

    # set the default suffix name if it was not parsed as an argument
    if getattr(sel_option, "suffix", None) is None:
        setattr(sel_option, "suffix", full_suffix)
    else:
        # Use provided suffix if available
        full_suffix = getattr(sel_option, "suffix")


    # CORRECTED: Check for 'out_name' and set 'out_name'
    if getattr(sel_option, "out_name", None) is None:
        # Append the determined full_suffix and extension
        setattr(sel_option, "out_name", basename + full_suffix + ext)

    # Set default for iname_new if not provided
    if getattr(sel_option, "iname_new", None) is None:
        # Append _preprocessed and the full_suffix and extension
        setattr(sel_option, "iname_new", basename + "_preprocessed" + full_suffix + ext)

    # Set default for noise_info_file if not provided (seems specific to diffusion?)
    if getattr(sel_option, "noise_info_file", None) is None:
        setattr(sel_option, "noise_info_file", basename + ".txt")

    args = sel_option

    # Print the final argument values
    logger.info("Final arguments after setup:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    return sel_option

def arguments_setup_ae(sel_option):
    """ Sets up default arguments, specifically for an Autoencoder context. """
    # CORRECTED: Use 'in_name' instead of 'iname'
    in_name = getattr(sel_option, "in_name")
    model_name = getattr(sel_option, "name")
    irm = getattr(sel_option, "intensity_range_mode")
    rri = getattr(sel_option, "robust_rescale_input")
    order = getattr(sel_option, "order")
    use_scipy = getattr(sel_option, "use_scipy")
    uf_h = getattr(sel_option, "uf_h")
    uf_w = getattr(sel_option, "uf_w")
    uf_z = getattr(sel_option, "uf_z")
    # eval_planes = getattr(sel_option, "eval_planes", None) # Safely get eval_planes or None

    if getattr(sel_option, "ext", None) is None: # Safely get ext or None
        fname = Path(in_name)
        basename = os.path.join(fname.parent, fname.stem)
        ext = fname.suffix
        if ext.lower() == ".gz": # Use lower() for case-insensitivity
            fname2 = Path(basename)
            basename = os.path.join(fname2.parent, fname2.stem)
            ext = fname2.suffix + ext
        setattr(sel_option, "ext", ext)
    else:
        # Use provided extension
        ext = getattr(sel_option, "ext")
        fname = Path(in_name)
        basename = os.path.join(fname.parent, fname.stem)
        # Handle .nii.gz case for basename if extension is provided separately
        potential_ext = fname.suffix
        if potential_ext.lower() == ".gz":
            fname2 = Path(basename)
            basename = os.path.join(fname2.parent, fname2.stem)

    # Re-read order in case it was changed
    order = getattr(sel_option, "order")
    #=====Setting up the filename suffix =====
    if irm == 0:
        suffix_3 = "_irm0"
    elif irm == 1:
        suffix_3 = "_irm1"
    elif irm == 2:
        suffix_3 = "_irm2"
    else:
        suffix_3 = f"_irm{irm}" # Handle unexpected values

    suffix_4 = "_rri1" if rri else "_rri0"
    # Include preprocessing details in suffix? Optional, depends on need.
    # suffix_5 = f'_order{order}'
    # suffix_6 = f'_scipy{int(use_scipy)}'
    suffix_7 = f'_UF{uf_h}x{uf_w}x{uf_z}'
    # Suffix for eval_planes if it exists?
    # suffix_13 = f'_EP{eval_planes}' if eval_planes is not None else ""

    # Combine relevant settings into a suffix string
    # Example: just upscaling factor
    settings_suffix = suffix_7 # + suffix_13
    # Example: include intensity range and robust rescale
    # settings_suffix = suffix_3 + suffix_4 + suffix_7 + suffix_13

    # Construct the full suffix string, always including the model name
    full_suffix = "_" + model_name + settings_suffix

    # set the default suffix name if it was not parsed as an argument
    if getattr(sel_option, "suffix", None) is None:
        setattr(sel_option, "suffix", full_suffix)
    else:
        # Use provided suffix if available
        full_suffix = getattr(sel_option, "suffix")

    # CORRECTED: Check for 'out_name' and set 'out_name'
    if getattr(sel_option, "out_name", None) is None:
        # Append the determined full_suffix and extension
        setattr(sel_option, "out_name", basename + full_suffix + ext)

    # Set default for iname_new if not provided
    if getattr(sel_option, "iname_new", None) is None:
         # Append _preprocessed and the full_suffix and extension
         # Ensure basename reflects potential removal of .gz earlier
         # Derive from final out_name's basename for safety?
        out_fname = Path(getattr(sel_option, "out_name"))
        out_basename = os.path.join(out_fname.parent, out_fname.stem)
        out_ext = out_fname.suffix
        if out_ext.lower() == ".gz":
             out_fname2 = Path(out_basename)
             out_basename = os.path.join(out_fname2.parent, out_fname2.stem)
             out_ext = out_fname2.suffix + out_ext

        setattr(sel_option, "iname_new", out_basename + "_preprocessed" + out_ext) # Use derived out_name base and ext

    args = sel_option

    # Print the final argument values
    logger.info("Final arguments after AE setup:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    return sel_option


def add_noise(x, noise='.'):
        """Adds Gaussian or Poisson noise to a numpy array."""
        if noise is None or noise == '.':
             logger.debug("No noise added.")
             return x

        try:
            noise_type = noise[0].upper() # Use upper for case-insensitivity
            noise_value = float(noise[1:]) / 100.0 # Ensure float division
        except (IndexError, ValueError) as e:
             logger.error(f"Invalid noise format '{noise}'. Expected 'G<std_dev_percent>' or 'S<lambda_percent>'. Error: {e}")
             return x # Return original data on error

        if noise_type == 'G':
            # Add Gaussian noise with std_dev = noise_value * max(x) ? Or just noise_value?
            # Assuming noise_value is the desired standard deviation directly.
            # Adjust scale if noise_value represents percentage of max intensity?
            # Current: Assumes noise_value is absolute std dev.
            noises = np.random.normal(scale=noise_value, size=x.shape)
            logger.debug(f"Adding Gaussian noise with std dev: {noise_value}")
            # noises = noises.round() # Rounding might not be desired
        elif noise_type == 'S':
            # Add Poisson noise. Intensity values scaled by noise_value.
            # Ensure x is non-negative for Poisson. Clamp?
            x_nonneg = np.maximum(x, 0)
            # The scaling seems unusual. Poisson lambda is usually proportional to signal intensity.
            # lambda = x * scale_factor. Noise ~ sqrt(lambda).
            # Maybe noise_value relates to the scale_factor? Revisit this logic.
            # A simpler approach: lambda = x * noise_value, noise = poisson(lambda)
            # Current implementation is kept but flagged as potentially needing review.
            logger.debug(f"Adding Poisson noise with scale factor related to: {noise_value}")
            noises = np.random.poisson(x_nonneg * noise_value) / noise_value
            # Centering the noise? This also seems non-standard for pure Poisson.
            noises = noises - noises.mean(axis=0).mean(axis=0) # Removed mean subtraction axis=0 assumption, center globally or not at all?
            logger.warning("Poisson noise implementation and centering might need review for physical accuracy.")
        else:
             logger.error(f"Unknown noise type '{noise_type}'. Use 'G' or 'S'.")
             return x

        # Ensure data types are compatible for addition (float)
        # Using abs() might not be correct, noise can be negative. Allow noisy signal to go below zero?
        # x_noise = abs(x.astype(np.float64) + noises.astype(np.float64)) # Original abs()
        x_noise = x.astype(np.float64) + noises.astype(np.float64)
        return x_noise

def filename_wizard(img_filename, save_as, save_as_new_orig):
    """ Extracts basenames, extensions, and gzip status from filenames. """
    logger.debug(f"Running filename_wizard for: IN={img_filename}, OUT={save_as}, NEW_ORIG={save_as_new_orig}")
    paths = {'in': img_filename, 'out': save_as, 'innew': save_as_new_orig}
    results = {}

    for key, file_path in paths.items():
        if not file_path:
             logger.warning(f"Empty path provided for key '{key}' in filename_wizard.")
             results[f'basename_{key}'] = ""
             results[f'ext_{key}'] = ""
             results[f'is_gzip_{key}'] = False
             continue

        try:
            fname = Path(file_path)
            basename = os.path.join(fname.parent, fname.stem)
            ext = fname.suffix
            is_gzip = False

            # Check for .gz extension specifically
            if ext.lower() == ".gz":
                is_gzip = True
                # Get the extension before .gz
                fname_no_gz = Path(basename)
                ext = fname_no_gz.suffix + ext # Combine original ext + .gz
                basename = os.path.join(fname_no_gz.parent, fname_no_gz.stem)

            results[f'basename_{key}'] = basename
            results[f'ext_{key}'] = ext
            results[f'is_gzip_{key}'] = is_gzip
            logger.debug(f"  {key}: basename='{basename}', ext='{ext}', gzip={is_gzip}")

        except Exception as e:
             logger.error(f"Error processing path '{file_path}' for key '{key}' in filename_wizard: {e}")
             results[f'basename_{key}'] = ""
             results[f'ext_{key}'] = ""
             results[f'is_gzip_{key}'] = False


    return (results['basename_in'], results['basename_out'], results['basename_innew'],
            results['ext_in'], results['ext_out'], results['ext_innew'],
            results['is_gzip_in'], results['is_gzip_out'], results['is_gzip_innew'])


def model_loading_wizard_ae_v2(args, logger):
    """
    Loads the appropriate AE model (KL, VQ, or FiLM) by reading the target
    from the config file or checkpoint hyperparameters. Includes robust checkpoint finding.
    """

    # --- Basic Logger Setup (ensure logger passed is used) ---
    logger.info("Starting AE model loading wizard v2...")

    # --- 1. Find Checkpoint File Path ---
    checkpoint_path = None
    search_name = args.name # The name/pattern/path provided by the user
    model_base_path = args.model_path # Optional base directory for models

    # Check 1: Is args.name already a direct, valid file path?
    if os.path.isfile(search_name):
        logger.info(f"Attempting to use provided path directly as checkpoint: {search_name}")
        checkpoint_path = search_name

    # Check 2: If not found directly, check relative to args.model_path (if provided)
    if checkpoint_path is None and model_base_path:
        potential_path = os.path.join(model_base_path, search_name)
        logger.info(f"Checking path relative to --model_path: {potential_path}")
        if os.path.isfile(potential_path):
            checkpoint_path = potential_path
            logger.info(f"Found checkpoint relative to --model_path.")

    # Check 3: If still not found, try common patterns (e.g., .ckpt extension)
    if checkpoint_path is None:
        potential_path_ckpt = search_name + ".ckpt"
        logger.info(f"Checking direct path with .ckpt extension: {potential_path_ckpt}")
        if os.path.isfile(potential_path_ckpt):
             checkpoint_path = potential_path_ckpt
        elif model_base_path:
            potential_path_ckpt_rel = os.path.join(model_base_path, potential_path_ckpt)
            logger.info(f"Checking path relative to --model_path with .ckpt: {potential_path_ckpt_rel}")
            if os.path.isfile(potential_path_ckpt_rel):
                checkpoint_path = potential_path_ckpt_rel

    # Check 4: If still not found, try glob patterns (relative to script/cwd first in 'checkpoints/')
    # Determine script directory safely
    try:
        script_dir_abs = os.path.abspath(os.path.dirname(__file__))
    except NameError:
        script_dir_abs = os.path.abspath(os.getcwd())
        logger.warning(f"__file__ not defined, using CWD for script_dir: {script_dir_abs}")

    search_dirs_glob = [os.path.join(script_dir_abs, 'checkpoints')] # Default search location
    if model_base_path and os.path.isdir(model_base_path):
        search_dirs_glob.append(os.path.abspath(model_base_path)) # Add model_path if valid

    if checkpoint_path is None:
        logger.info(f"Checkpoint not found directly, searching with glob pattern '*{search_name}*' in: {search_dirs_glob}")
        for glob_dir in search_dirs_glob:
            ckpt_pattern = os.path.join(glob_dir, f'*{search_name}*')
            try:
                logger.debug(f"Globbing pattern: {ckpt_pattern}")
                list_of_files = sorted(glob.glob(ckpt_pattern)) # Sort for consistency
                # Prioritize .ckpt files if multiple matches
                ckpt_files = [f for f in list_of_files if f.endswith(".ckpt")]
                if ckpt_files:
                    checkpoint_path = ckpt_files[0] # Use first .ckpt match
                    logger.info(f"Found checkpoint via glob: {checkpoint_path}")
                    if len(ckpt_files) > 1:
                        logger.warning(f"Multiple .ckpt files matched pattern '{ckpt_pattern}', using first one.")
                    break # Stop searching once found
                elif list_of_files:
                     # If no .ckpt files, use the first match of any type
                     checkpoint_path = list_of_files[0]
                     logger.info(f"Found potential checkpoint via glob (non-.ckpt): {checkpoint_path}")
                     if len(list_of_files) > 1:
                         logger.warning(f"Multiple files matched pattern '{ckpt_pattern}', using first one.")
                     break # Stop searching once found
            except Exception as e:
                logger.warning(f"Error during glob search in '{glob_dir}': {e}")
        if checkpoint_path:
            logger.info(f"Using checkpoint found via glob: {checkpoint_path}")


    # --- FINAL CHECK for Checkpoint Path ---
    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        logger.error(f"CRITICAL: Checkpoint file could not be found for name/pattern: '{search_name}'")
        logger.error(f"Searched: direct path, relative to --model_path ('{model_base_path}'), common patterns, glob in standard locations.")
        sys.exit(1)

    logger.info(f"Final checkpoint path determined: {checkpoint_path}")


    # --- 2. Find and Load Config File ---
    config = None
    config_file = None
    # Expect config name to match checkpoint name (without extension) + .yaml
    base_ckpt_name_for_config = Path(checkpoint_path).stem
    expected_config_filename = f'{base_ckpt_name_for_config}.yaml'
    logger.info(f"Searching for corresponding config file: '{expected_config_filename}'")


    # Define potential base directories for the 'configs' folder relative to key locations
    ckpt_dir_abs = os.path.abspath(os.path.dirname(checkpoint_path))
    cwd_abs = os.path.abspath(os.getcwd())

    potential_config_dirs = [
        ckpt_dir_abs,                          # Same dir as checkpoint
        os.path.join(ckpt_dir_abs, 'configs'), # 'configs' subdir relative to checkpoint
        cwd_abs,                               # Current working directory
        os.path.join(cwd_abs, 'configs'),      # 'configs' subdir relative to CWD
        script_dir_abs,                        # Same dir as this script
        os.path.join(script_dir_abs, 'configs')# 'configs' subdir relative to script
    ]
    # Add parent directories if needed
    potential_config_dirs.extend([
        os.path.dirname(script_dir_abs),
        os.path.dirname(ckpt_dir_abs)
    ])

    # De-duplicate and ensure directories exist
    valid_config_search_dirs = []
    for d in potential_config_dirs:
         abs_d = os.path.abspath(d)
         if os.path.isdir(abs_d) and abs_d not in valid_config_search_dirs:
             valid_config_search_dirs.append(abs_d)

    logger.info(f"Searching for config in potential directories: {valid_config_search_dirs}")

    # Construct and check absolute paths for the config file
    found = False
    for search_dir in valid_config_search_dirs:
        potential_config_path = os.path.join(search_dir, expected_config_filename)
        potential_config_path = os.path.normpath(potential_config_path)
        logger.debug(f"  Checking absolute path: {potential_config_path}")
        if os.path.isfile(potential_config_path):
            config_file = potential_config_path
            logger.info(f"  Config file FOUND at: {config_file}")
            found = True
            break

    # Load the found config file
    if found and config_file:
        logger.info(f"Loading config from {config_file}")
        try:
            config = OmegaConf.load(config_file)
        except Exception as e:
            logger.error(f"Error loading config file {config_file}: {e}. Proceeding without config.")
            config = None # Ensure config is None if loading fails
    else:
        logger.warning(f"No config file '{expected_config_filename}' found in standard locations.")
        logger.warning(f"Will attempt to load model type from checkpoint hparams.")
        config = None # Explicitly set to None


    # --- 3. Determine Model Class ---
    model_class = None
    target_class_path = None
    ckpt_data = None # To store loaded checkpoint data if needed

    # Try 1: From loaded config file
    if config and hasattr(config, 'model') and hasattr(config.model, 'target'):
        try:
            target_class_path = config.model.target
            logger.info(f"Attempting to load model class from config target: {target_class_path}")
            module_path, class_name = target_class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            logger.info(f"Successfully resolved model class from config: {model_class.__name__}")
        except Exception as e:
            logger.warning(f"Failed to load model class from config target '{target_class_path}': {e}. Will try checkpoint hparams.")
            model_class = None

    # Try 2: From checkpoint hparams if config failed or missing target
    if model_class is None:
        logger.info("Attempting to determine model class from checkpoint hyperparameters...")
        try:
            # Load checkpoint on CPU to inspect hparams safely
            # Ensure ckpt_data is loaded only once if needed for fallback later
            if ckpt_data is None:
                ckpt_data = torch.load(checkpoint_path, map_location='cpu')

            # Look for hparams in common locations
            if 'hyper_parameters' in ckpt_data:
                hparams = ckpt_data['hyper_parameters']
            elif 'hparams' in ckpt_data: # Alternative key
                 hparams = ckpt_data['hparams']
            else:
                 hparams = {} # No hparams found

            if isinstance(hparams, dict): # Check if hparams is a dictionary
                # Navigate potentially nested structure for target
                target_class_path = hparams.get('model', {}).get('target') # Common structure
                if not target_class_path and 'target' in hparams: # Check top-level target
                     target_class_path = hparams['target']

                if target_class_path:
                    logger.info(f"Found target in checkpoint hparams: {target_class_path}")
                    module_path, class_name = target_class_path.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    model_class = getattr(module, class_name)
                    logger.info(f"Successfully resolved model class from checkpoint hparams: {model_class.__name__}")
                else:
                    logger.warning("Could not find 'model.target' or 'target' path in checkpoint hyper_parameters.")
                    model_class = None
            else:
                 logger.warning(f"Checkpoint 'hyper_parameters' or 'hparams' field is not a dictionary ({type(hparams)}). Cannot extract target.")
                 model_class = None

        except ImportError as e:
            logger.error(f"ImportError resolving class from hparams '{target_class_path}': {e}. Check path and environment.")
            model_class = None
        except KeyError as e:
            logger.warning(f"KeyError accessing hyper_parameters in checkpoint: {e}")
            model_class = None
        except Exception as e:
            logger.error(f"Error loading or inspecting checkpoint for hparams: {e}")
            model_class = None

    # Try 3: Final check - Exit if class still unknown
    if model_class is None:
        logger.error(f"CRITICAL: Could not determine model class from config file OR checkpoint hyperparameters for checkpoint: {checkpoint_path}")
        logger.error("Please ensure a valid config file exists OR the checkpoint was saved with `model.target` or `target` in `hyper_parameters`.")
        # Optional: Add filename heuristic fallback here if necessary? Risky.
        sys.exit(1)


    # --- 4. Load Model Weights ---
    logger.info(f"Loading checkpoint weights into instance of {model_class.__name__}")
    model = None
    try:
        # Prefer load_from_checkpoint if the class supports it (e.g., LightningModule)
        if hasattr(model_class, 'load_from_checkpoint'):
            model = model_class.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                strict=False, # Allow missing/unexpected keys initially
                map_location='cpu' # Load to CPU first
                )
            logger.info("Model weights loaded successfully via load_from_checkpoint.")
        else:
            logger.info("Model class does not have load_from_checkpoint. Using fallback.")
            raise NotImplementedError("Fallback needed") # Trigger fallback

    except Exception as e1:
        # Fallback to manual instantiation + load_state_dict
        logger.warning(f"load_from_checkpoint failed or not available ({e1}). Attempting fallback with manual instantiation + state_dict load.")

        # Ensure config or hparams are available for fallback
        # Use hparams from ckpt_data if config is None and ckpt_data was loaded
        if config is None:
            if ckpt_data and ('hyper_parameters' in ckpt_data or 'hparams' in ckpt_data):
                hparams_key = 'hyper_parameters' if 'hyper_parameters' in ckpt_data else 'hparams'
                # Ensure hparams are dict before creating OmegaConf
                if isinstance(ckpt_data[hparams_key], dict):
                     config = OmegaConf.create(ckpt_data[hparams_key])
                     logger.info("Using hparams from checkpoint for fallback instantiation.")
                else:
                    logger.error(f"Checkpoint '{hparams_key}' not a dictionary. Cannot use for fallback config.")
                    config = None # Keep config as None
            else:
                logger.error("Cannot use fallback: Config file missing and usable hparams not found in checkpoint.")
                raise RuntimeError("Cannot instantiate model for fallback loading.") from e1

        # Double check config is usable after potential hparam loading
        if config is None:
             logger.error("Cannot use fallback: Failed to obtain usable config/hparams.")
             raise RuntimeError("Cannot instantiate model for fallback loading.") from e1


        try:
            # Attempt to instantiate the model using config/hparams
            # This assumes config contains necessary params under 'model.params' or directly
            if hasattr(config, 'model') and hasattr(config.model, 'params'):
                 params = config.model.params
                 logger.info("Using model.params from config/hparams for instantiation.")
            elif hasattr(config, 'params'): # Check if params are directly under config
                 params = config.params
                 logger.info("Using direct params from config/hparams for instantiation.")
            else:
                 params = config # Assume config itself contains the direct params? Risky.
                 logger.warning("Config structure missing 'model.params' or 'params'. Assuming config root holds params.")

            # Instantiate using the determined class and extracted params
            # Need to handle potential kwargs mismatch carefully
            try:
                model = model_class(**params) # Instantiate
            except TypeError as te:
                logger.error(f"TypeError during manual instantiation of {model_class.__name__}: {te}")
                logger.error(f"Check if params {list(params.keys())} match class __init__ signature.")
                raise te # Re-raise error

            # Load state dict manually
            if ckpt_data is None: # Load if not already loaded
                ckpt_data = torch.load(checkpoint_path, map_location='cpu')
            state_dict = ckpt_data.get("state_dict", ckpt_data) # Allow state_dict directly

            ignore_keys = [] # Define keys to ignore if necessary
            # Use custom load function if available, otherwise standard Pytorch load
            if 'load_model_from_ckpt' in globals():
                model = load_model_from_ckpt(model, checkpoint_path, verbose=True, ignore_keys=ignore_keys)
            else:
                logger.warning("'load_model_from_ckpt' helper not found. Using standard torch load_state_dict.")
                m, u = model.load_state_dict(state_dict, strict=False)
                if m: logger.warning(f"Missing keys during state_dict load: {m}")
                if u: logger.warning(f"Unexpected keys during state_dict load: {u}")

            logger.info("Model loaded using fallback mechanism (manual instantiation + state_dict).")

        except Exception as e2:
            logger.error(f"Fallback loading also failed: {e2}")
            # Log traceback for detailed debugging if needed
            # import traceback
            # logger.error(traceback.format_exc())
            raise RuntimeError("Failed to load model using both load_from_checkpoint and fallback.") from e2


    # --- 5. Determine if Loaded Model is FiLM ---
    is_film_model = False
    try:
        film_class_ref = AutoencoderKL3DFiLM_BiCond # From top-level import (or None)

        # If initial import failed, try dynamic import again for the check
        if film_class_ref is None and MODELS_IMPORTED is False:
            try:
                film_target_path = 'taming.models.autoencoders.AutoencoderKL3DFiLM_BiCond' # Standard path
                film_module_path, film_class_name = film_target_path.rsplit('.', 1)
                film_module = importlib.import_module(film_module_path)
                film_class_ref = getattr(film_module, film_class_name)
                logger.debug("Dynamically imported FiLM class for type check.")
            except Exception as import_err:
                logger.warning(f"Could not dynamically import FiLM class for final type check: {import_err}")
                film_class_ref = None # Stay None if import fails

        # Perform the check if we have a valid reference class
        if film_class_ref and isinstance(model, film_class_ref):
            is_film_model = True
            logger.info("Confirmed loaded model IS an instance of FiLM model class.")
        elif model_class and model_class.__name__ == "AutoencoderKL3DFiLM_BiCond":
             # Fallback check using the resolved class name if isinstance fails or ref is None
             is_film_model = True
             logger.info("Confirmed loaded model IS a FiLM model (by resolved class name check).")
        else:
             is_film_model = False
             logger.info("Confirmed loaded model is NOT a FiLM model.")

    except Exception as e:
         logger.warning(f"Could not perform robust check for FiLM model type: {e}.")
         # Fallback using the initially resolved class name as last resort
         is_film_model = model_class.__name__ == "AutoencoderKL3DFiLM_BiCond" if model_class else False
         logger.warning(f"Using initially resolved class name for FiLM check: {is_film_model}")


    # --- 6. Post-Load Setup ---
    model.eval() # Set to evaluation mode

    # Determine device based on args and availability
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # Use device already set in main script's args object if available
    if hasattr(args, 'device'):
        device = args.device
        logger.info(f"Using device '{device}' provided via args.")
    else:
        # Fallback if device not set in args yet
        device = torch.device("cuda" if use_cuda else "cpu")
        logger.warning(f"Device not set in args, determined device: {device}")
        args.device = device # Set it back in args?

    logger.info(f"Moving model to device: {device}")
    model.to(device)

    # Handle DataParallel (Optional, often better alternatives like DDP exist)
    model_parallel = False
    # Avoid wrapping if only 1 device or CPU
    if use_cuda and torch.cuda.device_count() > 1:
        # Check if model is ALREADY wrapped (e.g., loaded from DP checkpoint)
        if not isinstance(model, nn.DataParallel):
             logger.info(f"Using nn.DataParallel for {torch.cuda.device_count()} GPUs.")
             model = nn.DataParallel(model)
             model_parallel = True # Mark as parallel after wrapping
        else:
             logger.info("Model seems to be already wrapped in nn.DataParallel.")
             model_parallel = True # Mark as parallel if already wrapped
    elif isinstance(model, nn.DataParallel): # Handle case where it was loaded wrapped but now on CPU/1 GPU
        logger.warning("Model loaded as DataParallel but running on CPU or 1 GPU. Using model.module.")
        model = model.module # Unwrap it
        model_parallel = False

    # Handle EMA (Requires specific logic based on how EMA was saved/managed)
    if args.use_ema:
        logger.warning("EMA usage requested (--use_ema).")
        # Add specific EMA application call here, e.g., by loading EMA weights separately
        # or calling a method on the model if it manages EMA internally.
        # Example placeholder:
        model_to_apply_ema = model.module if model_parallel else model
        if hasattr(model_to_apply_ema, 'apply_ema'):
             logger.info("Attempting to apply EMA weights using model.apply_ema()")
             try:
                model_to_apply_ema.apply_ema() # Assumes this method exists and switches weights
             except Exception as ema_e:
                logger.error(f"Failed to apply EMA weights: {ema_e}")
        elif hasattr(model_to_apply_ema, 'ema_state_dict'):
             logger.warning("Model has 'ema_state_dict', but no automatic application method found.")
             # Manual loading might be needed here if structure is known
        else:
             logger.error("EMA requested, but no EMA application method (e.g., 'apply_ema') or EMA state found on model.")


    # Bundle parameters useful for the main script
    params_model = {
        'device': device,
        "use_cuda": use_cuda,
        "batch_size": args.batch_size, # Use batch_size from args
        "model_parallel": model_parallel,
        "use_ema_for_sampling": args.use_ema,
        "is_eval": True
    }

    logger.info("Model loading wizard finished.")
    # Return the dynamically determined is_film_model flag
    return model, params_model, is_film_model

# --- load_model_from_ckpt ---
# Consider adding type hints for clarity
# def load_model_from_ckpt(model: torch.nn.Module, ckpt: str, verbose: bool = False, ignore_keys: list = []):
def load_model_from_ckpt(model, ckpt, verbose=False, ignore_keys=[]):
    """Loads state_dict from a checkpoint file into a model instance."""
    logger.info(f"Loading model state_dict from checkpoint: {ckpt}")
    try:
        pl_sd = torch.load(ckpt, map_location="cpu") # Load to CPU first

        if "global_step" in pl_sd:
            logger.info(f"  Checkpoint Global Step: {pl_sd['global_step']}")

        # Look for state_dict in common locations
        if 'state_dict' in pl_sd:
            sd = pl_sd["state_dict"]
            logger.debug("  Found state_dict in 'state_dict' key.")
        elif 'model_state_dict' in pl_sd: # Another common key
             sd = pl_sd["model_state_dict"]
             logger.debug("  Found state_dict in 'model_state_dict' key.")
        else:
             # Assume the loaded object *is* the state_dict if key not found
             sd = pl_sd
             logger.debug("  No standard state_dict key found, assuming loaded object is the state_dict.")

        if not isinstance(sd, dict):
             logger.error(f"Loaded state_dict is not a dictionary (type: {type(sd)}). Cannot load.")
             return model # Return original model on error

        original_keys = list(sd.keys())
        keys_to_load = {}
        deleted_keys_count = 0

        # Filter ignored keys
        for k in original_keys:
            ignore = False
            for ik in ignore_keys:
                if ik and k.startswith(ik):
                    if verbose: logger.info(f"  Ignoring key '{k}' due to ignore pattern '{ik}'.")
                    ignore = True
                    deleted_keys_count += 1
                    break
            if not ignore:
                keys_to_load[k] = sd[k]

        if deleted_keys_count > 0:
            logger.info(f"  Ignored {deleted_keys_count} keys based on ignore_keys patterns.")

        # Load the filtered state dict
        missing_keys, unexpected_keys = model.load_state_dict(keys_to_load, strict=False)

        if missing_keys:
            logger.warning(f"  Missing keys in model state_dict when loading checkpoint:")
            if verbose or len(missing_keys) < 10: # Print keys if verbose or few missing
                 for k in missing_keys: logger.warning(f"    {k}")
            else:
                 logger.warning(f"    ({len(missing_keys)} keys omitted for brevity)")
        if unexpected_keys:
            logger.warning(f"  Unexpected keys found in checkpoint state_dict (not in model):")
            if verbose or len(unexpected_keys) < 10: # Print keys if verbose or few unexpected
                 for k in unexpected_keys: logger.warning(f"    {k}")
            else:
                 logger.warning(f"    ({len(unexpected_keys)} keys omitted for brevity)")

        logger.info(f'State_dict loading summary: Missing={len(missing_keys)}, Unexpected={len(unexpected_keys)}')
        logger.info(f"Successfully loaded state_dict into model.")

    except FileNotFoundError:
        logger.error(f"Checkpoint file not found at {ckpt}")
        raise # Re-raise the error
    except Exception as e:
        logger.error(f"Error loading model from checkpoint {ckpt}: {e}")
        # Log traceback for detailed debugging if needed
        # import traceback
        # logger.error(traceback.format_exc())
        raise # Re-raise the error

    # model.eval() # Set to eval mode after loading? Usually done outside this function.
    return model



# Example Usage (can be removed or kept for testing)
if __name__ == "__main__":
    logger.info("Utils script executed directly. Running basic tests or info.")
    # Example: Test is_anisotropic
    logger.info(f"Is anisotropic (1,1,1): {is_anisotropic(1,1,1)}")
    logger.info(f"Is anisotropic (1,1,2): {is_anisotropic(1,1,2)}")
    logger.info(f"Is anisotropic (1,2,4): {is_anisotropic(1,2,4)}")

    # Example: Test filename_wizard
    test_in = "/path/to/image.nii.gz"
    test_out = "/output/dir/image_processed.nii.gz"
    test_new = "/output/dir/image_processed_preprocessed.nii"
    b_in, b_out, b_new, e_in, e_out, e_new, g_in, g_out, g_new = filename_wizard(test_in, test_out, test_new)
    logger.info(f"Filename Wizard Test Results:")
    logger.info(f"  Base In: {b_in}, Ext In: {e_in}, Gzip In: {g_in}")
    logger.info(f"  Base Out: {b_out}, Ext Out: {e_out}, Gzip Out: {g_out}")
    logger.info(f"  Base New: {b_new}, Ext New: {e_new}, Gzip New: {g_new}")