# -*- coding: utf-8 -*-
"""
autoencoder_eval_3D_patches_v14_cascade.py

Bugfix release for v13. Corrects a ValueError caused by improper argument
passing to the run_autoencoder_3d function.

Key Changes from v13.0:
 • BUGFIX: Corrected the function calls in `process_one_stage` to pass arguments
   individually instead of as a single tuple, resolving the "not enough values
   to unpack" error.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import time
import sys
import logging
import math
from pathlib import Path
from itertools import zip_longest

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import SimpleITK as sitk
from tqdm import tqdm
import inspect

# --- Dependency Imports ---
try:
    from omegaconf import OmegaConf
except ImportError:
    print("ERROR: OmegaConf not found. Please install it: pip install omegaconf")
    sys.exit(1)

try:
    from taming.models.autoencoders import AutoencoderKL3D, VQModel3D, AutoencoderKL3DFiLM_BiCond
except ImportError as e:
    print(f"ERROR: Could not import Taming 3D models: {e}\n"
          "Please ensure the 'taming-transformers-rom1504' or equivalent library is installed.")
    sys.exit(1)

try:
    from data_loader.load_neuroimaging_data_final import OrigDataThickPatches
    # ==================== FIX #1: Correct the import statement =======================
    from utils_v2 import arguments_setup_ae, model_loading_wizard_ae_v2
except ImportError as e:
    print(f"ERROR: Missing local modules: {e}\n"
          "Ensure 'data_loader/load_neuroimaging_data_final.py' and 'utils_v2.py' are on the PYTHONPATH.")
    sys.exit(1)

# ---------------------------- Logging ----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger("eval_ae_3d_v14_cascade")

HELPTEXT = (
    "v14: 3D Autoencoder Evaluation with optional saving of all intermediate cascade stages."
)

# ---------------------------- Utilities ----------------------------
def get_window_weights(patch_shape, window_type='triangular'):
    if not isinstance(patch_shape, (tuple, list)) or len(patch_shape) != 3:
        raise ValueError(f"patch_shape must be (D,H,W), got {patch_shape}")
    if window_type not in ['triangular', 'hann']:
        raise ValueError(f"Unsupported window_type '{window_type}'.")
    dims = patch_shape
    ws = []
    for dim_size in dims:
        if dim_size <= 1:
            w = torch.ones(dim_size, dtype=torch.float32)
        else:
            coords = torch.arange(dim_size, dtype=torch.float32)
            if window_type == 'triangular':
                center = (dim_size - 1.0) / 2.0
                dist = torch.abs(coords - center)
                half = dim_size / 2.0
                w = torch.clamp(1.0 - dist / half, min=0.0)
            else:  # hann
                w = 0.5 * (1.0 - torch.cos(2.0 * math.pi * coords / (dim_size - 1.0)))
        ws.append(w)
    w3 = ws[0][:, None, None] * ws[1][None, :, None] * ws[2][None, None, :]
    mx = torch.max(w3)
    return w3 / mx if float(mx) > 1e-8 else w3

def _compute_cascade_schedule(total_whd, max_step):
    def steps_for_axis(F, ms):
        F = float(F)
        if F <= 1.0: return [1.0]
        steps = []
        remaining = F
        while remaining > ms * (1 + 1e-8):
            steps.append(ms)
            remaining /= ms
        steps.append(remaining)
        return steps
    sw = steps_for_axis(total_whd[0], max_step)
    sh = steps_for_axis(total_whd[1], max_step)
    sd = steps_for_axis(total_whd[2], max_step)
    return [(float(w), float(h), float(d)) for w, h, d in zip_longest(sw, sh, sd, fillvalue=1.0)]

def _resample_sitk_image(img, scale_whd, order):
    cur_spacing = np.array(list(img.GetSpacing()))
    new_spacing = cur_spacing / np.array(scale_whd)
    cur_size = np.array(list(img.GetSize()))
    new_size = tuple(int(round(s * f)) for s, f in zip(cur_size, scale_whd))
    interp_map = {0: sitk.sitkNearestNeighbor, 1: sitk.sitkLinear, 3: sitk.sitkBSpline}
    return sitk.Resample(img, new_size, sitk.Transform(), interp_map.get(order, sitk.sitkLinear),
                         img.GetOrigin(), tuple(new_spacing.tolist()), img.GetDirection(), 0, img.GetPixelID())

# ---------------------------- Arg Parsing ----------------------------

def options_parse():
    p = argparse.ArgumentParser(description=HELPTEXT, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--in_name', type=str, required=True, help='Path to input 3D volume.')
    p.add_argument('--out_name', type=str, required=True, help='Path for the final output file (or base directory name if saving intermediates).')
    p.add_argument('--iname_new', type=str, default=None, help='If set, save initial preprocessed input here.')
    p.add_argument('--skip_existing', action='store_true', default=False, help='Skip if final output already exists.')
    p.add_argument('--save_intermediate_stages', action='store_true', default=False, help='If set, saves all intermediate cascade files in a structured output folder.')
    p.add_argument('--uf_w', type=float, default=1.0)
    p.add_argument('--uf_h', type=float, default=1.0)
    p.add_argument('--uf_z', type=float, default=1.0)
    p.add_argument('--order', type=int, default=1, help='Resample interp: 0=Nearest, 1=Linear, 3=BSpline.')
    p.add_argument('--robust_rescale_input', action='store_true', default=True)
    p.add_argument('--use_scipy', action='store_true', default=False, help='(Compatibility) Dummy argument.')
    p.add_argument('--cascade_max_step', type=float, default=2.0)
    g_cas = p.add_mutually_exclusive_group()
    g_cas.add_argument('--enable_cascade', dest='enable_cascade', action='store_true')
    g_cas.add_argument('--disable_cascade', dest='enable_cascade', action='store_false')
    p.set_defaults(enable_cascade=True)
    p.add_argument('--intensity_range_mode', type=int, default=2, choices=[0, 1, 2])
    p.add_argument('--net2out_rescaling', default='clamp', choices=['clamp', 'linear'])
    p.add_argument('--histogram_matching_mode', type=str, default='none', choices=['nonlinear', 'linear', 'none'])
    p.add_argument('--name', type=str, required=True, help='Model identifier for loading.')
    p.add_argument('--model_path', type=str, default=None)
    p.add_argument('--use_amp', action='store_true', default=False)
    p.add_argument('--no_cuda', action='store_true', default=False)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--max_patch_size', type=str, default='160,32,160')
    p.add_argument('--stride', type=str, default='80,16,80')
    p.add_argument('--merging_method', type=str, default='soft', choices=['average', 'soft'])
    p.add_argument('--window_type', type=str, default='triangular', choices=['triangular', 'hann'])
    p.add_argument('--multi_orientation_avg', action='store_true', default=True)
    p.add_argument('--use_ema', action='store_true', default=True)
    p.add_argument('--proc_time_txt', type=str, default=None)
    args = p.parse_args()
    try:
        args.max_patch_size = tuple(int(v) for v in args.max_patch_size.split(','))
        args.stride = tuple(int(v) for v in args.stride.split(','))
    except Exception as e:
        p.error(f"--max_patch_size/--stride must be 'D,H,W' integers: {e}")
    if not args.no_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda'); logger.info(f"CUDA: {torch.cuda.get_device_name(0)}")
    else:
        args.device = torch.device('cpu'); logger.info("Using CPU.")
    try:
        # Corrected function call
        args = arguments_setup_ae(args)
    except Exception as e:
        logger.error(f"arguments_setup_ae failed: {e}"); sys.exit(1)
    return args

# ---------------------------- Inference (single orientation) ----------------------------
def run_autoencoder_3d(img_filename, orig_data_normalized_dhw, params_model, model, logger,
                       args_obj, is_film_model, source_zooms_list_whd, target_zooms_list_whd,
                       patch_shape_dhw, stride_dhw):
    D, H, W = orig_data_normalized_dhw.shape
    dataset = OrigDataThickPatches(img_filename, orig_data_normalized_dhw, max_patch_size=patch_shape_dhw, stride=stride_dhw)
    loader = DataLoader(dataset, batch_size=args_obj.batch_size, shuffle=False, num_workers=0, pin_memory=(args_obj.device.type == 'cuda'))
    padded_shape = dataset.volume.shape
    output_accum = torch.zeros(padded_shape, dtype=torch.float32, device='cpu')
    count_accum = torch.zeros(padded_shape, dtype=torch.float32, device='cpu')
    weight_mask = get_window_weights(patch_shape_dhw, args_obj.window_type).cpu() if args_obj.merging_method == 'soft' else torch.ones(patch_shape_dhw, dtype=torch.float32).cpu()
    src_z_tensor = tgt_z_tensor = None
    if is_film_model:
        if source_zooms_list_whd is None or target_zooms_list_whd is None: raise ValueError("FiLM model requires zooms.")
        src_z_tensor = torch.tensor(source_zooms_list_whd, dtype=torch.float32, device=args_obj.device).unsqueeze(0)
        tgt_z_tensor = torch.tensor(target_zooms_list_whd, dtype=torch.float32, device=args_obj.device).unsqueeze(0)
    sig = inspect.signature(model.module.forward if hasattr(model, 'module') else model.forward)
    needs_source_cond = 'source_cond_input' in sig.parameters
    needs_target_cond = 'target_cond_input' in sig.parameters
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(loader), desc="Patches", file=sys.stdout, ncols=98, leave=False)
        for batch in loader:
            patches = batch['patch'].to(args_obj.device)
            d_idx, h_idx, w_idx = batch['indices'][0].int(), batch['indices'][1].int(), batch['indices'][2].int()
            B = patches.shape[0]
            fwd_kwargs = {}
            if is_film_model:
                if needs_source_cond: fwd_kwargs['source_cond_input'] = src_z_tensor.expand(B, -1)
                if needs_target_cond: fwd_kwargs['target_cond_input'] = tgt_z_tensor.expand(B, -1)
            with torch.amp.autocast(device_type=args_obj.device.type, enabled=args_obj.use_amp):
                preds, *_ = model(patches, **fwd_kwargs)
            preds = preds.float().cpu()
            pD, pH, pW = patch_shape_dhw
            for b in range(B):
                ds, hs, ws = int(d_idx[b]), int(h_idx[b]), int(w_idx[b])
                sd, sh, sw = slice(ds, ds + pD), slice(hs, hs + pH), slice(ws, ws + pW)
                output_accum[sd, sh, sw] += preds[b, 0] * weight_mask
                count_accum[sd, sh, sw] += weight_mask
            if args_obj.device.type == 'cuda':
                allocated = torch.cuda.memory_allocated(0)/(1024**3); reserved = torch.cuda.memory_reserved(0)/(1024**3)
                pbar.set_postfix(GPU_GB=f"{allocated:.2f}A/{reserved:.2f}R", refresh=False)
            pbar.update(1)
        pbar.close()
    eps = 1e-8; out = output_accum / torch.clamp(count_accum, min=eps); out[count_accum < eps] = 0.0
    return out[:D, :H, :W]

# ---------------------------- Stage Processing Logic ----------------------------

def process_one_stage(current_input_img, stage_scale_whd, args, model, params_model, is_film):
    logger.info(f"Resampling for stage with factors {stage_scale_whd}...")
    stage_input_resampled = _resample_sitk_image(current_input_img, stage_scale_whd, args.order)
    resampled_np = sitk.GetArrayFromImage(stage_input_resampled)
    resampled_min, resampled_max = resampled_np.min(), resampled_np.max()
    denom = resampled_max - resampled_min
    norm_np = np.zeros_like(resampled_np, dtype=np.float32) if denom < 1e-8 else ((resampled_np.astype(np.float32) - resampled_min) / denom) * 2.0 - 1.0
    source_zooms_whd = list(current_input_img.GetSpacing())
    target_zooms_whd = list(stage_input_resampled.GetSpacing())
    if args.multi_orientation_avg:
        data_perms, zoom_perms, inv_perms = [(0,1,2),(1,0,2),(2,1,0)], [(0,1,2),(0,2,1),(2,1,0)], [(0,1,2),(1,0,2),(2,1,0)]
        results_cpu = []
        for i, (dp, zp, ip) in enumerate(zip(data_perms, zoom_perms, inv_perms)):
            rotated_data = np.transpose(norm_np, dp)
            src_zooms = [source_zooms_whd[k] for k in zp] if is_film else None
            tgt_zooms = [target_zooms_whd[k] for k in zp] if is_film else None
            out_rot = run_autoencoder_3d(
                "stage_inference", torch.from_numpy(rotated_data.copy()),
                params_model, model, logger, args, is_film,
                src_zooms, tgt_zooms, args.max_patch_size, args.stride
            )
            results_cpu.append(torch.from_numpy(np.transpose(out_rot.cpu().numpy(), ip).copy()))
        out_tensor = torch.stack(results_cpu).mean(dim=0).float()
    else:
        out_tensor = run_autoencoder_3d(
            "stage_inference", torch.from_numpy(norm_np.copy()),
            params_model, model, logger, args, is_film,
            source_zooms_whd, target_zooms_whd, args.max_patch_size, args.stride
        )
    out_tensor = out_tensor.cpu()
    mn, mx = out_tensor.min(), out_tensor.max()
    if (mx - mn).abs() < 1e-6:
        out_np = np.zeros_like(resampled_np, dtype=np.float32)
    else:
        scaled_01 = (out_tensor - mn) / (mx - mn) if args.net2out_rescaling == "linear" else (torch.clamp(out_tensor, -1, 1) + 1) / 2
        out_np_float = (scaled_01 * (resampled_max - resampled_min) + resampled_min).numpy()
        try:
            original_dtype = resampled_np.dtype
            if np.issubdtype(original_dtype, np.integer):
                info = np.iinfo(original_dtype); out_np_float = np.clip(np.round(out_np_float), info.min, info.max)
            out_np = out_np_float.astype(original_dtype)
        except Exception: out_np = out_np_float.astype(np.float32)
    stage_output_sitk = sitk.GetImageFromArray(out_np)
    stage_output_sitk.CopyInformation(stage_input_resampled)
    if args.histogram_matching_mode in ['nonlinear', 'linear']:
        try:
            source_float = sitk.Cast(stage_output_sitk, sitk.sitkFloat32); ref_float = sitk.Cast(stage_input_resampled, sitk.sitkFloat32)
            if args.histogram_matching_mode == 'nonlinear':
                matcher = sitk.HistogramMatchingImageFilter(); matcher.SetNumberOfHistogramLevels(1024); matcher.SetNumberOfMatchPoints(7); matcher.ThresholdAtMeanIntensityOn()
                matched_float = matcher.Execute(source_float, ref_float)
            else:
                stats = sitk.StatisticsImageFilter(); stats.Execute(source_float); mean_src, std_src = stats.GetMean(), stats.GetSigma()
                stats.Execute(ref_float); mean_ref, std_ref = stats.GetMean(), stats.GetSigma()
                matched_float = ((source_float - mean_src) / (std_src or 1e-8) * std_ref + mean_ref)
            stage_output_sitk = sitk.Cast(matched_float, stage_output_sitk.GetPixelID())
        except Exception as e: logger.warning(f"Histogram matching for stage failed: {e}")
    return stage_output_sitk

# ---------------------------- Main ----------------------------
def main():
    args = options_parse()
    start_total = time.time()
    in_path = Path(args.in_name).resolve()
    out_path = Path(args.out_name).resolve()

    # --- Setup output directories ---
    if args.save_intermediate_stages:
        base_out_dir = out_path.parent / out_path.stem
        intermediate_dir = base_out_dir / "intermediate"
        final_dir = base_out_dir / "final"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)
        final_output_path = final_dir / out_path.name
        logger.info(f"Saving intermediate stages is ON. Final output will be in: {final_dir}")
    else:
        final_output_path = out_path
        intermediate_dir = out_path.parent
    
    if args.skip_existing and final_output_path.exists():
        logger.info(f"Final output exists; skipping: {final_output_path}")
        return

    # --- Load Model ---
    try:
        model, params_model, is_film = model_loading_wizard_ae_v2(args, logger)
        model.to(args.device)
        logger.info(f"Model loaded successfully. FiLM={is_film}")
    except Exception as e:
        logger.exception(f"Model loading failed: {e}"); sys.exit(1)

    # --- Pre-processing ---
    current_img = sitk.ReadImage(str(in_path))
    if args.robust_rescale_input:
        logger.info("Applying robust intensity rescaling to initial image...")
        img_np = sitk.GetArrayFromImage(current_img)
        p_low, p_high = np.percentile(img_np, [0.2, 99.8])
        img_np = np.clip(img_np, p_low, p_high)
        rescaled_img = sitk.GetImageFromArray(img_np)
        rescaled_img.CopyInformation(current_img)
        current_img = rescaled_img

    # --- Cascade Processing Loop ---
    total_scale = (float(args.uf_w), float(args.uf_h), float(args.uf_z))
    schedule = _compute_cascade_schedule(total_scale, args.cascade_max_step)
    logger.info(f"Processing with {len(schedule)} cascade stage(s): {schedule}")
    
    # Pre-calculate the final geometry based on the original image
    original_shape_whd = current_img.GetSize()
    original_zooms_whd = current_img.GetSpacing()
    final_target_shape_whd = tuple(int(round(s * f)) for s, f in zip(original_shape_whd, total_scale))
    final_target_zooms_whd = tuple(spc / f for spc, f in zip(original_zooms_whd, total_scale))
    
    intermediate_files_to_delete = []
    current_input_path = in_path

    for i, stage_scale in enumerate(schedule, 1):
        # Determine output path for this stage
        if i == len(schedule):
            stage_output_path = final_output_path
        else:
            base, ext = out_path.stem, out_path.name[len(out_path.stem):]
            stage_filename = f"{base}_stage_{i}_of_{len(schedule)}{ext}"
            stage_output_path = intermediate_dir / stage_filename

        # --- Calculate and log shapes and zooms for this stage ---
        input_shape_whd = current_img.GetSize()
        input_zooms_whd = current_img.GetSpacing()
        target_shape_whd = tuple(int(round(s * f)) for s, f in zip(input_shape_whd, stage_scale))
        target_zooms_whd = tuple(spc / f for spc, f in zip(input_zooms_whd, stage_scale))

        logger.info("\n" + "="*80)
        logger.info(f"STARTING CASCADE STAGE {i}/{len(schedule)}")
        logger.info(f"  Input File:            {current_input_path.name}")
        logger.info(f"  Output File:           {stage_output_path.name}")
        logger.info(f"  Upsampling Factors (W,H,D): {stage_scale}")
        logger.info(f"  Stage Input Geometry:")
        logger.info(f"    Shape (W,H,D):       {input_shape_whd}")
        logger.info(f"    Zooms (mm):          ({input_zooms_whd[0]:.4f}, {input_zooms_whd[1]:.4f}, {input_zooms_whd[2]:.4f})")
        logger.info(f"  Stage Target Geometry:")
        logger.info(f"    Shape (W,H,D):       {target_shape_whd}")
        logger.info(f"    Zooms (mm):          ({target_zooms_whd[0]:.4f}, {target_zooms_whd[1]:.4f}, {target_zooms_whd[2]:.4f})")
        
        # Add final geometry information if it's a multi-stage process
        if len(schedule) > 1:
            logger.info(f"  Final Target Geometry (all stages):")
            logger.info(f"    Shape (W,H,D):       {final_target_shape_whd}")
            logger.info(f"    Zooms (mm):          ({final_target_zooms_whd[0]:.4f}, {final_target_zooms_whd[1]:.4f}, {final_target_zooms_whd[2]:.4f})")

        # Process the stage
        stage_output_img = process_one_stage(current_img, stage_scale, args, model, params_model, is_film)
        sitk.WriteImage(stage_output_img, str(stage_output_path))
        logger.info(f"Successfully wrote stage {i} output.")

        # Update variables for the next loop iteration
        previous_input_path = current_input_path
        current_input_path = stage_output_path
        current_img = stage_output_img
        
        if previous_input_path != in_path:
            intermediate_files_to_delete.append(previous_input_path)
            
    logger.info("\n" + "="*80)

    # --- Cleanup ---
    if not args.save_intermediate_stages and intermediate_files_to_delete:
        logger.info("Cascade complete. Cleaning up temporary intermediate files...")
        for f in intermediate_files_to_delete:
            try:
                f.unlink()
                logger.info(f"Deleted: {f}")
            except OSError as e:
                logger.warning(f"Could not delete {f}: {e}")
    elif args.save_intermediate_stages:
        logger.info(f"Cascade complete. Intermediate stages saved in: {intermediate_dir}")
    
    logger.info(f"Final output is located at: {final_output_path}")
    logger.info(f"Total end-to-end script time: {time.time() - start_total:.2f}s")


if __name__ == "__main__":
    main()