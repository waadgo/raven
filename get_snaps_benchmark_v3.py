#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subject-level evaluation & visualization (centralized results subfolder)

- Scans methods under processed/<degradation>/<factor>/
- For each method:
    * If NOT cached (or --force): resample & reorient, compute MAE/PSNR/SSIM/LPIPS (masked),
      save 3D maps + raw/masked 2D slices, cache under:
      <subject>/<results_dirname>/per_method/<deg>/<factor>/<method>/
    * If cached and NOT --force: do NOT recompute per-method metrics or regenerate per-method 2D PNGs.
- Always builds cross-method grids (supports multiple planes/positions) with 8 rows:
    0) Native LR (from degraded) repeated
    1) GT HR (from original) repeated
    2) Method image (upsampled) per method
    3) LPIPS map (masked), shared scale across methods, colorbar ONLY on the rightmost applicable column
    4) MAE map (masked), shared scale across methods, colorbar ONLY on the rightmost column
    5) GT HR + GT mask (merged 8 ROI) — fixed label colors
    6) Method image + seg_<seg_method> (merged 8 ROI) — fixed label colors
    7) Method image + segmentation ERROR overlay (pred≠GT) — semi-transparent red

- Column order (if present): RAVEN, ArSSR-RDN, SuperFormer, ArSSR-ResCNN, NLMUPSAMPLE, Trilinear
  (mapped from folder names: raven_f2_nfg64, arssr_rdn, superformer, arssr_rescnn, nlmupsample, trilinear)

- Masked display control:
    * --grid_use_mask (default): Every grid row is multiplied by the foreground mask (background hidden).
    * --no-grid_use_mask: Show unmasked images in the grid.

- Writes two CSVs at <subject>/<results_dirname>/csv/ :
    * subject_reconstruction_metrics.csv  (from cached or newly computed metrics.json)
    * subject_segmentation_agreement.csv  (8 ROI DICE/IOU vs GT mask)

Display orientation (always enforced for 2D):
- Axial   (z-fixed): top=Anterior, bottom=Posterior, right=Right, left=Left
- Sagittal(x-fixed): top=Superior, bottom=Inferior, right=Anterior, left=Posterior
- Coronal (y-fixed): top=Superior, bottom=Inferior, right=Right, left=Left
"""
from __future__ import annotations
import argparse, logging, sys, math, json, csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

# --- Lazy deps
try:
    import SimpleITK as sitk
except ImportError:
    print("ERROR: SimpleITK is not installed. Please run 'pip install SimpleITK-SimpleElastix'", file=sys.stderr)
    sys.exit(1)

try:
    from skimage.metrics import structural_similarity as sk_ssim
except ImportError:
    sk_ssim = None

try:
    import torch, lpips
except ImportError:
    torch = None
    lpips = None

import matplotlib.pyplot as plt
from matplotlib import gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib import colors as mcolors
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# --------------------- Logging ----------------------
def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        stream=sys.stdout,
    )


# --------------------- I/O helpers ----------------------
def robust_read_float(p: Path) -> Optional["sitk.Image"]:
    try:
        return sitk.ReadImage(str(p), sitk.sitkFloat64)
    except Exception as e:
        logging.error("Failed to read image: %s (%s)", p, e)
        return None

def robust_read_mask_img(p: Path) -> Optional["sitk.Image"]:
    try:
        return sitk.ReadImage(str(p), sitk.sitkInt32)  # preserve labels
    except Exception as e:
        logging.error("Failed to read seg/mask: %s (%s)", p, e)
        return None

def resample_like(mov: "sitk.Image", ref: "sitk.Image", is_mask: bool = False) -> "sitk.Image":
    rf = sitk.ResampleImageFilter()
    rf.SetReferenceImage(ref)
    rf.SetDefaultPixelValue(0)
    rf.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    return rf.Execute(mov)

def to_np(img: "sitk.Image", dtype=np.float64) -> np.ndarray:
    return sitk.GetArrayFromImage(img).astype(dtype, copy=False)

def np_to_like_img(vol: np.ndarray, ref_img: "sitk.Image", pixel_type=sitk.sitkFloat32) -> "sitk.Image":
    out = sitk.GetImageFromArray(vol.astype(np.float32, copy=False))
    out.CopyInformation(ref_img)
    return sitk.Cast(out, pixel_type)

def write_nifti_from_np(vol: np.ndarray, ref_img: "sitk.Image", out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(np_to_like_img(vol, ref_img), str(out_path), useCompression=True)
    logging.info("Saved: %s", out_path)

def orient_to_RAS(img: "sitk.Image") -> "sitk.Image":
    """Reorient to RAS (+X=R, +Y=A, +Z=S) for consistent display."""
    try:
        return sitk.DICOMOrient(img, 'RAS')
    except Exception:
        f = sitk.DICOMOrientImageFilter()
        f.SetDesiredCoordinateOrientation('RAS')
        return f.Execute(img)


# --------------------- Normalization ----------------------
def _minmax01(vol: np.ndarray) -> np.ndarray:
    vmin, vmax = float(np.nanmin(vol)), float(np.nanmax(vol))
    den = max(vmax - vmin, 1e-12)
    return ((vol - vmin) / den).astype(np.float32)

def _robust01(vol: np.ndarray, mask: Optional[np.ndarray], p_lo: float, p_hi: float) -> np.ndarray:
    x = vol
    if mask is not None:
        m = (mask > 0) & np.isfinite(x)
        x = x[m] if np.any(m) else x[np.isfinite(x)]
    else:
        x = x[np.isfinite(x)]
    if x.size == 0:
        return _minmax01(vol)
    lo = float(np.percentile(x, p_lo))
    hi = float(np.percentile(x, p_hi))
    den = max(hi - lo, 1e-12)
    return ((np.clip(vol, lo, hi) - lo) / den).astype(np.float32)


# --------------------- Metrics ----------------------
def psnr_voxel_map(gt01: np.ndarray, pr01: np.ndarray, max_val: float = 1.0) -> np.ndarray:
    eps = 1e-12
    se = (pr01 - gt01) ** 2
    denom = np.sqrt(np.maximum(se, eps))
    psnr = 20.0 * np.log10(max_val / denom)
    psnr[se < eps] = 0.0
    return psnr.astype(np.float32)

class LPIPSSlicer:
    def __init__(self, device: str = "cuda"):
        if torch is None or lpips is None:
            raise ImportError("LPIPS requires 'torch' and 'lpips'.")
        self.t = torch
        self.device = device
        self.model = lpips.LPIPS(net="alex", spatial=True).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _slice(arr, axis, i):
        if axis == 0: return arr[i, :, :]
        if axis == 1: return arr[:, i, :]
        return arr[:, :, i]

    def mean_score(self, gt01: np.ndarray, pr01: np.ndarray, mask: np.ndarray) -> float:
        vals: List[float] = []
        for axis in (0, 1, 2):
            n = gt01.shape[axis]
            for i in range(n):
                m2 = self._slice(mask, axis, i).astype(bool)
                if not np.any(m2): continue
                g2 = self._slice(gt01, axis, i).astype(np.float32, copy=False)
                p2 = self._slice(pr01, axis, i).astype(np.float32, copy=False)
                g_t = self.t.from_numpy(g2)[None, None].repeat(1, 3, 1, 1).to(self.device)
                p_t = self.t.from_numpy(p2)[None, None].repeat(1, 3, 1, 1).to(self.device)
                with self.t.no_grad():
                    d = self.model(p_t, g_t, normalize=True).detach().cpu().squeeze().numpy()
                roi = d[:m2.shape[0], :m2.shape[1]][m2]
                roi = roi[np.isfinite(roi)]
                if roi.size: vals.append(float(np.mean(roi)))
        return float(np.nanmean(vals)) if vals else float("nan")

    def map3d(self, gt01: np.ndarray, pr01: np.ndarray) -> np.ndarray:
        out = np.zeros_like(gt01, dtype=np.float32)
        cnt = np.zeros_like(gt01, dtype=np.float32)
        for axis in (0, 1, 2):
            n = gt01.shape[axis]
            for i in range(n):
                g2 = self._slice(gt01, axis, i).astype(np.float32, copy=False)
                p2 = self._slice(pr01, axis, i).astype(np.float32, copy=False)
                g_t = self.t.from_numpy(g2)[None, None].repeat(1, 3, 1, 1).to(self.device)
                p_t = self.t.from_numpy(p2)[None, None].repeat(1, 3, 1, 1).to(self.device)
                with self.t.no_grad():
                    d = self.model(p_t, g_t, normalize=True).detach().cpu().squeeze().numpy().astype(np.float32)
                if axis == 0:
                    out[i, :d.shape[0], :d.shape[1]] += d; cnt[i, :d.shape[0], :d.shape[1]] += 1
                elif axis == 1:
                    out[:d.shape[0], i, :d.shape[1]] += d; cnt[:d.shape[0], i, :d.shape[1]] += 1
                else:
                    out[:d.shape[0], :d.shape[1], i] += d; cnt[:d.shape[0], :d.shape[1], i] += 1
        cnt[cnt == 0] = 1.0
        return (out / cnt).astype(np.float32)


# --------------------- Metric Orchestration ----------------------
def normalize_pair(gt_np: np.ndarray, pr_np: np.ndarray, mask_np: np.ndarray,
                   norm_mode: str, p_lo: float, p_hi: float) -> Tuple[np.ndarray, np.ndarray]:
    if norm_mode == "minmax":
        return _minmax01(gt_np), _minmax01(pr_np)
    elif norm_mode == "robust":
        return _robust01(gt_np, None, p_lo, p_hi), _robust01(pr_np, None, p_lo, p_hi)
    else:
        return _robust01(gt_np, mask_np, p_lo, p_hi), _robust01(pr_np, mask_np, p_lo, p_hi)

def calculate_all_metrics(gt_np: np.ndarray, pr_np: np.ndarray, mask_np: np.ndarray,
                          do_lpips: bool, lpips_dev: str, lpips_helper: Optional[LPIPSSlicer],
                          norm_mode: str, p_lo: float, p_hi: float) -> Dict[str, float]:
    out = {"MAE": np.nan, "PSNR": np.nan, "SSIM": np.nan, "LPIPS": np.nan}
    mask = mask_np.astype(bool)
    if not np.any(mask):
        logging.error("Mask is empty. Cannot compute metrics.")
        return out

    logging.info("Normalizing with '%s' (%.1f/%.1f percentiles).", norm_mode, p_lo, p_hi)
    gt01, pr01 = normalize_pair(gt_np, pr_np, mask_np, norm_mode, p_lo, p_hi)

    mae_map = np.abs(pr01 - gt01)
    vals = mae_map[mask]; vals = vals[np.isfinite(vals)]
    out["MAE"] = float(np.mean(vals)) if vals.size else float("nan")

    psnr_map = psnr_voxel_map(gt01, pr01, max_val=1.0)
    vals = psnr_map[mask]; vals = vals[np.isfinite(vals)]
    out["PSNR"] = float(np.mean(vals)) if vals.size else float("nan")

    if sk_ssim is not None:
        ssim_vals: List[float] = []
        def sl2(arr, axis, i):
            if axis == 0: return arr[i, :, :]
            if axis == 1: return arr[:, i, :]
            return arr[:, :, i]
        for axis in (0, 1, 2):
            n = gt01.shape[axis]
            for i in range(n):
                m2 = sl2(mask, axis, i)
                if not np.any(m2): continue
                g2 = sl2(gt01, axis, i); p2 = sl2(pr01, axis, i)
                try:
                    _, ssim_map = sk_ssim(g2, p2, data_range=1.0, full=True,
                                          gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
                    roi = ssim_map[:m2.shape[0], :m2.shape[1]][m2]
                    roi = roi[np.isfinite(roi)]
                    if roi.size: ssim_vals.append(float(np.mean(roi)))
                except Exception as e:
                    logging.warning("SSIM slice failed axis=%d i=%d: %s", axis, i, e)
        out["SSIM"] = float(np.mean(ssim_vals)) if ssim_vals else float("nan")
    else:
        logging.warning("scikit-image not installed; SSIM=NaN.")

    if do_lpips:
        try:
            if lpips_helper is None:
                lpips_helper = LPIPSSlicer(device=lpips_dev)
            out["LPIPS"] = lpips_helper.mean_score(gt01, pr01, mask.astype(np.uint8))
        except Exception as e:
            logging.error("LPIPS failed: %s", e)
            out["LPIPS"] = float("nan")

    return out


# --------------------- Utilities: planes/positions ----------------------
def _parse_positions_percent_or_unit(s: str) -> List[float]:
    out: List[float] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok: continue
        v = float(tok)
        if v > 1.0:
            v = v / 100.0
        out.append(max(0.0, min(1.0, v)))
    return out

def _parse_planes(s: str) -> List[str]:
    planes = []
    for tok in s.split(","):
        t = tok.strip().lower()
        if t in ("axial","coronal","sagittal"):
            planes.append(t)
    return planes or ["axial"]

def _get_slice(arr: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0: return arr[idx, :, :]   # axial (z-fixed) => (Y,X)
    if axis == 1: return arr[:, idx, :]   # coronal (y-fixed) => (Z,X)
    return arr[:, :, idx]                 # sagittal (x-fixed) => (Z,Y)


# --------------------- ROI Merging & Metrics ----------------------
LABELS = {
    0: "background",
    2: "left cerebral white matter", 3: "left cerebral cortex",
    4: "left lateral ventricle", 5: "left inferior lateral ventricle",
    7: "left cerebellum white matter", 8: "left cerebellum cortex",
    10: "left thalamus", 11: "left caudate", 12: "left putamen", 13: "left pallidum",
    14: "3rd ventricle", 15: "4th ventricle", 16: "brain-stem",
    17: "left hippocampus", 18: "left amygdala",
    24: "CSF", 26: "left accumbens area", 28: "left ventral DC",
    41: "right cerebral white matter", 42: "right cerebral cortex",
    43: "right lateral ventricle", 44: "right inferior lateral ventricle",
    46: "right cerebellum white matter", 47: "right cerebellum cortex",
    49: "right thalamus", 50: "right caudate", 51: "right putamen",
    52: "right pallidum", 53: "right hippocampus", 54: "right amygdala",
    58: "right accumbens area", 60: "right ventral DC",
    77: "WM extra (grouping only)"
}

S_CEREBRAL_GM = {3, 42}
S_CEREBRAL_WM = {2, 41, 77}
S_HIPPOCAMPUS = {17, 53}
S_VENTRICLES  = {4, 5, 14, 15, 43, 44}
S_DEEP_GM     = {10, 11, 12, 13, 18, 26, 28, 49, 50, 51, 52, 54, 58, 60}
S_CBL_GM      = {8, 47}
S_CBL_WM      = {7, 46}
S_BRAINSTEM   = {16}  # 8th ROI

# Fixed ROI order for stable colors:
ROI_NAMES: List[str] = [
    "cerebral_gm",
    "cerebral_wm",
    "hippocampus",
    "ventricles",
    "deep_gm",
    "cbl_gm",
    "cbl_wm",
    "brainstem",
]

ROI_GROUPS: Dict[str, set] = {
    "cerebral_gm": S_CEREBRAL_GM,
    "cerebral_wm": S_CEREBRAL_WM,
    "hippocampus": S_HIPPOCAMPUS,
    "ventricles":  S_VENTRICLES,
    "deep_gm":     S_DEEP_GM,
    "cbl_gm":      S_CBL_GM,
    "cbl_wm":      S_CBL_WM,
    "brainstem":   S_BRAINSTEM,
}

ROI_COLORS = {
    "cerebral_gm": (0.89, 0.10, 0.11),
    "cerebral_wm": (0.17, 0.63, 0.17),
    "hippocampus": (0.12, 0.47, 0.71),
    "ventricles":  (0.60, 0.31, 0.64),
    "deep_gm":     (1.00, 0.50, 0.05),
    "cbl_gm":      (0.65, 0.34, 0.16),
    "cbl_wm":      (0.58, 0.82, 0.31),
    "brainstem":   (0.55, 0.55, 0.55),
}

def merge_to_8rois(seg_np: np.ndarray) -> np.ndarray:
    out = np.zeros_like(seg_np, dtype=np.uint8)
    for k, name in enumerate(ROI_NAMES, start=1):
        ids = ROI_GROUPS[name]
        mask = np.isin(seg_np, list(ids))
        out[mask] = k
    return out

def dice_iou(bin_pred: np.ndarray, bin_gt: np.ndarray) -> Tuple[float, float]:
    bin_pred = bin_pred.astype(bool); bin_gt = bin_gt.astype(bool)
    inter = np.logical_and(bin_pred, bin_gt).sum()
    a = bin_pred.sum(); b = bin_gt.sum()
    dice = (2.0 * inter) / (a + b) if (a + b) > 0 else (1.0 if a == b else 0.0)
    union = a + b - inter
    iou = inter / union if union > 0 else (1.0 if a == b else 0.0)
    return float(dice), float(iou)


# --------------------- Per-method computation (only when uncached / forced) ----------------------
def compute_and_save_for_method(
    method_name: str,
    gt_img: "sitk.Image",
    test_img: "sitk.Image",
    brain_mask_img: "sitk.Image",
    out_root: Path,
    with_lpips: bool,
    lpips_dev: str,
    norm_mode: str, p_lo: float, p_hi: float,
    err_cmap: str
) -> Dict[str, float]:
    out_root.mkdir(parents=True, exist_ok=True)
    nii_dir      = out_root / "nii_maps"
    metrics_json = out_root / "metrics.json"

    # Reorient & arrays
    gt_img_r = orient_to_RAS(gt_img)
    ts_img_r = orient_to_RAS(test_img)
    mk_img_r = orient_to_RAS(brain_mask_img)

    gt_np = to_np(gt_img_r, dtype=np.float64)
    pr_np = to_np(ts_img_r, dtype=np.float64)
    mk_np = to_np(mk_img_r, dtype=np.int32)

    # LPIPS helper
    lpips_h = None
    if with_lpips:
        try:
            lpips_h = LPIPSSlicer(device=lpips_dev)
            logging.info("LPIPS on %s", lpips_dev)
        except Exception as e:
            logging.error("LPIPS init failed; disabling. %s", e)
            with_lpips = False

    # Metrics
    metrics = calculate_all_metrics(
        gt_np, pr_np, mk_np,
        do_lpips=with_lpips,
        lpips_dev=lpips_dev,
        lpips_helper=lpips_h,
        norm_mode=norm_mode,
        p_lo=p_lo, p_hi=p_hi
    )

    # Visuals & maps (store canonical 30–70% slices)
    gt01, pr01 = normalize_pair(gt_np, pr_np, mk_np, norm_mode, p_lo, p_hi)
    mask_bool = mk_np.astype(bool)

    mae_map  = np.abs(pr01 - gt01).astype(np.float32)
    psnr_map = psnr_voxel_map(gt01, pr01, max_val=1.0).astype(np.float32)
    lp_map   = None
    if with_lpips:
        try:
            lp_map = lpips_h.map3d(gt01, pr01).astype(np.float32)
        except Exception as e:
            logging.error("LPIPS 3D map failed: %s", e)

    slices_raw   = out_root / "slices" / "raw"
    slices_mask  = out_root / "slices" / "masked"
    errors_raw   = out_root / "errors" / "raw"
    errors_mask  = out_root / "errors" / "masked"
    for d in [nii_dir, slices_raw/"gt", slices_raw/"test", slices_mask/"gt", slices_mask/"test",
              errors_raw/"mae", errors_raw/"psnr", errors_mask/"mae", errors_mask/"psnr"]:
        d.mkdir(parents=True, exist_ok=True)
    if lp_map is not None:
        (errors_raw/"lpips").mkdir(parents=True, exist_ok=True)
        (errors_mask/"lpips").mkdir(parents=True, exist_ok=True)

    # ---- ZERO-MARGIN single-slice saver ----
    def _save_slice_png(sl: np.ndarray, out_path: Path, cmap: str):
        fig_w, fig_h = sl.shape[1] / 100, sl.shape[0] / 100
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
        ax.imshow(sl, cmap=cmap, origin='lower', aspect='auto', interpolation='nearest')
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        plt.savefig(str(out_path), bbox_inches='tight', pad_inches=0.0)
        plt.close()

    Z, Y, X = gt01.shape
    for axis, n, tag in ((0, Z, 'z'), (1, Y, 'y'), (2, X, 'x')):
        for pos in [0.3, 0.4, 0.5, 0.6, 0.7]:
            idx = max(0, min(n-1, int(pos*n)))
            sl_mk = _get_slice(mask_bool, axis, idx)
            _save_slice_png(_get_slice(gt01, axis, idx),            slices_raw/"gt"/f"gt_{tag}{int(pos*100)}.png",   'gray')
            _save_slice_png(_get_slice(gt01, axis, idx)*sl_mk,      slices_mask/"gt"/f"gt_{tag}{int(pos*100)}.png",   'gray')
            _save_slice_png(_get_slice(pr01, axis, idx),            slices_raw/"test"/f"test_{tag}{int(pos*100)}.png", 'gray')
            _save_slice_png(_get_slice(pr01, axis, idx)*sl_mk,      slices_mask/"test"/f"test_{tag}{int(pos*100)}.png", 'gray')
            _save_slice_png(_get_slice(mae_map, axis, idx),         errors_raw/"mae"/f"mae_{tag}{int(pos*100)}.png",   'magma')
            _save_slice_png(_get_slice(mae_map, axis, idx)*sl_mk,   errors_mask/"mae"/f"mae_{tag}{int(pos*100)}.png",   'magma')
            _save_slice_png(_get_slice(psnr_map, axis, idx),        errors_raw/"psnr"/f"psnr_{tag}{int(pos*100)}.png",  'magma')
            _save_slice_png(_get_slice(psnr_map, axis, idx)*sl_mk,  errors_mask/"psnr"/f"psnr_{tag}{int(pos*100)}.png", 'magma')
            if lp_map is not None:
                _save_slice_png(_get_slice(lp_map, axis, idx),        errors_raw/"lpips"/f"lpips_{tag}{int(pos*100)}.png",'magma')
                _save_slice_png(_get_slice(lp_map, axis, idx)*sl_mk,  errors_mask/"lpips"/f"lpips_{tag}{int(pos*100)}.png",'magma')

    write_nifti_from_np(mae_map  * mask_bool,  gt_img_r, nii_dir / "mae_map.nii.gz")
    write_nifti_from_np(psnr_map * mask_bool,  gt_img_r, nii_dir / "psnr_map.nii.gz")
    if lp_map is not None:
        write_nifti_from_np(lp_map * mask_bool, gt_img_r, nii_dir / "lpips_map.nii.gz")

    metrics_json.write_text(json.dumps(metrics, indent=2))
    return metrics


# --------------------- Cross-method grid ----------------------
def idx_for_plane_and_pos(shape: Tuple[int,int,int], plane: str, pos: float) -> Tuple[int,int]:
    Z,Y,X = shape
    plane = plane.lower()
    if plane == "axial":
        return 0, max(0, min(Z-1, int(pos*Z)))
    elif plane == "coronal":
        return 1, max(0, min(Y-1, int(pos*Y)))
    elif plane == "sagittal":
        return 2, max(0, min(X-1, int(pos*X)))
    else:
        raise ValueError("plane must be axial|coronal|sagittal")

def make_roi_cmap(alpha: float = 0.55) -> ListedColormap:
    ordered = [ROI_COLORS[n] for n in ROI_NAMES]
    base = [(0,0,0,0.0)] + [(*rgb, alpha) for rgb in ordered]  # 0=transparent bg, 1..8 = fixed ROI colors
    return ListedColormap(base, name="roi8")

# Fixed, discrete normalization for labels 0..8 (prevents auto-rescaling per slice)
LABEL_BOUNDS = [-0.5] + [i + 0.5 for i in range(9)]   # [-0.5, 0.5, 1.5, ..., 8.5]
LABEL_NORM   = mcolors.BoundaryNorm(LABEL_BOUNDS, ncolors=9, clip=True)

# Error overlay colormap: 0 = transparent, 1 = vibrant red with alpha
def make_error_overlay_cmap(alpha: float = 0.6) -> ListedColormap:
    return ListedColormap([(0,0,0,0.0), (1.0, 0.0, 0.0, alpha)], name="err_overlay")

ERR_BOUNDS = [-0.5, 0.5, 1.5]
ERR_NORM   = mcolors.BoundaryNorm(ERR_BOUNDS, ncolors=2, clip=True)

# ---- ZERO-GAP GRID FUNCTION (no panel titles/labels) ----
def draw_cross_method_grid(
    methods: List[str],
    display_names: Dict[str, str],   # kept for future use; not shown
    gt01: np.ndarray,
    pr01_by_method: Dict[str, Dict[str, np.ndarray]],
    mask_bool: np.ndarray,
    mae_map_by_method: Dict[str, np.ndarray],
    lpips_map_by_method: Dict[str, Optional[np.ndarray]],
    seg_overlay_by_method: Dict[str, np.ndarray],
    seg_error_by_method: Dict[str, np.ndarray],
    gt_overlay_merged8: np.ndarray,
    plane: str, pos: float,
    out_path: Path,
    err_cmap: str,
    panel_inches: float = 5.0,
    dpi: int = 220,
    apply_mask: bool = True
):
    axis, idx = idx_for_plane_and_pos(gt01.shape, plane, pos)

    def slice_from(m: np.ndarray) -> np.ndarray:
        if axis == 0: return m[idx, :, :]
        if axis == 1: return m[:, idx, :]
        return m[:, :, idx]

    msl = slice_from(mask_bool)
    def maybe_mask(img2d: np.ndarray) -> np.ndarray:
        return img2d * msl if apply_mask else img2d

    # Shared scales
    lp_indices = [j for j, m in enumerate(methods) if lpips_map_by_method.get(m) is not None]
    last_lp_col = lp_indices[-1] if lp_indices else -1
    mae_slices = [maybe_mask(slice_from(mae_map_by_method[m])) for m in methods]
    mae_vmax = np.nanpercentile(np.stack(mae_slices), 99.0) if mae_slices else 1.0
    lp_vmax = 1.0
    if lp_indices:
        lp_slices = [maybe_mask(slice_from(lpips_map_by_method[m])) for m in methods if lpips_map_by_method[m] is not None]
        lp_vmax = float(np.nanmax(np.stack(lp_slices)))

    gt_sl = maybe_mask(slice_from(gt01))
    gt_overlay_sl = slice_from(gt_overlay_merged8)
    roi_cmap = make_roi_cmap()
    err_overlay_cmap = make_error_overlay_cmap()

    # Figure: 8 rows × len(methods) cols, ZERO spacing
    ncols = len(methods)
    nrows = 8
    fig_h = nrows * panel_inches
    fig_w = ncols * panel_inches
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.0, hspace=0.0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    def imshow_panel(r, c, img, cmap='gray', vmin=None, vmax=None, show_cbar=False, norm=None):
        ax = fig.add_subplot(gs[r, c])
        im = ax.imshow(img, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, norm=norm,
                       interpolation='nearest', aspect='auto')
        ax.set_axis_off()
        if show_cbar:
            # Place colorbar just outside the image space (slightly to the right)
            cax = inset_axes(
                ax,
                width="3%",            # colorbar width (relative)
                height="98%",          # almost full panel height
                loc="lower left",
                bbox_to_anchor=(1.02, 0.01, 1, 1),  # x shift 2% to the right of the axes
                bbox_transform=ax.transAxes,
                borderpad=0.0
            )
            cb = fig.colorbar(im, cax=cax)
            cb.ax.tick_params(length=2, labelsize=8)

    # Row 0: Native LR
    for j, m in enumerate(methods):
        native = pr01_by_method[m]["native"]
        imshow_panel(0, j, maybe_mask(slice_from(native)), cmap='gray')

    # Row 1: GT HR
    for j, m in enumerate(methods):
        imshow_panel(1, j, gt_sl, cmap='gray')

    # Row 2: Method image
    for j, m in enumerate(methods):
        test = pr01_by_method[m]["test"]
        imshow_panel(2, j, maybe_mask(slice_from(test)), cmap='gray')

    # Row 3: LPIPS (masked) — colorbar only on rightmost applicable column (outside)
    for j, m in enumerate(methods):
        lp = lpips_map_by_method.get(m)
        if lp is None:
            blank = np.zeros_like(gt_sl, dtype=np.float32)
            imshow_panel(3, j, blank, cmap=err_cmap, vmin=0.0, vmax=1.0, show_cbar=False)
        else:
            imshow_panel(3, j, maybe_mask(slice_from(lp)),
                         cmap=err_cmap, vmin=0.0, vmax=lp_vmax, show_cbar=(j==last_lp_col))

    # Row 4: MAE (masked) — colorbar only on last column (outside)
    for j, m in enumerate(methods):
        mae = mae_map_by_method[m]
        imshow_panel(4, j, maybe_mask(slice_from(mae)),
                     cmap=err_cmap, vmin=0.0, vmax=float(mae_vmax), show_cbar=(j==ncols-1))

    # Row 5: GT + GT Mask
    for j, m in enumerate(methods):
        ax = fig.add_subplot(gs[5, j])
        ax.imshow(gt_sl, cmap='gray', origin='lower', interpolation='nearest', aspect='auto')
        ax.imshow(gt_overlay_sl, cmap=roi_cmap, norm=LABEL_NORM, origin='lower',
                  interpolation='nearest', aspect='auto')
        ax.set_axis_off()

    # Row 6: Method + Method Seg
    for j, m in enumerate(methods):
        base = pr01_by_method[m]["test"]
        seg  = seg_overlay_by_method[m]
        base_sl = maybe_mask(slice_from(base))
        seg_sl  = slice_from(seg)
        ax = fig.add_subplot(gs[6, j])
        ax.imshow(base_sl, cmap='gray', origin='lower', interpolation='nearest', aspect='auto')
        ax.imshow(seg_sl, cmap=roi_cmap, norm=LABEL_NORM, origin='lower',
                  interpolation='nearest', aspect='auto')
        ax.set_axis_off()

    # Row 7: Method + Segmentation ERROR overlay (pred != GT)
    for j, m in enumerate(methods):
        base = pr01_by_method[m]["test"]
        base_sl = maybe_mask(slice_from(base))
        err_sl  = slice_from(seg_error_by_method[m]).astype(np.uint8)
        ax = fig.add_subplot(gs[7, j])
        ax.imshow(base_sl, cmap='gray', origin='lower', interpolation='nearest', aspect='auto')
        ax.imshow(err_sl, cmap=err_overlay_cmap, norm=ERR_NORM, origin='lower',
                  interpolation='nearest', aspect='auto')
        ax.set_axis_off()

    # Legend placed further down outside the image space
    handles = [mpatches.Patch(color=ROI_COLORS[name], label=name.replace('_',' ').title(), alpha=0.55)
               for name in ROI_NAMES]
    legend = fig.legend(
        handles=handles,
        loc='lower center',
        ncol=min(8, len(handles)),
        bbox_to_anchor=(0.5, -0.025),  # push legend slightly below the image grid
        fontsize=11
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), bbox_inches='tight', pad_inches=0.0)
    plt.close()
    logging.info("Saved grid: %s", out_path)


# --------------------- File & method helpers ----------------------
def first_nii_in(folder: Path) -> Optional[Path]:
    if not folder.exists(): return None
    for ext in ("*.nii.gz","*.nii","*.mnc","*.mha","*.nrrd"):
        matches = sorted(folder.glob(ext))
        if matches: return matches[0]
    return None

def scan_methods(processed_root: Path) -> Dict[str, Path]:
    out = {}
    if not processed_root.exists(): return out
    for d in sorted(p for p in processed_root.iterdir() if p.is_dir()):
        f = first_nii_in(d)
        if f is not None:
            out[d.name] = f
    return out

def method_cached(out_root: Path) -> bool:
    nii_dir = out_root / "nii_maps"
    return (nii_dir / "mae_map.nii.gz").exists() and (nii_dir / "psnr_map.nii.gz").exists() and (out_root / "metrics.json").exists()

def order_methods(method_to_file: Dict[str, Path]) -> List[str]:
    # desired display order mapped to folder names
    desired = ["raven_f2_nfg64", "arssr_rdn", "superformer", "arssr_rescnn", "nlmupsample", "trilinear"]
    present = [m for m in desired if m in method_to_file]
    # append any extra methods not in desired list, stable by name
    extras = sorted([m for m in method_to_file.keys() if m not in desired])
    return present + extras

def display_name_for_method(method: str) -> str:
    mapping = {
        "raven_f2_nfg64": "RAVEN",
        "arssr_rdn": "ArSSR-RDN",
        "superformer": "SuperFormer",
        "arssr_rescnn": "ArSSR-ResCNN",
        "nlmupsample": "NLMUPSAMPLE",
        "trilinear": "Trilinear",
    }
    return mapping.get(method, method)


# --------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser(
        description="Subject-level MAE/PSNR/SSIM/LPIPS + segmentation agreement + cross-method grids (centralized results)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--subject_dir", type=Path, required=True)
    ap.add_argument("--seg_method", type=str, default="synthseg", help="seg_<seg_method> folder to use for overlays & agreement")
    ap.add_argument("--degradation", type=str, default="kspacecrop_raw")
    ap.add_argument("--factor", type=str, default="x2")
    ap.add_argument("--with_lpips", action="store_true", default=True)
    ap.add_argument("--lpips_device", choices=["cpu","cuda"], default="cuda")
    ap.add_argument("--log_level", choices=["DEBUG","INFO","WARNING","ERROR"], default="INFO")
    ap.add_argument("--norm_mode", choices=["minmax","robust","robust_masked"], default="robust_masked")
    ap.add_argument("--norm_p_lo", type=float, default=0.5)
    ap.add_argument("--norm_p_hi", type=float, default=99.5)
    # Multi-plane, multi-position grids:
    ap.add_argument("--grid_planes", type=str, default="axial, coronal, sagittal",
                    help="Comma-separated planes for grids: axial,coronal,sagittal")
    ap.add_argument("--grid_positions", type=str, default="0.3, 0.4, 0.5, 0.6, 0.7",
                    help="Comma-separated positions (unit or percent), e.g. '0.5' or '30,60'")
    # Show masked background or not in grids:
    ap.add_argument("--grid_use_mask", dest="grid_use_mask", action="store_true", default=True,
                    help="Apply foreground mask to all grid rows (default).")
    ap.add_argument("--no-grid_use_mask", dest="grid_use_mask", action="store_false",
                    help="Do NOT apply the mask in grids (show full FOV).")
    # Bigger grids by default:
    ap.add_argument("--grid_panel_inches", type=float, default=5.0,
                    help="Width/height (in) per panel cell in the grid.")
    ap.add_argument("--grid_dpi", type=int, default=220, help="Figure DPI for saved grids.")
    # Centralized results subfolder name:
    ap.add_argument("--results_dirname", type=str, default="results",
                    help="Folder created under subject_dir to store ALL outputs (default: 'results').")
    ap.add_argument("--force", action="store_true", default=False,
                    help="Force recomputation of per-method metrics/maps & per-method 2D PNGs.")
    args = ap.parse_args()
    setup_logging(args.log_level)

    subj = args.subject_dir.resolve()
    results_root = subj / args.results_dirname
    results_root.mkdir(parents=True, exist_ok=True)

    # Locate core inputs
    original = first_nii_in(subj / "original")
    degraded = first_nii_in(subj / "degraded" / args.degradation / args.factor)
    processed_root = subj / "processed" / args.degradation / args.factor
    seg_root_base = subj / f"seg_{args.seg_method}" / args.degradation / args.factor
    mask_gt = first_nii_in(subj / "mask")
    if original is None or degraded is None or (not processed_root.exists()) or mask_gt is None:
        logging.error("Missing required files/folders.\n original=%s\n degraded=%s\n processed_root=%s\n mask_gt=%s",
                      original, degraded, processed_root, mask_gt)
        sys.exit(2)

    # Read GT HR + brain/labels mask
    gt_img  = robust_read_float(original);  assert gt_img is not None
    mask_img = robust_read_mask_img(mask_gt); assert mask_img is not None
    brain_mask_img = mask_img

    # Methods & their upsampled images (ordered per spec)
    method_to_file = scan_methods(processed_root)
    if not method_to_file:
        logging.error("No methods found under %s", processed_root)
        sys.exit(3)
    methods = order_methods(method_to_file)
    display_names = {m: display_name_for_method(m) for m in methods}

    # CSV collectors
    recon_rows: List[Dict[str, object]] = []
    seg_rows: List[Dict[str, object]] = []

    # Precompute normalized GT once per subject (for grids)
    gt_img_r = orient_to_RAS(gt_img)
    mk_img_r = orient_to_RAS(brain_mask_img)
    gt_np = to_np(gt_img_r, dtype=np.float64)
    mask_np = to_np(mk_img_r, dtype=np.int32)
    mask_bool = mask_np.astype(bool)
    gt_overlay_merged8 = merge_to_8rois(mask_np)

    # Native LR normalized [0,1] on GT grid
    lr_img = robust_read_float(degraded); assert lr_img is not None
    lr_img_res = resample_like(lr_img, gt_img, is_mask=False)
    lr_img_r = orient_to_RAS(lr_img_res)
    lr_np01 = _robust01(to_np(lr_img_r, dtype=np.float64), mask_bool, args.norm_p_lo, args.norm_p_hi)

    # Containers for grids (loaded either from cache or fresh run)
    lpips_maps_for_grid: Dict[str, Optional[np.ndarray]] = {}
    mae_maps_for_grid: Dict[str, np.ndarray] = {}
    pr01_for_grid: Dict[str, Dict[str, np.ndarray]] = {}
    seg_overlay_for_grid: Dict[str, np.ndarray] = {}
    seg_error_for_grid: Dict[str, np.ndarray] = {}

    # Main per-method loop
    for m in methods:
        test_path = method_to_file[m]
        out_root = results_root / "per_method" / args.degradation / args.factor / m
        cached = method_cached(out_root)

        # Read & resample test to GT geometry (needed for grids in any case)
        ts_img = robust_read_float(test_path); assert ts_img is not None
        ts_img = resample_like(ts_img, gt_img, is_mask=False)

        if not cached or args.force:
            logging.info("=== Method: %s -> computing metrics & maps (force=%s) ===", m, args.force)
            metrics = compute_and_save_for_method(
                method_name=m, gt_img=gt_img, test_img=ts_img, brain_mask_img=brain_mask_img,
                out_root=out_root, with_lpips=args.with_lpips, lpips_dev=args.lpips_device,
                norm_mode=args.norm_mode, p_lo=args.norm_p_lo, p_hi=args.norm_p_hi,
                err_cmap="magma"
            )
        else:
            logging.info("=== Method: %s -> cached; skipping recompute, will reload maps ===", m)
            metrics_path = out_root / "metrics.json"
            if metrics_path.exists():
                try:
                    metrics = json.loads(metrics_path.read_text())
                except Exception:
                    metrics = {"MAE": float("nan"), "PSNR": float("nan"), "SSIM": float("nan"), "LPIPS": float("nan")}
            else:
                metrics = {"MAE": float("nan"), "PSNR": float("nan"), "SSIM": float("nan"), "LPIPS": float("nan")}

        recon_rows.append({
            "subject_folder_name": subj.name,
            "degradation": args.degradation, "factor": args.factor, "method": display_names[m],
            "MAE": metrics.get("MAE"), "PSNR": metrics.get("PSNR"),
            "SSIM": metrics.get("SSIM"), "LPIPS": metrics.get("LPIPS"),
        })

        # Load per-method artifacts for the grids (cached or fresh)
        nii_dir = out_root / "nii_maps"
        mae_map = to_np(sitk.ReadImage(str(nii_dir / "mae_map.nii.gz"), sitk.sitkFloat32), dtype=np.float32)
        mae_maps_for_grid[m] = mae_map
        lp_map = None
        lp_path = nii_dir / "lpips_map.nii.gz"
        if lp_path.exists():
            lp_map = to_np(sitk.ReadImage(str(lp_path), sitk.sitkFloat32), dtype=np.float32)
        lpips_maps_for_grid[m] = lp_map

        # Normalize method image to [0,1] for overlays & raw row
        ts_img_r = orient_to_RAS(ts_img)
        pr_np = to_np(ts_img_r, dtype=np.float64)
        gt01, pr01 = normalize_pair(gt_np, pr_np, mask_np, args.norm_mode, args.norm_p_lo, args.norm_p_hi)
        pr01_for_grid[m] = {"test": pr01, "native": lr_np01}

        # Prepare merged-ROI seg overlay
        seg_path = first_nii_in(seg_root_base / m)
        if seg_path is None:
            logging.warning("No %s segmentation found for method %s at %s", args.seg_method, m, seg_root_base / m)
            seg_overlay_for_grid[m] = np.zeros_like(mask_np, dtype=np.uint8)
            pred_merged = seg_overlay_for_grid[m]
        else:
            seg_img = robust_read_mask_img(seg_path)
            if seg_img is None:
                seg_overlay_for_grid[m] = np.zeros_like(mask_np, dtype=np.uint8)
                pred_merged = seg_overlay_for_grid[m]
            else:
                seg_img = resample_like(seg_img, gt_img, is_mask=True)
                seg_img = orient_to_RAS(seg_img)
                seg_np = to_np(seg_img, dtype=np.int32)
                seg_overlay_for_grid[m] = merge_to_8rois(seg_np)
                pred_merged = seg_overlay_for_grid[m]

        gt_merged = gt_overlay_merged8
        # Segmentation error mask: union of ROI presence (pred or GT) where labels differ
        union_roi = (pred_merged > 0) | (gt_merged > 0)
        seg_error_for_grid[m] = (union_roi & (pred_merged != gt_merged)).astype(np.uint8)

        # Segmentation agreement per ROI vs GT labels
        for k, roi_name in enumerate(ROI_NAMES, start=1):
            dice, iou = dice_iou(pred_merged == k, gt_merged == k)
            seg_rows.append({
                "subject_folder_name": subj.name,
                "seg_method": args.seg_method,
                "degradation": args.degradation, "factor": args.factor, "method": display_names[m],
                "ROI": roi_name, "DICE": dice, "IOU": iou
            })

    # Write CSVs at results_root/csv
    csv_dir = results_root / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    if recon_rows:
        recon_csv = csv_dir / "subject_reconstruction_metrics.csv"
        with recon_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(recon_rows[0].keys()))
            w.writeheader(); w.writerows(recon_rows)
        logging.info("Saved: %s", recon_csv)

    if seg_rows:
        seg_csv = csv_dir / "subject_segmentation_agreement.csv"
        with seg_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(seg_rows[0].keys()))
            w.writeheader(); w.writerows(seg_rows)
        logging.info("Saved: %s", seg_csv)
    else:
        logging.warning("No segmentation agreement rows (missing seg files?)")

    # Cross-method grids for requested planes & positions
    grids_dir = results_root / "grids" / args.degradation / args.factor
    planes = _parse_planes(args.grid_planes)
    positions = _parse_positions_percent_or_unit(args.grid_positions)

    gt01_for_grid = _robust01(to_np(orient_to_RAS(gt_img), dtype=np.float64), mask_bool, args.norm_p_lo, args.norm_p_hi)

    for plane in planes:
        for pos in positions:
            pct = int(round(pos * 100))
            out_grid  = grids_dir / f"grid_{plane}_{pct}.png"
            draw_cross_method_grid(
                methods=methods,
                display_names=display_names,
                gt01=gt01_for_grid,
                pr01_by_method=pr01_for_grid,
                mask_bool=mask_bool,
                mae_map_by_method=mae_maps_for_grid,
                lpips_map_by_method=lpips_maps_for_grid,
                seg_overlay_by_method=seg_overlay_for_grid,
                seg_error_by_method=seg_error_for_grid,
                gt_overlay_merged8=gt_overlay_merged8,
                plane=plane, pos=pos,
                out_path=out_grid,
                err_cmap="magma",
                panel_inches=args.grid_panel_inches,
                dpi=args.grid_dpi,
                apply_mask=args.grid_use_mask
            )

if __name__ == "__main__":
    main()
