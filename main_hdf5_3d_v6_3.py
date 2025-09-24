import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor, RichProgressBar

from pytorch_lightning.utilities import rank_zero_only
from taming.data.utils import AsegDatasetWithAugmentation3D
import math
import signal
# from rich.console import Console
from pytorch_lightning.callbacks import TQDMProgressBar
from tqdm import tqdm
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
print(f'Changing working directory to the current script’s absolute path: {dname}')
os.chdir(dname)

# ------------------------------------------------------------------------------
# 1) CUSTOM TQDM PROGRESS BAR
# ------------------------------------------------------------------------------
import sys
import shutil
import torch
from pytorch_lightning.callbacks import TQDMProgressBar
from tqdm import tqdm


import collections # Import collections for defaultdict

class CustomTQDMProgressBar(TQDMProgressBar):
    """
    Custom TQDM Progress Bar that:
    - Prints selected step metrics (s_*) and Cumulative Moving Averages (a_*)
      below the progress bar on separate lines.
    - Formats metrics for readability.
    - Aims for compatibility with non-interactive terminals (like SLURM logs).
    - Resets CMA at the start of each training epoch.
    """
    def __init__(
        self,
        refresh_rate=1,
        process_position=0,
        metrics_per_line=4,
        print_train_freq=None,
        print_val_freq=None
    ):
        super().__init__(refresh_rate, process_position)
        self.metrics_per_line = metrics_per_line
        self.print_train_freq = print_train_freq if print_train_freq is not None else max(1, refresh_rate)
        self.print_val_freq = print_val_freq if print_val_freq is not None else max(1, refresh_rate * 5)

        # --- CMA Tracking ---
        self.ema_metrics = {}
        # Add '_step' suffix to ALL keys to match trainer.logged_metrics
        self.ema_tracking_keys = [
            "train/aeloss_step",
            "train/rec_loss_step",
            "train/gan_g_loss_step",
            "train/d_weight_step",
            "train/gan_d_loss_step",
            "train/logits_real_step",
            "train/logits_fake_step",
            "train/kl_loss_step",
            "train/total_loss_step",
            "train/nll_loss_step",
            "train/g_loss_step",
            "train/disc_factor_step",
            "train/logvar_step",
            # "train/rampup_factor_step", # Uncomment if logged and needed
        ]
        self._reset_ema_metrics()
        self.last_logged_train_batch = -1 # Use global step for tracking

    def _reset_ema_metrics(self):
        """Resets the CMA storage. Called at the start of each train epoch."""
        self.ema_metrics = {
            key: {"sum": 0.0, "count": 0} for key in self.ema_tracking_keys
        }
        # print("--- [DEBUG] CMA Metrics Reset ---") # Optional: confirm reset

    def on_train_epoch_start(self, trainer, pl_module):
        """Reset CMA metrics at the start of each training epoch."""
        super().on_train_epoch_start(trainer, pl_module)
        self._reset_ema_metrics() # <<< CMA RESET LOGIC >>>
        self.last_logged_train_batch = -1 # Reset step tracker
        if trainer.is_global_zero:
             print(f"\n--- Starting Training Epoch {trainer.current_epoch} ---")

    def on_validation_start(self, trainer, pl_module):
        """Called before validation loop. Does NOT reset training CMA."""
        super().on_validation_start(trainer, pl_module)
        if trainer.is_global_zero and not trainer.sanity_checking:
             # No reset here - validation is separate
             print(f"\n--- Starting Validation Epoch {trainer.current_epoch} ---")

    def _get_gpu_stats(self, trainer):
        # (Keep implementation as before)
        if torch.cuda.is_available() and hasattr(torch.cuda, 'memory_allocated'):
            try:
                if hasattr(trainer, 'strategy') and hasattr(trainer.strategy, 'local_rank'):
                   device_idx = trainer.strategy.local_rank
                elif hasattr(trainer, 'local_rank'):
                   device_idx = trainer.local_rank
                else:
                   device_idx = 0
                if device_idx >= torch.cuda.device_count(): device_idx = 0
                device = torch.device(f"cuda:{device_idx}")
                mem_allocated = torch.cuda.memory_allocated(device)
                mem_reserved = torch.cuda.memory_reserved(device)
                total_mem = torch.cuda.get_device_properties(device).total_memory
                if total_mem == 0: return {}
                alloc_GB = mem_allocated / (1024 ** 3)
                resv_GB = mem_reserved / (1024 ** 3)
                alloc_percent = mem_allocated / total_mem * 100
                resv_percent = mem_reserved / total_mem * 100
                return {
                    "GPU-A": f"{alloc_GB:.1f}G({alloc_percent:.0f}%)",
                    "GPU-R": f"{resv_GB:.1f}G({resv_percent:.0f}%)"
                }
            except Exception: return {}
        return {}

    def _format_value(self, value):
        # (Keep implementation as before)
        if isinstance(value, torch.Tensor): value = value.item()
        if isinstance(value, float):
            if abs(value) == 0.0: return "0.00e+0"
            if abs(value) < 1e-5 or abs(value) >= 1e5: return f"{value:.2e}"
            elif abs(value) < 0.01: return f"{value:.5f}"
            elif abs(value) < 1: return f"{value:.4f}"
            elif abs(value) < 100: return f"{value:.3f}"
            else: return f"{value:.2f}"
        return str(value)

    def _shorten_key(self, key):
        # (Keep implementation as before - handles _step suffix)
        key = key.split('/')[-1]
        suffixes_to_remove = ['_step', '_epoch']
        for suffix in suffixes_to_remove:
            if key.endswith(suffix): key = key[:-len(suffix)]
        key = key.replace("total_loss", "total")
        key = key.replace("aeloss", "G_loss")
        key = key.replace("discloss", "D_loss")
        key = key.replace("rec_loss", "rec")
        key = key.replace("nll_loss", "nll")
        key = key.replace("gan_g_loss", "ganG")
        key = key.replace("gan_d_loss", "ganD")
        key = key.replace("logits_real", "Lreal")
        key = key.replace("logits_fake", "Lfake")
        key = key.replace("d_weight", "dW")
        key = key.replace("disc_factor", "dFac")
        key = key.replace("rampup_factor", "rampF")
        key = key.replace("logvar", "lvar")
        key = key.replace("kl_loss", "kl")
        return key

    def _print_formatted_metrics(self, metrics_dict):
        # (Keep implementation as before)
        if not metrics_dict: return
        items = list(metrics_dict.items())
        for i in range(0, len(items), self.metrics_per_line):
            group = items[i:i + self.metrics_per_line]
            line_parts = [f"{k}={v}" for k, v in group]
            print(f"  {', '.join(line_parts)}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update CMA and print metrics (including CMA) periodically."""
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

        if trainer.sanity_checking: return

        metrics = trainer.logged_metrics if trainer.logged_metrics else {}

        # --- Update CMA ---
        current_global_step = trainer.global_step # Use global step for tracking

        # Check if this global step has already been processed for CMA update
        if current_global_step != self.last_logged_train_batch:
            self.last_logged_train_batch = current_global_step

            # <<< --- START DEBUG PRINT (Optional)--- >>>
            if trainer.is_global_zero and (batch_idx + 1) % (self.print_train_freq * 20) == 1: # Even less frequent
                print("\n--- [DEBUG] Available logged_metrics keys: ---")
                print(sorted(list(metrics.keys())))
                print(f"--- [DEBUG] Keys tracked for CMA: {self.ema_tracking_keys} ---")
                missing_keys = [k for k in self.ema_tracking_keys if k not in metrics]
                if missing_keys:
                     # This is expected due to alternating optimizers, so maybe downgrade from WARNING
                     print(f"--- [DEBUG] Info: CMA keys not found in *this* step's logged_metrics: {missing_keys}")
                types_found = {k: type(metrics[k]).__name__ for k in self.ema_tracking_keys if k in metrics}
                if types_found: print(f"--- [DEBUG] Types found for tracked CMA keys in this step: {types_found}")
            # <<< --- END DEBUG PRINT --- >>>

            for key in self.ema_tracking_keys:
                if key in metrics:
                    value = metrics[key]
                    float_val = None
                    processed = False
                    try:
                        if isinstance(value, torch.Tensor):
                            if value.numel() == 1:
                                float_val = value.cpu().item()
                                processed = True
                        elif isinstance(value, (float, int)):
                            float_val = float(value)
                            processed = True
                        # Add elif for numpy types if necessary

                        if processed and float_val is not None:
                           if not math.isnan(float_val) and not math.isinf(float_val):
                               self.ema_metrics[key]["sum"] += float_val
                               self.ema_metrics[key]["count"] += 1
                           # Optional: Log if NaN/Inf encountered
                        # else: # Optional: Log if not processed or float_val is None
                           # if trainer.is_global_zero and (batch_idx + 1) % (self.print_train_freq * 20) == 1:
                           #    print(f"--- [DEBUG] Did not process CMA for {key} (Type: {type(value)}, Processed: {processed}, FloatVal: {float_val})")

                    except Exception as e:
                        if trainer.is_global_zero:
                            print(f"Warning: Could not process metric {key} (value: {value}, type: {type(value)}) for CMA: {e}")


        # --- Periodic Printing ---
        if trainer.is_global_zero and (batch_idx + 1) % self.print_train_freq == 0:

            metrics_to_print = collections.OrderedDict()
            current_metrics_for_print = metrics # Use metrics potentially updated above

            # 1. GPU Stats
            metrics_to_print.update(self._get_gpu_stats(trainer))

            # 2. Current Step Metrics (s_*)
            step_metrics_found = False
            for key, value in current_metrics_for_print.items():
                is_scalar_tensor = isinstance(value, torch.Tensor) and value.numel() == 1
                is_numeric = isinstance(value, (float, int))
                if key.startswith("train/") and (is_numeric or is_scalar_tensor):
                    # Check if the base key (without _step) should be included in step printout
                    # This avoids printing things like train/disc_factor_step twice if logged in both G/D steps
                    # However, let's keep it simple and print whatever train/ key is available in this step's log
                    s_key = "s_" + self._shorten_key(key) # _shorten_key removes _step suffix
                    metrics_to_print[s_key] = self._format_value(value)
                    step_metrics_found = True

            # 3. CMA Metrics (a_*) --- NO COUNTS ---
            cma_metrics_found = False
            for key in self.ema_tracking_keys:
                # Check internal CMA state if count > 0
                if key in self.ema_metrics and self.ema_metrics[key]["count"] > 0:
                    count_val = self.ema_metrics[key]["count"]
                    avg_val = self.ema_metrics[key]["sum"] / count_val
                    a_key = "a_" + self._shorten_key(key) # _shorten_key removes _step suffix
                    metrics_to_print[a_key] = self._format_value(avg_val) # Add average value
                    # The line adding the count key (c_key) is now removed/commented out.
                    cma_metrics_found = True

            # 4. Learning Rate(s)
            if hasattr(trainer, 'optimizers') and trainer.optimizers:
                 try:
                    for i, opt in enumerate(trainer.optimizers):
                        if opt and hasattr(opt, 'param_groups') and opt.param_groups:
                                current_lr = opt.param_groups[0]['lr']
                                metrics_to_print[f'lr{i}'] = self._format_value(current_lr)
                 except (IndexError, KeyError, TypeError): pass

            # Perform Printing only if there's something new to print
            if step_metrics_found or cma_metrics_found:
                if self.train_progress_bar is not None:
                    try: self.train_progress_bar.refresh()
                    except Exception: pass

                print("") # Newline
                print(f"Epoch {trainer.current_epoch} Batch {batch_idx+1} (Step {trainer.global_step}):")
                self._print_formatted_metrics(metrics_to_print) # Will now only contain s_* and a_* (and GPU/LR)
                sys.stdout.flush()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # (Keep implementation as before - no changes needed here regarding training CMA)
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if trainer.sanity_checking: return
        if trainer.is_global_zero and (batch_idx + 1) % self.print_val_freq == 0:
            metrics_to_print = collections.OrderedDict()
            metrics = trainer.logged_metrics if trainer.logged_metrics else {}
            metrics_to_print.update(self._get_gpu_stats(trainer))
            val_metrics_found = False
            for key, value in metrics.items():
                 is_scalar_tensor = isinstance(value, torch.Tensor) and value.numel() == 1
                 is_numeric = isinstance(value, (float, int))
                 if key.startswith("val/") and (is_numeric or is_scalar_tensor):
                     # Shorten key - _shorten_key removes _step/_epoch if present
                     short_key = self._shorten_key(key)
                     metrics_to_print[short_key] = self._format_value(metrics[key])
                     val_metrics_found = True
            if val_metrics_found:
                 if self.val_progress_bar is not None:
                     try: self.val_progress_bar.refresh()
                     except Exception: pass
                 print("")
                 dl_id_str = f" DL {dataloader_idx}" if hasattr(trainer, 'num_val_dataloaders') and trainer.num_val_dataloaders > 1 else ""
                 print(f"Validation Epoch {trainer.current_epoch} Batch {batch_idx+1}{dl_id_str}:")
                 self._print_formatted_metrics(metrics_to_print)
                 sys.stdout.flush()

    # Add other TQDM methods if needed (on_test_*, on_predict_*, etc.)
    # Ensure cleanup methods like on_train_end are handled if necessary
# ------------------------------------------------------------------------------
# 2) HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

# ------------------------------------------------------------------------------
# 3) ARGPARSE UTILITIES
# ------------------------------------------------------------------------------
def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-n", "--name", type=str, default="", help="postfix for logdir")
    parser.add_argument("-r", "--resume", type=str, default="", help="resume from logdir or checkpoint")
    parser.add_argument("-b", "--base", nargs="*", metavar="base_config.yaml",
                        default=list(), help="paths to base configs. Loaded left-to-right.")
    parser.add_argument("-t", "--train", type=str2bool, default=False, help="train (True/False)")
    parser.add_argument("--no-test", type=str2bool, default=True, help="disable test")
    parser.add_argument("-p", "--project", type=str, help="project name or path to existing project")
    parser.add_argument("-d", "--debug", type=str2bool, default=False, help="enable post-mortem debugging")
    parser.add_argument("-s", "--seed", type=int, default=23, help="seed for seed_everything")
    parser.add_argument("-f", "--postfix", type=str, default="", help="post-postfix for default name")
    # (Optional: support old --gpus argument)
    parser.add_argument("--gpus", type=str, default="", help="(deprecated) list of GPU ids, e.g. '0,1'")
    return parser

# ------------------------------------------------------------------------------
# 4) DATA MODULES & WRAPPED DATASET
# ------------------------------------------------------------------------------
class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class DataModuleFromConfig3D(pl.LightningDataModule):
    """
    DataModule for managing training, validation, and test datasets.
    Handles lazy loading of HDF5 files.
    """
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None, transform_train=None,
                 lazy_loading=False,
                 seed=23):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = {}
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.transform_train = transform_train
        self.seed = seed
        self.lazy_loading = lazy_loading

        if train:
            self.dataset_configs["train"] = train
        if validation:
            self.dataset_configs["validation"] = validation
        if test:
            self.dataset_configs["test"] = test

        self.full_train_filelist = []
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _load_paths_from_txt_files(self, txt_file_list):
        combined_paths = []
        for txt_file in txt_file_list:
            with open(txt_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                combined_paths.extend(lines)
        return combined_paths

    def setup(self, stage=None):
        if "validation" in self.dataset_configs:
            config = self.dataset_configs["validation"]
            if "params" in config and "dataset_paths_files" in config["params"]:
                val_txt_files = config["params"]["dataset_paths_files"]
                val_paths = self._load_paths_from_txt_files(val_txt_files)
                print("For validation, the following HDF5 files were found:")
                for p in val_paths:
                    print(f"  {p}")
                self.val_dataset = AsegDatasetWithAugmentation3D(val_paths, transforms=None, lazy_loading=self.lazy_loading)
            else:
                self.val_dataset = instantiate_from_config(config)
        if "test" in self.dataset_configs:
            config = self.dataset_configs["test"]
            if "params" in config and "dataset_paths_files" in config["params"]:
                test_txt_files = config["params"]["dataset_paths_files"]
                test_paths = self._load_paths_from_txt_files(test_txt_files)
                print("For testing, the following HDF5 files were found:")
                for p in test_paths:
                    print(f"  {p}")
                self.test_dataset = AsegDatasetWithAugmentation3D(test_paths, transforms=None, lazy_loading=self.lazy_loading)
            else:
                self.test_dataset = instantiate_from_config(config)
        if "train" in self.dataset_configs:
            config = self.dataset_configs["train"]
            if "params" in config and "dataset_paths_files" in config["params"]:
                train_txt_files = config["params"]["dataset_paths_files"]
                self.full_train_filelist = self._load_paths_from_txt_files(train_txt_files)
                print("For training, the following HDF5 files were found:")
                for p in self.full_train_filelist:
                    print(f"  {p}")
                print(f"Total training HDF5 files found: {len(self.full_train_filelist)}")
                self.train_dataset = AsegDatasetWithAugmentation3D(
                    self.full_train_filelist,
                    transforms=self.transform_train, lazy_loading=self.lazy_loading
                )
            else:
                raise ValueError("Train dataset configuration must include a 'dataset_paths_files' list.")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=self.worker_init_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )

    def worker_init_fn(self, worker_id):
        worker_seed = self.seed + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)


# ------------------------------------------------------------------------------
# 5) CALLBACKS: SetupCallback, ImageLogger3D, etc.
# ------------------------------------------------------------------------------
class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)
            print("Project config:")
            # Print config as YAML
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config, os.path.join(self.cfgdir, f"{self.now}-project.yaml"))
            print("Lightning config:")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, f"{self.now}-lightning.yaml"))
        else:
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass

def make_3d_grid(vol, clamp=True):
    if clamp:
        vol = torch.clamp(vol, -1., 1.)

    if vol.ndim == 5:
        # Assume vol shape is [N, C, H, W, D] for a 3D volume
        # grid1 = torchvision.utils.make_grid(vol[:, :, vol.shape[2] // 2, :, :])
        # grid2 = torchvision.utils.make_grid(vol[:, :, :, vol.shape[3] // 2, :])
        grid3 = torchvision.utils.make_grid(vol[:, :, :, :, vol.shape[4] // 2])
        return grid3
    elif vol.ndim == 4:
        # Assume vol shape is [N, C, H, W] (a 2D image)
        return torchvision.utils.make_grid(vol)
    else:
        raise ValueError("Unsupported tensor dimensions: expected 4 or 5 dimensions, got {}".format(vol.ndim))

class ImageLogger3D(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images

        # Replace TestTubeLogger with TensorBoardLogger:
        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb,
            pl.loggers.TensorBoardLogger: self._tensorboard,
        }

        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.increase_log_steps = increase_log_steps
        self.logger_log_images = {
            # pl.loggers.WandbLogger: self._wandb, # Keep commented if not used
            pl.loggers.TensorBoardLogger: self._tensorboard,
        }

        # Calculate log_steps based on the stored attribute
        if self.increase_log_steps and self.batch_freq > 0: # Added check for batch_freq > 0
             # Calculate max power of 2 less than batch_freq safely
             max_power = math.floor(math.log2(self.batch_freq)) if self.batch_freq > 0 else 0
             self.log_steps = [2**n for n in range(max_power + 1)]
             # Ensure batch_freq itself is included if not a power of 2?
             # Often it's meant to just log at 1, 2, 4, 8... up to freq
        else:
             self.log_steps = [self.batch_freq] # Log only at the main frequency

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        """
        Adapted to handle 5D volumes [N, C, H, W, Z]. For each key in `images`,
        we extract a mid-slice grid along each axis, combine them, and log to WandB.
        """
        # If using wandb, uncomment the import
        # raise ValueError("No way wandb")  # Remove or handle as needed

        grids = {}
        for k in images:
            vol = images[k]
            if vol.ndim == 5:
                # Create a single grid representing this 5D volume
                grid_3d = make_3d_grid(vol, clamp=self.clamp)
                grids[f"{split}/{k}"] = grid_3d
            elif vol.ndim == 4:
                # Original 2D approach [N, C, H, W]
                grid = torchvision.utils.make_grid(vol)
                grid = (grid + 1.0) / 2.0 if self.clamp else grid
                grids[f"{split}/{k}"] = grid
            else:
                continue

    @rank_zero_only
    def _tensorboard(self, pl_module, images, batch_idx, split):
        """
        Adapted to handle 5D volumes. Slices and logs to the TensorBoard via add_image.
        """
        for k in images:
            vol = images[k]
            if vol.ndim == 5:
                # Convert the volume to a 2D grid
                grid_3d = make_3d_grid(vol, clamp=self.clamp)
                tag = f"{split}/{k}"
                pl_module.logger.experiment.add_image(
                    tag, grid_3d, global_step=pl_module.global_step
                )
            elif vol.ndim == 4:
                # Original 2D approach
                grid = torchvision.utils.make_grid(vol)
                grid = (grid + 1.0) / 2.0 if self.clamp else grid
                tag = f"{split}/{k}"
                pl_module.logger.experiment.add_image(
                    tag, grid, global_step=pl_module.global_step
                )
            else:
                continue

    @rank_zero_only
    def log_local3D(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k, vol in images.items():
            # Expecting shape [N, C, H, W, Z]
            if vol.ndim != 5:
                continue
            vol = (vol + 1.0) / 2.0 #[Range back to 0-1]
            combined_grid = make_3d_grid(vol, clamp=self.clamp)  # [C, H_total, W_total]
            combined_grid = combined_grid.cpu().numpy()
            combined_grid = (combined_grid * 255).astype(np.uint8)

            filename = f"{k}_gs-{global_step:06}_e-{current_epoch:06}_b-{batch_idx:06}.png"
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)

            # If grid has 1 or 3 channels, transpose to HWC
            if combined_grid.shape[0] in [1, 3]:
                combined_grid = np.transpose(combined_grid, (1, 2, 0))

            Image.fromarray(combined_grid).save(path)

    def log_img3D(self, pl_module, batch, batch_idx, split="train"):
        if not self.check_frequency(batch_idx, split):
             return # Only log based on frequency or epoch end checks

        if (hasattr(pl_module, "log_images3D") and
                callable(pl_module.log_images3D) and self.max_images > 0):

            logger = pl_module.logger
            is_train = pl_module.training
            if is_train: pl_module.eval()

            with torch.no_grad():
                # Get the full log dictionary from the model
                log_dict_from_model = pl_module.log_images3D(batch, split=split)

            # --- Filter log_dict_from_model to only include tensors ---
            images_to_log = {}
            metadata_to_log = {} # Optional: store other info separately if needed
            for k, v in log_dict_from_model.items():
                if isinstance(v, torch.Tensor): # Check if it's a tensor
                    images_to_log[k] = v
                else:
                    metadata_to_log[k] = v # Store non-tensors elsewhere
            # --- End filtering ---

            # Ensure images dict is not empty
            if not images_to_log:
                 if is_train: pl_module.train()
                 return

            # Process ONLY the image tensors
            processed_images = {} # Create a new dict for processed images
            for k in images_to_log:
                # Access shape and slice - Now safe because we know it's a tensor
                N = min(images_to_log[k].shape[0], self.max_images)
                img_tensor = images_to_log[k][:N].detach().cpu() # Process the tensor
                if self.clamp:
                    img_tensor = torch.clamp(img_tensor, -1., 1.)
                processed_images[k] = img_tensor # Store processed tensor

            # Log processed images locally and to logger
            self.log_local3D(logger.save_dir, split, processed_images,
                             pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_func = self.logger_log_images.get(type(logger))
            if logger_log_func:
                # Pass the dict containing only processed image tensors
                logger_log_func(pl_module, processed_images, pl_module.global_step, split)

            # Optional: Log metadata separately if desired (e.g., as text)
            # if metadata_to_log and hasattr(logger.experiment, 'add_text'):
            #     metadata_str = "\n".join([f"{key}: {value}" for key, value in metadata_to_log.items()])
            #     logger.experiment.add_text(f"{split}/ImageMetadata_e{pl_module.current_epoch}_b{batch_idx}",
            #                                metadata_str, global_step=pl_module.global_step)


            if is_train: pl_module.train()

    def check_frequency(self, batch_idx, split="train"): # <<< ADD split="train" argument
        # You can optionally use the 'split' argument here later if needed
        # for different logging frequencies for train vs val.
        # For now, the logic only depends on batch_idx.

        # Batch frequency check
        # Use (batch_idx + 1) if you want to log on the Nth batch (e.g., 1000th, 2000th)
        # Use batch_idx if you want to log on batch 0, N, 2N...
        # Let's use batch_idx based on the original logic:
        if (batch_idx % self.batch_freq == 0):
             return True

        # Log steps check (used for initial frequent logging)
        if self.increase_log_steps and batch_idx in self.log_steps:
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True

        return False # Default to not logging

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.log_img3D(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.log_img3D(pl_module, batch, batch_idx, split="val")


def setup_logging_and_config(opt):
    """
    Handles 8 scenarios based on:
      - opt.resume (bool)
      - opt.name (bool)
      - len(opt.base) > 0 (bool)
    and enforces your requested logic for each.
    """

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    resume = bool(opt.resume)
    have_name = bool(opt.name)
    have_config = len(opt.base) > 0  # True if user passed at least one config file

    # Helper to parse the checkpoint path (file or folder)
    def parse_checkpoint_path(ckpt_path):
        """Returns (logdir, ckpt, base_configs) from a resume path."""
        if not os.path.exists(ckpt_path):
            raise ValueError(f"Cannot find {ckpt_path}")

        if os.path.isfile(ckpt_path):
            # E.g. ".../logs/.../checkpoints/something.ckpt"
            paths = ckpt_path.split("/")
            try:
                idx = len(paths) - paths[::-1].index("logs") + 1
            except ValueError:
                raise ValueError("Cannot find 'logs' in checkpoint path.")
            old_logdir = "/".join(paths[:idx])  # old log folder
            ckpt_file = ckpt_path
        else:
            # ckpt_path is a directory
            assert os.path.isdir(ckpt_path), ckpt_path
            old_logdir = ckpt_path.rstrip("/")
            ckpt_file = os.path.join(old_logdir, "checkpoints", "last.ckpt")

        # Collect base configs from old logdir
        base_configs = sorted(glob.glob(os.path.join(old_logdir, "configs/*.yaml")))

        return old_logdir, ckpt_file, base_configs

    # Final outputs we need:
    logdir = None
    nowname = None
    ckptdir = None
    cfgdir = None

    # -------------------------
    # A) NOT RESUMING
    # -------------------------
    if not resume:
        # A block: No resume
        if (not have_name) and (not have_config):
            # A1: error
            raise ValueError(
                "A1: No resume, no name, no config -> Error. At least one configuration file is required."
            )

        elif (not have_name) and have_config:
            # A2: the log directory should be now + the config filename
            cfg_fname = os.path.basename(opt.base[0])  # e.g. "something.yaml"
            cfg_base  = os.path.splitext(cfg_fname)[0]   # e.g. "something"
            nowname = f"{now}_{cfg_base}{opt.postfix}"
            logdir = os.path.join("logs", nowname)

        elif have_name and (not have_config):
            # A3: error
            raise ValueError(
                "A3: No resume, name provided, but no config -> Error. A configuration file is required if not resuming."
            )

        else:  # have_name and have_config
            # A4: use the config, name of log dir is (now + name)
            nowname = f"{now}_{opt.name}{opt.postfix}"
            logdir = os.path.join("logs", nowname)

        # No resume => no existing checkpoint
        opt.resume = None  # or leave it as None if it wasn't set

        # We do not alter opt.base beyond what's provided by user for A2 or A4

    # -------------------------
    # B) RESUMING
    # -------------------------
    else:
        # parse the checkpoint
        old_logdir, ckpt, base_configs = parse_checkpoint_path(opt.resume)
        opt.resume = ckpt
        if (not have_name) and (not have_config):
            opt.base = base_configs + opt.base
            tmp = old_logdir.split("/")
            nowname = tmp[tmp.index("logs") + 1]
            logdir = old_logdir

        elif (not have_name) and have_config:
            cfg_fname = os.path.basename(opt.base[0])
            cfg_base  = os.path.splitext(cfg_fname)[0]
            nowname = f"{now}_{cfg_base}{opt.postfix}"
            logdir = os.path.join("logs", nowname)
            opt.base = base_configs + opt.base

        elif have_name and (not have_config):
            nowname = f"{now}_{opt.name}{opt.postfix}"
            logdir = os.path.join("logs", nowname)

            # Use only the checkpoint’s config, ignore user config
            opt.base = base_configs

        else:
            nowname = f"{now}_{opt.name}{opt.postfix}"
            logdir = os.path.join("logs", nowname)

    # Final path setups
    print(f"===== logdir is {logdir}")
    print(f"===== nowname is {nowname}")

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir  = os.path.join(logdir, "configs")
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)

    return logdir, nowname, ckptdir, cfgdir

# ------------------------------------------------------------------------------
# 6) MAIN TRAINING / SCRIPT ENTRYPOINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    logdir, nowname, ckptdir, cfgdir = setup_logging_and_config(opt)
    seed_everything(opt.seed)

    try:
        # Load and merge configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli_cfg = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli_cfg)
        #Uncomment if you want to bypass lightning default mode. You should also uncomment the respective lines from __init__ in architecture module (autoencoderkl3dv3)
        if opt.resume is not None:
            config.model.params.ckpt_path = opt.resume
        lightning_config = config.pop("lightning", OmegaConf.create())
        trainer_config = lightning_config.get("trainer", OmegaConf.create())

        # Convert trainer_config to dictionary
        trainer_kwargs = dict(**OmegaConf.to_container(trainer_config))
        trainer_kwargs.pop("resume_from_checkpoint", None)

        # Support old --gpus argument (if provided via opt.gpus)
        if hasattr(opt, "gpus") and opt.gpus:
            gpus_val = opt.gpus.strip()
            if gpus_val == "":
                trainer_kwargs["accelerator"] = "cpu"
                trainer_kwargs["devices"] = 1
            elif gpus_val == "-1":
                # Use all GPUs
                trainer_kwargs["accelerator"] = "gpu"
                trainer_kwargs["devices"] = -1
            else:
                # Parse a comma-separated string of GPU indices, e.g. "0,1"
                devs = [x.strip() for x in gpus_val.split(",") if x.strip().isdigit()]
                trainer_kwargs["accelerator"] = "gpu"
                if not devs:
                    trainer_kwargs["devices"] = 1
                else:
                    trainer_kwargs["devices"] = list(map(int, devs))
        trainer_kwargs.pop("gpus", None)  # remove old key

        if "accelerator" not in trainer_kwargs:
            # If not specified in config, default to CPU
            trainer_kwargs["accelerator"] = "cpu"
            trainer_kwargs["devices"] = 1

        # Report GPU info
        if trainer_kwargs.get("accelerator", "cpu") == "gpu":
            print(f"Running on GPUs: {trainer_kwargs.get('devices')}")
            cpu = False
        else:
            cpu = True

        # 1) Instantiate model from config
        model = instantiate_from_config(config.model)

        # 2) Build callbacks + logger
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "tensorboard": {
                "target": "pytorch_lightning.loggers.TensorBoardLogger",
                "params": {
                    "name": "tensorboard",
                    "save_dir": logdir
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["tensorboard"]
        logger_cfg = lightning_config.get("logger", OmegaConf.create())
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        logger = instantiate_from_config(logger_cfg)
        # Convert internal hparams to OmegaConf if necessary:
        if hasattr(logger, '_hparams') and isinstance(logger._hparams, dict):
            logger._hparams = OmegaConf.create(logger._hparams)

        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06d}-{step:06d}-{val_rec_loss:.4f}",
                "verbose": True,
                "save_last": True
            }
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = -1
            default_modelckpt_cfg["params"]["save_last"] = True

        modelckpt_cfg = lightning_config.get("modelcheckpoint", OmegaConf.create())
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        checkpoint_callback = instantiate_from_config(modelckpt_cfg)

        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main_hdf5_3d_v6_3.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "main_hdf5_3d_v6_3.ImageLogger3D",
                "params": {
                    "batch_frequency": 1000,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "custom_tqdm_bar": {
                "target": "main_hdf5_3d_v6_3.CustomTQDMProgressBar",  # adjust this path accordingly
                "params": {
                    "refresh_rate": 1
                }
            },
            "learning_rate_logger": {
                "target": "pytorch_lightning.callbacks.LearningRateMonitor",
                "params": {
                    "logging_interval": "step"
                }
            },
        }
        callbacks_cfg = lightning_config.get("callbacks", OmegaConf.create())
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        callbacks_list = [checkpoint_callback]
        for cb_key in callbacks_cfg:
            cb = instantiate_from_config(callbacks_cfg[cb_key])
            callbacks_list.append(cb)

        trainer = Trainer(
            logger=logger,
            callbacks=callbacks_list,
            **trainer_kwargs
        )

        data = instantiate_from_config(config.data)
        data.setup()

        # ----------------------------------------------------------------------
        # Handle the case where --devices could be -1, a list, or an int
        # to compute "ngpu" for LR scaling:
        # ----------------------------------------------------------------------
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if trainer_kwargs.get("accelerator", "cpu") == "gpu":
            devices_val = trainer_kwargs.get("devices", 1)
            if isinstance(devices_val, int):
                # If user passes --devices -1 => use all GPUs
                if devices_val == -1:
                    ngpu = torch.cuda.device_count()
                else:
                    ngpu = devices_val
            elif isinstance(devices_val, str):
                device_ids = [d for d in devices_val.split(",") if d.strip() != ""]
                ngpu = len(device_ids)
            elif isinstance(devices_val, list):
                ngpu = len(devices_val)
            else:
                ngpu = 1
        else:
            ngpu = 1

        accumulate_grad_batches = config.model.params.trainconfig.accumulate_grad_batches_g
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)"
            .format(
                model.learning_rate,
                accumulate_grad_batches,
                ngpu,
                bs,
                base_lr
            )
        )

        # For debugging signals
        def melk(*args, **kwargs):
            if trainer.global_rank == 0:
                print("Summoning checkpoint via USR1 signal.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb
                pudb.set_trace()

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # ----------------------------------------------------------------------
        # Run training (optionally) and then testing
        # ----------------------------------------------------------------------
        if opt.train:
            try:
                trainer.fit(model, data, ckpt_path=getattr(opt, "resume_from_checkpoint", None))
                # trainer.fit(model, data, ckpt_path=opt.resume)
            except Exception:
                melk()
                raise

        if not opt.no_test and not trainer.interrupted:
            # trainer.test(model, data)
            trainer.test(...)

    except Exception as e:
        # If debugging is on, let us drop into post-mortem
        if opt.debug and "trainer" in locals() and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise e

    finally:
        # If debug mode was used and not resuming, move logs to debug_runs
        if opt.debug and not opt.resume and "trainer" in locals() and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)