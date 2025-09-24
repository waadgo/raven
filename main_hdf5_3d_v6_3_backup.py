import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only

# Assuming taming imports are correct relative to your project structure
from taming.data.utils import AsegDatasetWithAugmentation3D

import math
import signal
from pytorch_lightning.callbacks import TQDMProgressBar
from tqdm import tqdm
import functools # Needed for NLayerDiscriminator3D_Large potentially

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
print(f'Changing working directory to the current script’s absolute path: {dname}')
os.chdir(dname)

# ------------------------------------------------------------------------------
# 1) CUSTOM TQDM PROGRESS BAR (Keep as is)
# ------------------------------------------------------------------------------
import sys
import shutil
# (Your CustomTQDMProgressBar code remains unchanged here)
class CustomTQDMProgressBar(TQDMProgressBar):
    def __init__(
        self,
        refresh_rate=1,
        process_position=0,
        metrics_per_line=4,  # How many metrics per line
        metric_groups=5      # Max number of metric lines
    ):
        super().__init__(refresh_rate, process_position)
        self.metrics_per_line = metrics_per_line
        self.metric_groups = metric_groups

        # Check if we're in an interactive terminal
        self.is_interactive = sys.stdout.isatty()

        # Track seen batches to avoid duplicate processing
        self.train_processed_batches = set()
        self.val_processed_batches = set()

    def _get_gpu_stats(self):
        """Helper to get GPU memory stats"""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            mem_allocated = torch.cuda.memory_allocated(device)
            mem_reserved = torch.cuda.memory_reserved(device)
            total_mem = torch.cuda.get_device_properties(device).total_memory
            alloc_GB = mem_allocated / (1024 ** 3)
            resv_GB = mem_reserved / (1024 ** 3)
            alloc_percent = mem_allocated / total_mem * 100
            resv_percent = mem_reserved / total_mem * 100
            return {
                "GPU-A": f"{alloc_GB:.1f}GB({alloc_percent:.0f}%)",
                "GPU-R": f"{resv_GB:.1f}GB({resv_percent:.0f}%)"
            }
        return {}

    def _format_value(self, value):
        """Format a value nicely"""
        if isinstance(value, torch.Tensor):
            value = value.item()

        # Format float values
        if isinstance(value, float):
            if abs(value) == 0.0: # Avoid -0.00e+00
                 return "0.00e+00"
            elif abs(value) < 0.0001 or abs(value) >= 10000:
                return f"{value:.2e}"
            elif abs(value) < 0.01:
                 return f"{value:.4f}" # More precision for small numbers
            elif abs(value) < 1:
                return f"{value:.4f}"
            elif abs(value) < 10:
                return f"{value:.3f}"
            else:
                return f"{value:.2f}"
        return str(value)

    def _get_formatted_metrics(self, trainer, prefix="train"):
        """Get all metrics with given prefix and format them"""
        metrics = {}

        # Get GPU stats first
        metrics.update(self._get_gpu_stats())

        # Get all metrics from trainer logged metrics
        # Filter out epoch-level metrics for step display if needed
        logged = trainer.logged_metrics
        step_metrics = {k: v for k, v in logged.items() if 'step' in k or ('epoch' not in k and k.startswith(f'{prefix}/'))}
        epoch_metrics = {k: v for k, v in logged.items() if 'epoch' in k and k.startswith(f'{prefix}/')}

        # Prefer step metrics if available, otherwise use epoch metrics
        metrics_to_display = step_metrics if step_metrics else epoch_metrics

        for key, value in metrics_to_display.items():
            # Smart key shortening
            parts = key.split('/')
            short_key = parts[-1] # Get last part (metric name)

            # Remove suffixes added by PL
            suffixes_to_remove = ['_step', '_epoch']
            for suffix in suffixes_to_remove:
                if short_key.endswith(suffix):
                    short_key = short_key[:-len(suffix)]

            # Common abbreviations
            short_key = short_key.replace("total_loss", "loss").replace("logvar", "lvar")
            short_key = short_key.replace("discriminator", "D").replace("generator", "G")
            short_key = short_key.replace("learning_rate", "lr")
            short_key = short_key.replace("gradient_norm", "grad")
            short_key = short_key.replace("reconstruction", "rec")
            short_key = short_key.replace("perceptual", "perc")
            short_key = short_key.replace("logits_real", "Lreal").replace("logits_fake", "Lfake")
            short_key = short_key.replace("factor", "fac").replace("weight", "w")

            formatted = self._format_value(value)
            metrics[f"{prefix[0]}_{short_key}"] = formatted # e.g., t_loss, v_rec_loss

        # Add learning rate(s) if available
        if hasattr(trainer.lr_scheduler_configs, '__iter__'): # Check if iterable
            for i, lr_config in enumerate(trainer.lr_scheduler_configs):
                 if lr_config.scheduler:
                     current_lr = lr_config.scheduler.optimizer.param_groups[0]['lr']
                     metrics[f'lr{i}'] = self._format_value(current_lr)
        elif hasattr(trainer, 'optimizers') and trainer.optimizers:
             for i, opt in enumerate(trainer.optimizers):
                  if opt and opt.param_groups:
                       current_lr = opt.param_groups[0]['lr']
                       metrics[f'lr{i}'] = self._format_value(current_lr)


        return metrics

    def _print_metrics(self, metrics):
        """Print metrics in a structured way"""
        if not metrics: return 0
        metric_items = list(metrics.items())
        lines_printed = 0

        for i in range(0, len(metric_items), self.metrics_per_line):
            if i // self.metrics_per_line >= self.metric_groups:
                remaining = len(metric_items) - i
                if remaining > 0:
                    print(f"  ... and {remaining} more metrics")
                break

            group = metric_items[i:i + self.metrics_per_line]
            line_parts = [f"{k}={v}" for k, v in group]
            metric_line = ", ".join(line_parts)
            print(f"  {metric_line}")
            lines_printed += 1

        return lines_printed

    # --- Keep the on_..._start and on_..._batch_end methods as you had them ---
    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        self.train_processed_batches.clear()
        # Optionally print epoch start message here if needed
        # print(f"\n--- Starting Epoch {trainer.current_epoch} ---")

    def on_validation_start(self, trainer, pl_module):
        super().on_validation_start(trainer, pl_module)
        self.val_processed_batches.clear()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        batch_id = f"{trainer.current_epoch}_{batch_idx}"
        if batch_id in self.train_processed_batches: return
        self.train_processed_batches.add(batch_id)

        # Use self.is_interactive to decide whether to update progress bar
        # TQDM handles this internally now, but let's update metrics display

        # Update metrics display only occasionally or at end
        is_last_batch = (batch_idx + 1) == trainer.num_training_batches
        should_update_metrics = is_last_batch or (self.refresh_rate != 0 and (batch_idx + 1) % (self.refresh_rate * 10) == 0) # Update less often

        if should_update_metrics and self.main_progress_bar.total > 0:
            metrics = self._get_formatted_metrics(trainer, prefix="train")
            if metrics:
                 # We don't need to manually print if TQDM postix works
                 self.main_progress_bar.set_postfix(metrics)


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        batch_id = f"val_{batch_idx}_{dataloader_idx}"
        if batch_id in self.val_processed_batches: return
        self.val_processed_batches.add(batch_id)

        # Update val bar postfix
        if self.val_progress_bar and self.val_progress_bar.total > 0:
             metrics = self._get_formatted_metrics(trainer, prefix="val")
             if metrics:
                  self.val_progress_bar.set_postfix(metrics)


# ------------------------------------------------------------------------------
# 2) HELPER FUNCTIONS (Keep as is)
# ------------------------------------------------------------------------------
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    # Ensure the module is imported correctly relative to the project structure
    # This assumes your 'taming' directory is importable from where you run the script
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not isinstance(config, dict) and not OmegaConf.is_dict(config):
         # Handle cases where config might be a list or other type unexpectedly
         raise TypeError(f"Expected config to be a dictionary, got {type(config)}")
    if "target" not in config:
        # If target is missing, assume it's just a dict of params and return it.
        # This might happen for nested configs like 'trainer.params'
        # However, top-level objects (model, data, logger) *must* have a target.
        # Let's be stricter: only allow instantiation if target exists.
        raise KeyError("Expected key `target` to instantiate.")

    params = config.get("params", dict())
    if params is None: # Handle explicit null params
         params = dict()

    # Ensure params is a dict for the **kwargs expansion
    if not isinstance(params, dict) and not OmegaConf.is_dict(params):
         raise TypeError(f"Expected params for target {config['target']} to be a dictionary, got {type(params)}")

    # Convert OmegaConf dictionary to primitive dict if needed before passing as **kwargs
    if OmegaConf.is_dict(params):
        params = OmegaConf.to_container(params, resolve=True)

    return get_obj_from_str(config["target"])(**params)


# ------------------------------------------------------------------------------
# 3) ARGPARSE UTILITIES (Add --load_mode)
# ------------------------------------------------------------------------------
def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool): return v
        if v.lower() in ("yes", "true", "t", "y", "1"): return True
        elif v.lower() in ("no", "false", "f", "n", "0"): return False
        else: raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-n", "--name", type=str, default="", help="postfix for logdir")
    parser.add_argument("-r", "--resume", type=str, default="", help="Resume from logdir or checkpoint file path (used for weight loading)")
    parser.add_argument("-b", "--base", nargs="*", metavar="base_config.yaml", default=list(),
                        help="paths to base configs. Loaded left-to-right. Required if not resuming OR if resuming but want to override configs.")
    parser.add_argument("-t", "--train", type=str2bool, default=False, help="train")
    parser.add_argument("--no-test", type=str2bool, default=True, help="disable test")
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument("-d", "--debug", type=str2bool, default=False, help="enable post-mortem debugging")
    parser.add_argument("-s", "--seed", type=int, default=23, help="seed for seed_everything")
    parser.add_argument("-f", "--postfix", type=str, default="", help="post-postfix for default name")
    # --- ADDED ARGUMENT ---
    parser.add_argument(
        "--load_mode",
        type=str,
        default="both",
        choices=["both", "generator"],
        help="Weight loading mode: 'both' (load all including discriminator, requires compatible checkpoint) or 'generator' (load generator only, reset discriminator/FiLM)",
    )
    # --- End Added ---
    parser.add_argument("--gpus", type=str, default="", help="(deprecated) list of GPU ids, e.g. '0,1'")
    return parser

# ------------------------------------------------------------------------------
# 4) DATA MODULES & WRAPPED DATASET (Keep as is)
# ------------------------------------------------------------------------------
# (Your WrappedDataset and DataModuleFromConfig3D code remains unchanged here)
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
             try:
                 with open(txt_file, 'r') as f:
                     lines = [line.strip() for line in f if line.strip() and not line.startswith("#")] # Ignore empty lines and comments
                     combined_paths.extend(lines)
             except FileNotFoundError:
                  print(f"Warning: Dataset list file not found: {txt_file}")
        return combined_paths

    def setup(self, stage=None):
        # Shared instantiation logic
        def instantiate_dataset(config):
            if "params" in config and "dataset_paths_files" in config["params"]:
                txt_files = config["params"]["dataset_paths_files"]
                data_paths = self._load_paths_from_txt_files(txt_files)
                if not data_paths:
                     print(f"Warning: No valid HDF5 paths found from {txt_files}. Dataset will be empty.")
                     # Return an empty dataset or handle appropriately
                     # For now, let AsegDataset handle empty list if it can
                # Instantiate AsegDataset directly, passing necessary args
                # Make sure AsegDatasetWithAugmentation3D signature matches
                return AsegDatasetWithAugmentation3D(
                    dataset_paths=data_paths, # Corrected argument name if needed
                    transforms=config["params"].get("transforms"), # Pass transforms if specified
                    lazy_loading=self.lazy_loading
                )
            else:
                 # Fallback to generic instantiation if path logic isn't used
                 print(f"Warning: Instantiating dataset {config.get('target', 'Unknown')} without explicit path files. Ensure config is correct.")
                 return instantiate_from_config(config)

        if "validation" in self.dataset_configs:
            config = self.dataset_configs["validation"]
            print("Setting up validation dataset...")
            self.val_dataset = instantiate_dataset(config)
            print(f"Validation dataset size: {len(self.val_dataset) if self.val_dataset else 0}")

        if "test" in self.dataset_configs:
            config = self.dataset_configs["test"]
            print("Setting up test dataset...")
            self.test_dataset = instantiate_dataset(config)
            print(f"Test dataset size: {len(self.test_dataset) if self.test_dataset else 0}")

        if "train" in self.dataset_configs:
            config = self.dataset_configs["train"]
            print("Setting up train dataset...")
            if "params" in config and "dataset_paths_files" in config["params"]:
                 # Special handling for train to get full file list if needed
                 train_txt_files = config["params"]["dataset_paths_files"]
                 self.full_train_filelist = self._load_paths_from_txt_files(train_txt_files)
                 print(f"Total training HDF5 files found: {len(self.full_train_filelist)}")
                 if not self.full_train_filelist:
                      print("Warning: No training files found. Training dataloader will be empty.")
                 self.train_dataset = AsegDatasetWithAugmentation3D(
                      dataset_paths=self.full_train_filelist, # Pass correct arg name
                      transforms=self.transform_train,
                      lazy_loading=self.lazy_loading
                 )
            else:
                 # Fallback, might need adjustment based on your exact config structure
                 print(f"Warning: Instantiating train dataset {config.get('target', 'Unknown')} without explicit path files.")
                 self.train_dataset = instantiate_from_config(config)
                 if hasattr(self.train_dataset, 'data_paths'): # Attempt to get paths if available
                      self.full_train_filelist = self.train_dataset.data_paths

            print(f"Train dataset size: {len(self.train_dataset) if self.train_dataset else 0}")


    def train_dataloader(self):
        if not self.train_dataset or len(self.train_dataset) == 0:
             print("Warning: Train dataset is empty or not initialized. Returning None.")
             return None # Avoid error if dataset is empty
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, pin_memory=True,
                          worker_init_fn=self.worker_init_fn)

    def val_dataloader(self):
        if not self.val_dataset or len(self.val_dataset) == 0: return None
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        if not self.test_dataset or len(self.test_dataset) == 0: return None
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False, pin_memory=True)

    def worker_init_fn(self, worker_id):
        # Ensure different random seeds for workers if needed
        worker_seed = torch.initial_seed() % 2**32 # Get seed set by DataLoader
        np.random.seed(worker_seed)
        # random.seed(worker_seed) # If using python's random


# ------------------------------------------------------------------------------
# 5) CALLBACKS: SetupCallback, ImageLogger3D, etc. (Modify SetupCallback slightly)
# ------------------------------------------------------------------------------
class SetupCallback(Callback):
    def __init__(self, resume_ckpt_path, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        # Store the original resume path for potential weight loading
        self.resume_ckpt_path = resume_ckpt_path
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Logdir, ckptdir, cfgdir are now determined externally before callback init
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print(f"Logging Training Process to: {self.logdir}")
            print(f"Saving Checkpoints to: {self.ckptdir}")
            if self.resume_ckpt_path:
                print(f"Attempting to load weights from: {self.resume_ckpt_path}")
            else:
                print("Starting training from scratch (no weights loaded).")

            # Save configs
            print("Project config:")
            print(OmegaConf.to_yaml(self.config)) # Log the final merged config
            OmegaConf.save(self.config, os.path.join(self.cfgdir, f"{self.now}-project.yaml"))

            print("Lightning config:")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save( OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, f"{self.now}-lightning.yaml"))

        # Note: The logic for renaming directories for child runs is removed,
        # as we now always create a unique logdir based on time/name for new/fine-tuning runs.


# (Your make_3d_grid and ImageLogger3D code remains unchanged here)
def make_3d_grid(vol, clamp=True):
    if clamp:
        vol = torch.clamp(vol, -1., 1.)

    if vol.ndim == 5 and all(s > 0 for s in vol.shape): # Check shape validity
        # Assume vol shape is [N, C, D, H, W] - Adjust if yours is different!
        # Make grids from central slices along each axis
        try:
             # Central slice along Depth (D)
             grid_d = torchvision.utils.make_grid(vol[:, :, vol.shape[2] // 2, :, :], nrow=int(math.sqrt(vol.shape[0])))
             # Central slice along Height (H)
             grid_h = torchvision.utils.make_grid(vol[:, :, :, vol.shape[3] // 2, :], nrow=int(math.sqrt(vol.shape[0])))
             # Central slice along Width (W)
             grid_w = torchvision.utils.make_grid(vol[:, :, :, :, vol.shape[4] // 2], nrow=int(math.sqrt(vol.shape[0])))

             # Combine the grids side-by-side (ensure consistent channel dim C=1 or 3)
             # Pad to max height if necessary
             max_h = max(grid_d.shape[1], grid_h.shape[1], grid_w.shape[1])
             grid_d = F.pad(grid_d, (0, 0, 0, max_h - grid_d.shape[1])) # Pad bottom
             grid_h = F.pad(grid_h, (0, 0, 0, max_h - grid_h.shape[1])) # Pad bottom
             grid_w = F.pad(grid_w, (0, 0, 0, max_h - grid_w.shape[1])) # Pad bottom

             combined_grid = torch.cat((grid_d, grid_h, grid_w), dim=2) # Concatenate horizontally
             return combined_grid
        except Exception as e:
             print(f"Error in make_3d_grid: {e}. Input shape: {vol.shape}")
             return None # Return None or a placeholder if slicing fails
    elif vol.ndim == 4 and all(s > 0 for s in vol.shape):
        # Assume vol shape is [N, C, H, W] (a 2D image)
        return torchvision.utils.make_grid(vol, nrow=int(math.sqrt(vol.shape[0])))
    else:
        # print(f"Warning: Unsupported tensor dimensions or zero size for grid: {vol.shape}")
        return None # Return None for invalid input

class ImageLogger3D(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True, log_on_train_epoch_end=False, log_on_val_epoch_end=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.clamp = clamp
        self.increase_log_steps = increase_log_steps
        self.log_steps = [2**i for i in range(int(math.log2(self.batch_freq))+1)] if increase_log_steps else [self.batch_freq]
        self.log_on_train_epoch_end = log_on_train_epoch_end
        self.log_on_val_epoch_end = log_on_val_epoch_end

        # Keep only supported loggers
        self.logger_log_images = {
            pl.loggers.TensorBoardLogger: self._tensorboard,
            # Add other supported loggers like WandB if needed
            # pl.loggers.WandbLogger: self._wandb,
        }

    # --- _tensorboard method --- (Keep as is, ensure make_3d_grid handles None)
    @rank_zero_only
    def _tensorboard(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = make_3d_grid(images[k], clamp=self.clamp)
            if grid is None: # Skip if grid creation failed
                 print(f"Warning: Skipping TensorBoard logging for {split}/{k} due to invalid grid.")
                 continue
            tag = f"{split}/{k}_visualization" # Use a distinct tag
            pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)

    # --- log_local3D method --- (Keep as is, ensure make_3d_grid handles None)
    @rank_zero_only
    def log_local3D(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        os.makedirs(root, exist_ok=True)
        for k, vol in images.items():
             grid = make_3d_grid(vol, clamp=self.clamp)
             if grid is None:
                  print(f"Warning: Skipping local saving for {split}/{k} due to invalid grid.")
                  continue

             grid = (grid + 1.0) / 2.0 # Scale to [0, 1] if clamping happened
             grid = grid.permute(1, 2, 0).cpu().numpy() # HWC
             grid = (grid * 255).astype(np.uint8)

             filename = f"{k}_gs-{global_step:06}_e-{current_epoch:06}_b-{batch_idx:06}.png"
             path = os.path.join(root, filename)
             try:
                  Image.fromarray(grid).save(path)
             except Exception as e:
                  print(f"Error saving image {path}: {e}")

    # --- log_img3D method --- (Keep as is)
    def log_img3D(self, pl_module, batch, batch_idx, split="train"):
        if not self.check_frequency(batch_idx, split):
             return # Only log based on frequency or epoch end checks

        if (hasattr(pl_module, "log_images3D") and
                callable(pl_module.log_images3D) and self.max_images > 0):

            logger = pl_module.logger
            is_train = pl_module.training
            if is_train: pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images3D(batch, split=split) # Removed pl_module arg if not needed by method

            # Ensure images dict is not empty
            if not images:
                 if is_train: pl_module.train()
                 return

            for k in images:
                if not isinstance(images[k], torch.Tensor): # Skip non-tensor entries
                     print(f"Warning: Skipping logging for key '{k}' in split '{split}' as it's not a Tensor.")
                     continue
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N].detach().cpu()

            self.log_local3D(logger.save_dir, split, images,
                             pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_func = self.logger_log_images.get(type(logger))
            if logger_log_func:
                logger_log_func(pl_module, images, pl_module.global_step, split)

            if is_train: pl_module.train()

    # --- check_frequency method --- (Modified slightly for clarity)
    def check_frequency(self, batch_idx, split):
        is_val_epoch_end_log = split == 'val' and self.log_on_val_epoch_end and self.trainer.current_epoch > 0
        is_train_epoch_end_log = split == 'train' and self.log_on_train_epoch_end and self.trainer.current_epoch > 0

        # Check if it's the last batch of the validation epoch for epoch end logging
        if split == 'val' and is_val_epoch_end_log:
             # Need to know if it's the last batch. This requires trainer access.
             # We'll tie epoch end logging to batch_idx 0 of the *next* epoch for simplicity,
             # or rely on on_validation_epoch_end hook if preferred.
             # Let's stick to batch frequency for simplicity here unless specifically logging on epoch end.
             pass # Handled by on_validation_epoch_end or on_train_epoch_end

        # Batch frequency check
        if (batch_idx + 1) % self.batch_freq == 0:
            return True
        # Log steps check (used for initial frequent logging)
        if self.increase_log_steps and batch_idx in self.log_steps:
            try: self.log_steps.pop(0)
            except IndexError: pass
            return True
        return False

    # --- Hooks --- (Log based on batch frequency OR epoch end flags)
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Use self.trainer for access
        self.trainer = trainer
        if not trainer.sanity_checking:
            self.log_img3D(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.trainer = trainer
        if not trainer.sanity_checking:
            self.log_img3D(pl_module, batch, batch_idx, split="val")

    # Optional: Add hooks for explicit epoch end logging if needed
    # def on_train_epoch_end(self, trainer, pl_module):
    #     if self.log_on_train_epoch_end:
    #         # Need a sample batch to log - might need to store one
    #         pass
    #
    # def on_validation_epoch_end(self, trainer, pl_module):
    #     if self.log_on_val_epoch_end:
    #          # Need a sample batch to log - might need to store one
    #          pass


# --- Function to parse checkpoint path (modified slightly) ---
def parse_ckpt_path(ckpt_path):
    """Parses a checkpoint path (file or dir) and returns logdir and ckpt file."""
    if not ckpt_path: return None, None
    if not os.path.exists(ckpt_path):
        print(f"Warning: Checkpoint path not found: {ckpt_path}")
        return None, None # Return None if path doesn't exist

    if os.path.isfile(ckpt_path):
        logdir = os.path.dirname(os.path.dirname(ckpt_path)) # ../checkpoints/../ = logdir
        ckpt_file = ckpt_path
    else: # Assumed directory
        logdir = ckpt_path.rstrip("/")
        ckpt_file = os.path.join(logdir, "checkpoints", "last.ckpt")
        if not os.path.exists(ckpt_file):
             # Try finding any .ckpt file if last.ckpt missing
             ckpts = sorted(glob.glob(os.path.join(logdir, "checkpoints", "*.ckpt")))
             if ckpts:
                  ckpt_file = ckpts[-1] # Use the latest one
                  print(f"Warning: last.ckpt not found, using latest: {ckpt_file}")
             else:
                  print(f"Warning: No checkpoint file found in {os.path.join(logdir, 'checkpoints')}")
                  ckpt_file = None # No checkpoint file found
    return logdir, ckpt_file

# ------------------------------------------------------------------------------
# 6) MAIN TRAINING / SCRIPT ENTRYPOINT (Modified Logic)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd()) # Ensure CWD is in path for imports

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    # --- Handle Checkpoint Path and Config Loading ---
    # Store the original resume path argument separately, as opt.resume might be cleared
    resume_ckpt_path_arg = opt.resume if opt.resume else None
    actual_ckpt_to_load = None # Path to the .ckpt file for weight loading
    configs_from_ckpt = []

    if resume_ckpt_path_arg:
        old_logdir, ckpt_file = parse_ckpt_path(resume_ckpt_path_arg)
        if ckpt_file: # Only proceed if a valid checkpoint file was found
            actual_ckpt_to_load = ckpt_file
            # Load configs from the *old* log directory ONLY if user didn't provide new base configs
            if not opt.base and old_logdir:
                 cfg_pattern = os.path.join(old_logdir, "configs", "*-project.yaml")
                 ckpt_cfg_files = sorted(glob.glob(cfg_pattern))
                 if ckpt_cfg_files:
                      configs_from_ckpt = [OmegaConf.load(f) for f in ckpt_cfg_files]
                      print(f"Loading base configuration from checkpoint directory: {old_logdir}")
                 else:
                      print(f"Warning: No *-project.yaml config found in checkpoint dir: {old_logdir}/configs")
            elif opt.base:
                 print("Using config files provided via --base, ignoring checkpoint's config.")
            else:
                 print("Warning: Resuming without providing --base configs and couldn't find config in checkpoint dir.")
        else:
            print(f"Could not resolve a valid checkpoint file from --resume path: {resume_ckpt_path_arg}. Proceeding without loading weights.")
            resume_ckpt_path_arg = None # Clear it if no valid ckpt found


    # --- Determine Log Directory and Name for *this* run ---
    # Prioritize explicit name, then config base name, for new/fine-tuning runs
    if opt.name:
        nowname = f"{now}_{opt.name}{opt.postfix}"
        logdir = os.path.join("logs", nowname)
    elif opt.base:
        # Use first provided base config name for logdir if no explicit name
        base_name = os.path.splitext(os.path.basename(opt.base[0]))[0]
        nowname = f"{now}_{base_name}{opt.postfix}"
        logdir = os.path.join("logs", nowname)
    elif resume_ckpt_path_arg and opt.load_mode == 'both':
         # True resume: Reuse old logdir if no new name/config provided
         old_logdir, _ = parse_ckpt_path(resume_ckpt_path_arg)
         if old_logdir:
              logdir = old_logdir
              tmp = old_logdir.split("/")
              nowname = tmp[tmp.index("logs") + 1] if "logs" in tmp else os.path.basename(old_logdir)
              print(f"Resuming training in existing log directory: {logdir}")
         else: # Fallback if old logdir couldn't be determined
              nowname = f"{now}_resumed{opt.postfix}"
              logdir = os.path.join("logs", nowname)
              print(f"Warning: Could not determine original logdir for resume. Using new logdir: {logdir}")
    else:
        # Fallback name if nothing else specified
        nowname = f"{now}_unnamed{opt.postfix}"
        logdir = os.path.join("logs", nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    # Create directories (rank_zero logic moved to SetupCallback)

    seed_everything(opt.seed)

    try:
        # --- Load and Merge Configs ---
        # Start with configs found in checkpoint dir (if applicable and no new base provided)
        configs_to_merge = configs_from_ckpt
        # Add configs from --base argument (these take precedence over checkpoint configs)
        if opt.base:
             configs_to_merge.extend([OmegaConf.load(cfg) for cfg in opt.base])

        if not configs_to_merge:
            parser.error("No configuration found. Please provide config files via --base or ensure resuming from a valid checkpoint with configs.")

        # Merge configs (later files override earlier ones)
        config = OmegaConf.merge(*configs_to_merge)

        # Apply command-line overrides
        cli_cfg = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(config, cli_cfg)

        # --- Inject ckpt_path and load_mode into model params ---
        # Ensure model.params exists
        if "params" not in config.model:
             config.model.params = OmegaConf.create()

        config.model.params.ckpt_path = actual_ckpt_to_load # The .ckpt file path or None
        config.model.params.load_pretrained_weights_mode = opt.load_mode

        # Pop lightning config
        lightning_config = config.pop("lightning", OmegaConf.create())
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        trainer_kwargs = OmegaConf.to_container(trainer_config, resolve=True) # Resolve interpolations

        # Handle device/GPU setup carefully
        # Remove deprecated 'gpus' key if present
        trainer_kwargs.pop("gpus", None)
        if hasattr(opt, "gpus") and opt.gpus: # Still handle deprecated command line arg
             print("Warning: --gpus argument is deprecated. Use trainer configuration in YAML.")
             # Basic parsing for backward compatibility - prefer config if both specified
             if "accelerator" not in trainer_kwargs and "devices" not in trainer_kwargs:
                  gpus_val = opt.gpus.strip()
                  if gpus_val == "-1":
                       trainer_kwargs["accelerator"] = "gpu"
                       trainer_kwargs["devices"] = -1
                  elif gpus_val:
                       devs = [int(x.strip()) for x in gpus_val.split(",") if x.strip().isdigit()]
                       if devs:
                            trainer_kwargs["accelerator"] = "gpu"
                            trainer_kwargs["devices"] = devs if len(devs) > 1 else devs[0]

        # Default to CPU if no accelerator/devices specified
        if "accelerator" not in trainer_kwargs:
            print("No accelerator specified, defaulting to CPU.")
            trainer_kwargs["accelerator"] = "cpu"
            trainer_kwargs["devices"] = 1 # Default devices for CPU

        # Ensure 'devices' is set if accelerator is 'gpu'
        if trainer_kwargs.get("accelerator") == "gpu" and "devices" not in trainer_kwargs:
             print("GPU accelerator specified but no devices. Defaulting to devices=1.")
             trainer_kwargs["devices"] = 1


        # --- Instantiate Model (will handle weight loading internally) ---
        print(f"Instantiating model: {config.model.target}")
        model = instantiate_from_config(config.model)

        # --- Callbacks & Logger ---
        # (Logger instantiation - Keep as is)
        default_logger_cfgs = {
            "tensorboard": { "target": "pytorch_lightning.loggers.TensorBoardLogger",
                           "params": {"save_dir": logdir, "name": "tensorboard"}}
            # Add wandb etc. here if needed
        }
        default_logger_cfg = default_logger_cfgs.get("tensorboard") # Default to TensorBoard
        logger_cfg = lightning_config.get("logger", OmegaConf.create())
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg) if default_logger_cfg else logger_cfg
        logger = instantiate_from_config(logger_cfg) if logger_cfg else None

        # (Checkpoint Callback - Keep as is)
        default_modelckpt_cfg = {
             "target": "pytorch_lightning.callbacks.ModelCheckpoint",
             "params": {"dirpath": ckptdir, "filename": "{epoch:06}-{step:09}-{val_rec_loss:.4f}",
                        "verbose": True, "save_last": True, "save_top_k": 3, "monitor": "val_rec_loss" }
        }
        if hasattr(model, "monitor") and model.monitor: # Allow model to override monitor
             print(f"Using monitor metric from model: {model.monitor}")
             default_modelckpt_cfg["params"]["monitor"] = model.monitor
        modelckpt_cfg = lightning_config.get("modelcheckpoint", OmegaConf.create())
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        checkpoint_callback = instantiate_from_config(modelckpt_cfg)


        # (Other Callbacks - Pass correct resume_ckpt_path to SetupCallback)
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main_hdf5_3d_v6_3.SetupCallback",
                "params": {
                    "resume_ckpt_path": actual_ckpt_to_load, # Pass the actual .ckpt path
                    "now": now, "logdir": logdir, "ckptdir": ckptdir, "cfgdir": cfgdir,
                    "config": config, "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "main_hdf5_3d_v6_3.ImageLogger3D",
                "params": {"batch_frequency": 1000, "max_images": 4, "clamp": True}
            },
            "learning_rate_logger": {
                "target": "pytorch_lightning.callbacks.LearningRateMonitor",
                "params": {"logging_interval": "step"}
            },
            # Use Custom TQDM Bar
             "progress_bar": {
                  "target": "main_hdf5_3d_v6_3.CustomTQDMProgressBar"
             },
        }
        # Remove default TQDM if custom one is used
        trainer_kwargs["enable_progress_bar"] = False # Disable default PL bar if using custom callback

        callbacks_cfg = lightning_config.get("callbacks", OmegaConf.create())
        # Ensure we don't merge the default progress bar if a custom one is defined
        if "progress_bar" in callbacks_cfg or "progress_bar" in default_callbacks_cfg:
             default_callbacks_cfg.pop("progress_bar", None) # Avoid conflict if user defines one

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        callbacks_list = [instantiate_from_config(callbacks_cfg[cb_key]) for cb_key in callbacks_cfg]
        callbacks_list.append(checkpoint_callback) # Ensure checkpoint callback is always included

        # --- Instantiate Trainer ---
        # Remove resume_from_checkpoint from kwargs, handle explicitly below
        trainer_kwargs.pop("resume_from_checkpoint", None)
        trainer = Trainer(logger=logger, callbacks=callbacks_list, **trainer_kwargs)

        # --- Instantiate Data ---
        print("Instantiating data module...")
        data = instantiate_from_config(config.data)
        # Note: data.setup() is called internally by trainer.fit()

        # --- Configure Learning Rate ---
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if trainer_kwargs.get("accelerator") == "gpu":
            devices_val = trainer_kwargs.get("devices", 1)
            if isinstance(devices_val, int): ngpu = torch.cuda.device_count() if devices_val == -1 else devices_val
            elif isinstance(devices_val, (list, tuple)): ngpu = len(devices_val)
            else: ngpu = 1 # Should not happen with PTL validation
        else: ngpu = 1

        accumulate_grad_batches = 1 # Default
        if hasattr(config.model, "params") and hasattr(config.model.params, "trainconfig") and hasattr(config.model.params.trainconfig, "accumulate_grad_batches_g"):
             accumulate_grad_batches = config.model.params.trainconfig.accumulate_grad_batches_g or 1

        # Apply LR scaling only if not fine-tuning from a checkpoint where LR might be different
        # A common strategy is to set a specific lower LR in the fine-tuning config directly.
        # Let's respect the base_learning_rate set in the *final* config.
        # Scaling logic can sometimes be complex, rely on config LR if possible.
        # model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        # print(f"Setting learning rate based on scaling: {model.learning_rate:.2e}")
        # Alternative: Use LR directly from config if fine-tuning
        if actual_ckpt_to_load and opt.load_mode == 'generator':
             model.learning_rate = base_lr # Use the potentially lower LR from fine-tuning config
             print(f"Fine-tuning: Using learning rate directly from config: {model.learning_rate:.2e}")
        else: # Scaling for training from scratch or full resume
             model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
             print(f"Setting learning rate based on scaling: {model.learning_rate:.2e} = {accumulate_grad_batches} * {ngpu} * {bs} * {base_lr:.2e}")


        # --- Setup Signals ---
        # (Keep signal handling as is)
        def melk(*args, **kwargs):
             if trainer.global_rank == 0:
                  print("Summoning checkpoint via USR1 signal.")
                  ckpt_path = os.path.join(ckptdir, "last.ckpt")
                  trainer.save_checkpoint(ckpt_path)
        def divein(*args, **kwargs):
             if trainer.global_rank == 0:
                  import pudb; pudb.set_trace()
        signal.signal(signal.SIGUSR1, melk); signal.signal(signal.SIGUSR2, divein)

        # --- Run Training ---
        # Determine checkpoint path for trainer.fit (only for resuming trainer state)
        trainer_ckpt_path = actual_ckpt_to_load if opt.load_mode == 'both' and actual_ckpt_to_load else None

        if opt.train:
            print(f"Starting training...")
            if trainer_ckpt_path:
                 print(f"  Resuming trainer state from: {trainer_ckpt_path}")
            elif actual_ckpt_to_load:
                 print(f"  Fine-tuning with weights loaded from: {actual_ckpt_to_load} (load_mode='{opt.load_mode}')")
            else:
                 print(f"  Training from scratch.")

            try:
                trainer.fit(model, datamodule=data, ckpt_path=trainer_ckpt_path)
            except Exception:
                 # Save checkpoint on error only if rank 0
                 if trainer.is_global_zero: melk()
                 raise # Re-raise the exception

        # --- Run Testing ---
        # (Keep testing logic as is, ensure ckpt_path is handled correctly if needed)
        if not opt.no_test and not trainer.interrupted:
             print("Starting testing...")
             # Load best checkpoint for testing unless resuming fully
             test_ckpt_path = checkpoint_callback.best_model_path if checkpoint_callback and checkpoint_callback.best_model_path and opt.load_mode != 'both' else trainer_ckpt_path
             if not test_ckpt_path and actual_ckpt_to_load: # Fallback if best path not found during fine-tune
                  test_ckpt_path = actual_ckpt_to_load
             print(f"  Using checkpoint for testing: {test_ckpt_path if test_ckpt_path else 'No specific checkpoint specified for test.'}")
             # Note: trainer.test loads weights internally from the specified ckpt_path
             # If None is passed, it uses the model currently in memory.
             trainer.test(model, datamodule=data, ckpt_path=test_ckpt_path if test_ckpt_path else None)


    except Exception as e:
        if opt.debug and trainer.is_global_zero:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise e
    finally:
        # Clean up processes or move logs on debug
        if opt.debug and trainer.is_global_zero:
             if not actual_ckpt_to_load or opt.load_mode=='generator': # Treat fine-tune debug like new run
                  dst = os.path.join("debug_runs", nowname)
                  try:
                       os.makedirs(os.path.dirname(dst), exist_ok=True)
                       os.rename(logdir, dst)
                       print(f"Debug run logs moved to: {dst}")
                  except Exception as mv_err:
                       print(f"Error moving debug logs: {mv_err}")