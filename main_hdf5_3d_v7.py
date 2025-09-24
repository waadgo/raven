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
class CustomTQDMProgressBar(TQDMProgressBar):
    def __init__(self, refresh_rate=1, process_position=0):
        super().__init__(refresh_rate, process_position)
        # Track last batch index to avoid duplicate outputs
        self._last_batch_idx = -1
        
    def init_train_tqdm(self):
        # Get the base progress bar from Lightning
        bar = super().init_train_tqdm()
        # Force TQDM to always output (disable interactive mode)
        bar.disable = False  
        # Ensure output goes to sys.stdout
        bar.file = sys.stdout
        return bar
    
    def init_validation_tqdm(self):
        # Apply same settings to validation bar
        bar = super().init_validation_tqdm()
        bar.disable = False
        bar.file = sys.stdout
        return bar
    
    def _get_gpu_stats(self):
        """Get GPU stats from the current device only"""
        if not torch.cuda.is_available():
            return {}
            
        # Get the local GPU index this process is using
        local_rank = 0
        if hasattr(self.trainer, "local_rank"):
            local_rank = self.trainer.local_rank
        
        device = torch.device(f"cuda:{local_rank}")
        try:
            mem_allocated = torch.cuda.memory_allocated(device)
            mem_reserved = torch.cuda.memory_reserved(device)
            total_mem = torch.cuda.get_device_properties(device).total_memory
            alloc_GB = mem_allocated / (1024 ** 3)
            resv_GB = mem_reserved / (1024 ** 3)
            alloc_percent = mem_allocated / total_mem * 100
            resv_percent = mem_reserved / total_mem * 100
            
            return {
                "GPU": f"{local_rank}",
                "Alloc": f"{alloc_GB:.1f}GB({alloc_percent:.0f}%)",
                "Resv": f"{resv_GB:.1f}GB({resv_percent:.0f}%)"
            }
        except Exception:
            # Fallback if anything goes wrong
            return {"GPU": f"{local_rank}", "Mem": "N/A"}
    
    def _get_metric_value(self, key, default="N/A"):
        """Safely get a metric value with formatting"""
        value = self.trainer.logged_metrics.get(key, default)
        if isinstance(value, torch.Tensor):
            try:
                return f"{value.item():.4f}"
            except:
                return "N/A"
        return value
    
    def _is_main_process(self):
        """Check if this is the main process (rank 0)"""
        return getattr(self.trainer, "global_rank", 0) == 0
    
    def _get_progress_bar(self, is_val=False):
        """Get the appropriate progress bar safely"""
        if is_val:
            return getattr(self, "val_progress_bar", None) or getattr(self, "_progress_bar", None)
        return getattr(self, "main_progress_bar", None) or getattr(self, "_progress_bar", None)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Call parent logic
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        
        # Only update on main process and not too frequently (every refresh_rate)
        if not self._is_main_process() or batch_idx % self.refresh_rate != 0:
            return
            
        # Avoid duplicate updates (can happen in some distributed setups)
        if self._last_batch_idx == batch_idx:
            return
        self._last_batch_idx = batch_idx
        
        pb = self._get_progress_bar(is_val=False)
        if pb is not None:
            # Get GPU stats
            gpu_stats = self._get_gpu_stats()
            
            # Get training metrics
            metrics = {
                "ae": self._get_metric_value("train/aeloss_step"),
                "tot": self._get_metric_value("train/total_loss_step")
            }
            
            # Build postfix
            parts = []
            parts.append(f"GPU:{gpu_stats.get('GPU', '?')}")
            parts.append(f"Alloc:{gpu_stats.get('Alloc', 'N/A')}")
            parts.append(f"ae:{metrics['ae']}")
            parts.append(f"tot:{metrics['tot']}")
            postfix_str = ", ".join(parts)
            
            # Update progress bar
            pb.set_postfix_str(postfix_str)
            elapsed = pb.format_dict.get("elapsed", 0)
            
            # Write to stdout in a controlled manner
            tqdm.write(pb.format_meter(pb.n, pb.total, elapsed, postfix=pb.postfix))

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Similar pattern as train batch end
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        
        if not self._is_main_process() or batch_idx % self.refresh_rate != 0:
            return
            
        pb = self._get_progress_bar(is_val=True)
        if pb is not None:
            gpu_stats = self._get_gpu_stats()
            
            metrics = {
                "ae": self._get_metric_value("val/aeloss_step"),
                "tot": self._get_metric_value("val/total_loss_step") 
            }
            
            parts = []
            parts.append(f"GPU:{gpu_stats.get('GPU', '?')}")
            parts.append(f"Alloc:{gpu_stats.get('Alloc', 'N/A')}")
            parts.append(f"ae:{metrics['ae']}")
            parts.append(f"tot:{metrics['tot']}")
            postfix_str = ", ".join(parts)
            
            pb.set_postfix_str(postfix_str)
            elapsed = pb.format_dict.get("elapsed", 0)
            tqdm.write(pb.format_meter(pb.n, pb.total, elapsed, postfix=pb.postfix))

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
        """
        Called at training/validation batch ends to log a small number of 3D volumes.
        """
        if (self.check_frequency(batch_idx)
                and hasattr(pl_module, "log_images3D")
                and callable(pl_module.log_images3D)
                and self.max_images > 0):
            logger_cls = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images3D(batch, split=split, pl_module=pl_module)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            # 1) Save locally
            self.log_local3D(pl_module.logger.save_dir, split, images,
                             pl_module.global_step, pl_module.current_epoch, batch_idx)

            # 2) Log to the logger (WandB, TensorBoard, etc.)
            logger_log_images = self.logger_log_images.get(logger_cls, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.log_img3D(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.log_img3D(pl_module, batch, batch_idx, split="val")

# ------------------------------------------------------------------------------
# 6) MAIN TRAINING / SCRIPT ENTRYPOINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    if opt.name and opt.resume:
        raise ValueError(
            "Cannot specify both -n/--name and -r/--resume at the same time.\n"
            "Use -n with --resume_from_checkpoint if needed."
        )

    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError(f"Cannot find {opt.resume}")
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            idx = len(paths) - paths[::-1].index("logs") + 1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("logs") + 1]
    else:
        if opt.name:
            name = f"_{opt.name}"
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = f"_{cfg_name}"
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join("logs", nowname)

    print(f"===== logdir is {logdir}")
    print(f"===== nowname is {nowname}")

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # Load and merge configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli_cfg = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli_cfg)
        lightning_config = config.pop("lightning", OmegaConf.create())
        trainer_config = lightning_config.get("trainer", OmegaConf.create())

        # Convert trainer_config to dictionary
        trainer_kwargs = dict(**OmegaConf.to_container(trainer_config))
        trainer_kwargs.pop("resume_from_checkpoint", None)

        # Support old --gpus argument (if provided via opt.gpus)
        if hasattr(opt, "gpus") and opt.gpus:
            gpus_val = opt.gpus
            if gpus_val.strip(",") == "":
                trainer_kwargs["accelerator"] = "cpu"
                trainer_kwargs["devices"] = 1
            else:
                trainer_kwargs["accelerator"] = "gpu"
                devs = [x for x in gpus_val.split(",") if x.isdigit()]
                if not devs:
                    trainer_kwargs["devices"] = 1
                else:
                    trainer_kwargs["devices"] = list(map(int, devs))
        # Also, ensure that any leftover "gpus" key is removed
        trainer_kwargs.pop("gpus", None)

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
                "target": "main_hdf5_3d_v6.SetupCallback",
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
                "target": "main_hdf5_3d_v6.ImageLogger3D",
                "params": {
                    "batch_frequency": 1000,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "custom_tqdm_bar": {
                "target": "main_hdf5_3d_v6.CustomTQDMProgressBar",  # adjust this path accordingly
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