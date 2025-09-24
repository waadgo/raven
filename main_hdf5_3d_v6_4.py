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
from pytorch_lightning.callbacks import TQDMProgressBar
from tqdm import tqdm

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
print(f"Changing working directory to the current script's absolute path: {dname}")
os.chdir(dname)



# --------------------------------------------------------------------------
# 1) CUSTOM TQDM PROGRESS BAR
# --------------------------------------------------------------------------
class CustomTQDMProgressBar(TQDMProgressBar):
    def __init__(self, refresh_rate=1, process_position=0):
        super().__init__(refresh_rate, process_position)
        self._last_batch_idx = -1

    def _is_main_process(self):
        return getattr(self.trainer, "global_rank", 0) == 0

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.disable = not self._is_main_process()  # Disable on non-main processes
        bar.file = sys.stdout
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.disable = not self._is_main_process()  # Disable on non-main processes
        bar.file = sys.stdout
        return bar
        
    # def init_train_tqdm(self):
    #     bar = super().init_train_tqdm()
    #     bar.disable = False
    #     bar.file = sys.stdout
    #     return bar
    
    # def init_validation_tqdm(self):
    #     bar = super().init_validation_tqdm()
    #     bar.disable = False
    #     bar.file = sys.stdout
    #     return bar
    
    def _get_gpu_stats(self):
        """Get GPU stats from the current device only."""
        if not torch.cuda.is_available():
            return {}
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
            return {"GPU": f"{local_rank}", "Mem": "N/A"}
    
    def _get_metric_value(self, key, default="N/A"):
        value = self.trainer.logged_metrics.get(key, default)
        if isinstance(value, torch.Tensor):
            try:
                return f"{value.item():.4f}"
            except:
                return "N/A"
        return value
    
    def _is_main_process(self):
        return getattr(self.trainer, "global_rank", 0) == 0
    
    def _get_progress_bar(self, is_val=False):
        if is_val:
            return getattr(self, "val_progress_bar", None) or getattr(self, "_progress_bar", None)
        return getattr(self, "main_progress_bar", None) or getattr(self, "_progress_bar", None)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if not self._is_main_process() or batch_idx % self.refresh_rate != 0:
            return
        if self._last_batch_idx == batch_idx:
            return
        self._last_batch_idx = batch_idx
        
        pb = self._get_progress_bar(is_val=False)
        if pb is not None:
            gpu_stats = self._get_gpu_stats()
            metrics = {
                "ae": self._get_metric_value("train/aeloss_step"),
                "tot": self._get_metric_value("train/total_loss_step")
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

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
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

# --------------------------------------------------------------------------
# 2) HELPER FUNCTIONS
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# 3) ARGPARSE UTILITIES
# --------------------------------------------------------------------------
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
    parser.add_argument("--gpus", type=str, default="", help="(deprecated) list of GPU ids, e.g. '0,1'")
    return parser

# --------------------------------------------------------------------------
# 4) DATA MODULES & WRAPPED DATASET
# --------------------------------------------------------------------------
class WrappedDataset(Dataset):
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

# --------------------------------------------------------------------------
# 5) CALLBACKS: SetupCallback, ImageLogger3D, etc.
# --------------------------------------------------------------------------
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
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config, os.path.join(self.cfgdir, f"{self.now}-project.yaml"))
            print("Lightning config:")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(
                OmegaConf.create({"lightning": self.lightning_config}),
                os.path.join(self.cfgdir, f"{self.now}-lightning.yaml")
            )
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
        # [N, C, H, W, D]
        grid3 = torchvision.utils.make_grid(vol[:, :, :, :, vol.shape[4] // 2])
        return grid3
    elif vol.ndim == 4:
        return torchvision.utils.make_grid(vol)
    else:
        raise ValueError(f"Unsupported tensor dimensions: {vol.ndim}")

class ImageLogger3D(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
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
        # If using wandb, handle images accordingly
        pass

    @rank_zero_only
    def _tensorboard(self, pl_module, images, batch_idx, split):
        for k in images:
            vol = images[k]
            if vol.ndim == 5:
                grid_3d = make_3d_grid(vol, clamp=self.clamp)
                tag = f"{split}/{k}"
                pl_module.logger.experiment.add_image(tag, grid_3d, global_step=pl_module.global_step)
            elif vol.ndim == 4:
                grid = torchvision.utils.make_grid(vol)
                grid = (grid + 1.0) / 2.0 if self.clamp else grid
                tag = f"{split}/{k}"
                pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)

    @rank_zero_only
    def log_local3D(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k, vol in images.items():
            if vol.ndim != 5:
                continue
            vol = (vol + 1.0) / 2.0
            combined_grid = make_3d_grid(vol, clamp=self.clamp)
            combined_grid = combined_grid.cpu().numpy()
            combined_grid = (combined_grid * 255).astype(np.uint8)
            filename = f"{k}_gs-{global_step:06}_e-{current_epoch:06}_b-{batch_idx:06}.png"
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            if combined_grid.shape[0] in [1, 3]:
                combined_grid = np.transpose(combined_grid, (1, 2, 0))
            Image.fromarray(combined_grid).save(path)

    def log_img3D(self, pl_module, batch, batch_idx, split="train"):
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

            self.log_local3D(pl_module.logger.save_dir, split, images,
                             pl_module.global_step, pl_module.current_epoch, batch_idx)

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

# --------------------------------------------------------------------------
# 6) MAIN TRAINING / SCRIPT ENTRYPOINT
# --------------------------------------------------------------------------
if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    parser = get_parser()
    opt, unknown = parser.parse_known_args()    
    # initialize_training_environment()
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
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli_cfg = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli_cfg)
        lightning_config = config.pop("lightning", OmegaConf.create())
        trainer_config = lightning_config.get("trainer", OmegaConf.create())

        trainer_kwargs = dict(**OmegaConf.to_container(trainer_config))
        trainer_kwargs.pop("resume_from_checkpoint", None)
        trainer_kwargs.pop("strategy", None)

        # Handle old --gpus argument
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
        trainer_kwargs.pop("gpus", None)

        if "accelerator" not in trainer_kwargs:
            trainer_kwargs["accelerator"] = "cpu"
            trainer_kwargs["devices"] = 1

        if trainer_kwargs.get("accelerator", "cpu") == "gpu":
            print(f"Running on GPUs: {trainer_kwargs.get('devices')}")
            cpu = False
        else:
            cpu = True

        # 1) Instantiate model from config
        
        model = instantiate_from_config(config.model)
        # model = fix_autoencoder_kl3d_and_discriminator(model)
            
        # 3) Build callbacks + logger
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
                "target": "main_hdf5_3d_v6_4.SetupCallback",
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
                "target": "main_hdf5_3d_v6_4.ImageLogger3D",
                "params": {
                    "batch_frequency": 1000,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "custom_tqdm_bar": {
                "target": "main_hdf5_3d_v6_4.CustomTQDMProgressBar",
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

        # Initialize the Trainer  CHECK HERE
        # trainer = Trainer(
        #     logger=logger,
        #     callbacks=callbacks_list,
        #     strategy=create_minimal_fsdp_strategy(),
        #     **trainer_kwargs
        # )

        from pytorch_lightning.strategies import FSDPStrategy, DDPStrategy
        import torch.distributed.fsdp.fully_sharded_data_parallel as fsdp

        # Create a proper FSDP strategy with unused parameters detection
        # strategy = FSDPStrategy(
        #     # Don't pass min_num_params directly
        #     # Don't use StateDictType enum directly
        #     # Use simpler parameters
        #     activation_checkpointing=True,
        #     cpu_offload=True,
        #     use_orig_params=True,  # Critical for unused parameters
        # )

        # strategy = DDPStrategy(find_unused_parameters=True)

        strategy = FSDPStrategy(
            # Only essential parameters
            use_orig_params=True  # For unused parameters
        )

        # trainer = Trainer(
        #     logger=logger,
        #     callbacks=callbacks_list,
        #     **trainer_kwargs
        # )

        trainer = Trainer(
            logger=logger,
            callbacks=callbacks_list,
            strategy=strategy,
            **trainer_kwargs
        )

        data = instantiate_from_config(config.data)
        data.setup()

        # LR scaling example
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if trainer_kwargs.get("accelerator", "cpu") == "gpu":
            devices_val = trainer_kwargs.get("devices", 1)
            if isinstance(devices_val, int):
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

        if opt.train:
            try:
                trainer.fit(model, data, ckpt_path=getattr(opt, "resume_from_checkpoint", None))
            except Exception:
                melk()
                raise

        if not opt.no_test and not trainer.interrupted:
            trainer.test(...)

    except Exception as e:
        if opt.debug and "trainer" in locals() and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise e

    finally:
        if opt.debug and not opt.resume and "trainer" in locals() and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)



# try:
#     from torch.distributed.fsdp.wrap import default_auto_wrap_policy
# except ImportError:
#     # Fallback: a simple default policy that always returns True
#     def default_auto_wrap_policy(module, recurse, unwrapped_params):
#         return True

# def conv3d_exclusion_policy(module, recurse, unwrapped_params):
#     # Exclude Conv3d modules from auto-wrapping
#     if isinstance(module, torch.nn.Conv3d):
#         return False
#     # Otherwise, use the (fallback) default policy
#     return default_auto_wrap_policy(module, recurse, unwrapped_params)


# def safe_conv3d_module(model):
#     """
#     Patches certain layers (Linear, BatchNorm3D, GroupNorm) to fix truly empty
#     weights at init-time. No longer patches Conv3d forward() to reshape parameters,
#     because that breaks FSDP full-shard when each rank only has a slice of the weights.
#     """
#     import torch
#     import torch.nn as nn
#     import torch.nn.functional as F
#     import math
#     import numpy as np

#     patched_count = 0  # Count how many modules we patch

#     #
#     # Removed or commented out the "patched_conv3d_forward" function:
#     #
#     # def patched_conv3d_forward(self, input):
#     #     ...
#     #     # This is removed because it breaks FSDP
#     #     pass
#     #

#     # ------------------------------------------------------------------
#     # 1) Patched forward for Linear layers (only if needed)
#     # ------------------------------------------------------------------
#     def patched_linear_forward(self, input):
#         if self.weight.numel() == 0:
#             print(f"WARNING: Fixing broken Linear weight with shape {self.weight.shape}")
#             expected_shape = (self.out_features, self.in_features)
#             new_weight = torch.zeros(expected_shape, device=input.device)
#             nn.init.kaiming_uniform_(new_weight, a=math.sqrt(5))
#             self.weight = nn.Parameter(new_weight)

#         if self.bias is not None:
#             if self.bias.numel() == 0 or self.bias.shape[0] != self.out_features:
#                 print(f"WARNING: Fixing broken Linear bias with shape {self.bias.shape}")
#                 fan_in = self.in_features
#                 bound = 1 / math.sqrt(fan_in)
#                 new_bias = torch.zeros(self.out_features, device=input.device)
#                 nn.init.uniform_(new_bias, -bound, bound)
#                 self.bias = nn.Parameter(new_bias)
#         return F.linear(input, self.weight, self.bias)

#     # ------------------------------------------------------------------
#     # 2) Patched forward for BatchNorm3d (only if needed)
#     # ------------------------------------------------------------------
#     def patched_batchnorm3d_forward(self, input):
#         if self.weight.numel() == 0 or self.weight.shape[0] != self.num_features:
#             print(f"WARNING: Fixing broken BatchNorm3d weight with shape {self.weight.shape}")
#             new_weight = torch.ones(self.num_features, device=input.device)
#             self.weight = nn.Parameter(new_weight)
#         if self.bias.numel() == 0 or self.bias.shape[0] != self.num_features:
#             print(f"WARNING: Fixing broken BatchNorm3d bias with shape {self.bias.shape}")
#             new_bias = torch.zeros(self.num_features, device=input.device)
#             self.bias = nn.Parameter(new_bias)
#         if self.running_mean.numel() == 0 or self.running_mean.shape[0] != self.num_features:
#             print("WARNING: Fixing broken BatchNorm3d running_mean")
#             self.running_mean = torch.zeros(self.num_features, device=input.device)
#         if self.running_var.numel() == 0 or self.running_var.shape[0] != self.num_features:
#             print("WARNING: Fixing broken BatchNorm3d running_var")
#             self.running_var = torch.ones(self.num_features, device=input.device)
#         return F.batch_norm(
#             input, self.running_mean, self.running_var,
#             self.weight, self.bias, self.training,
#             self.momentum, self.eps
#         )

#     # ------------------------------------------------------------------
#     # 3) Patched forward for GroupNorm (only if needed)
#     # ------------------------------------------------------------------
#     def patched_groupnorm_forward(self, input):
#         if self.weight.numel() == 0 or self.weight.shape[0] != self.num_channels:
#             print(f"WARNING: Fixing broken GroupNorm weight with shape {self.weight.shape}")
#             new_weight = torch.ones(self.num_channels, device=input.device)
#             self.weight = nn.Parameter(new_weight)
#         if self.bias is not None:
#             if self.bias.numel() == 0 or self.bias.shape[0] != self.num_channels:
#                 print(f"WARNING: Fixing broken GroupNorm bias with shape {self.bias.shape}")
#                 new_bias = torch.zeros(self.num_channels, device=input.device)
#                 self.bias = nn.Parameter(new_bias)
#         return F.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)

#     # ------------------------------------------------------------------
#     # 4) If your model has a "discriminator" submodule, we can patch
#     #    its conv/batchnorm parameters ONCE at init, but do NOT override forward conv3D
#     # ------------------------------------------------------------------
#     def patch_discriminator(disc):
#         nonlocal patched_count
#         if disc is None:
#             return
#         for name, module in disc.named_modules():
#             if isinstance(module, nn.Conv3d):
#                 # We do NOT set module.forward = patched_conv3d_forward, because
#                 # forcibly reshaping conv weights breaks FSDP.
#                 # Instead, you can do a "once-only" shape check below if needed
#                 pass
#             elif isinstance(module, nn.BatchNorm3d):
#                 module.forward = patched_batchnorm3d_forward.__get__(module, nn.BatchNorm3d)
#                 patched_count += 1

#     # ------------------------------------------------------------------
#     # 5) Recursively patch child modules in your main model
#     # ------------------------------------------------------------------
#     def patch_module(module):
#         nonlocal patched_count

#         # If there's a 'discriminator', patch that
#         if hasattr(module, 'loss') and hasattr(module.loss, 'discriminator'):
#             patch_discriminator(module.loss.discriminator)

#         # Special-case for logvar in a loss module.
#         if hasattr(module, 'loss') and hasattr(module.loss, 'logvar'):
#             logvar = module.loss.logvar
#             if logvar.numel() == 0:
#                 print("WARNING: Fixing empty logvar in loss module")
#                 new_logvar = torch.zeros(1, device=next(module.parameters()).device)
#                 module.loss.logvar = nn.Parameter(new_logvar)
#                 patched_count += 1

#         # Now patch children
#         for name, child in module.named_children():
#             if isinstance(child, nn.Conv3d):
#                 # Optionally do a one-time check if the shape is truly empty or mismatched
#                 ks = child.kernel_size if isinstance(child.kernel_size, tuple) else (child.kernel_size,) * 3
#                 expected_shape = (child.out_channels, child.in_channels) + ks
#                 numel_expected = np.prod(expected_shape)
#                 # If truly empty or mismatch, re-init once
#                 if child.weight.numel() == 0 or child.weight.numel() != numel_expected:
#                     print(f"Fixing {name}: Conv3d weight {child.weight.shape} -> expected {numel_expected} elems")
#                     new_weight = torch.zeros(expected_shape, device=child.weight.device)
#                     nn.init.kaiming_uniform_(new_weight, a=math.sqrt(5))
#                     # Store it flattened as PyTorch typically does, or store as normal
#                     child.weight = nn.Parameter(new_weight)

#                 if child.bias is not None and (
#                     child.bias.numel() == 0 or child.bias.numel() != child.out_channels
#                 ):
#                     print(f"Fixing {name}: Conv3d bias {child.bias.shape} -> expected {child.out_channels}")
#                     fan_in = child.in_channels * ks[0] * ks[1] * ks[2]
#                     bound = 1 / math.sqrt(fan_in)
#                     new_bias = torch.zeros(child.out_channels, device=child.bias.device)
#                     nn.init.uniform_(new_bias, -bound, bound)
#                     child.bias = nn.Parameter(new_bias)

#                 # We do NOT override child.forward with a patched conv3d forward.
#                 # patched_count += 1
#             elif isinstance(child, nn.Linear):
#                 child.forward = patched_linear_forward.__get__(child, nn.Linear)
#                 patched_count += 1
#             elif isinstance(child, nn.BatchNorm3d):
#                 child.forward = patched_batchnorm3d_forward.__get__(child, nn.BatchNorm3d)
#                 patched_count += 1
#             elif isinstance(child, nn.GroupNorm):
#                 child.forward = patched_groupnorm_forward.__get__(child, nn.GroupNorm)
#                 patched_count += 1
#             else:
#                 # Recursively patch deeper modules
#                 patch_module(child)

#     # Run the patching logic
#     patch_module(model)
#     print(f"Patched {patched_count} modules to be FSDP-safe (without overriding Conv3d forward).")
#     return model


# def fix_autoencoder_kl3d_and_discriminator(model):
#     """
#     Fix AutoencoderKL3D and its discriminator for FSDP compatibility:
#       - We still call safe_conv3d_module(model) to do one-time shape checks for
#         any obviously empty weights (Conv3d, BN, Linear).
#       - We do NOT override the forward() method of Conv3d with a reshape call,
#         because that breaks full-shard FSDP.
#       - Then we do additional checks for quant_conv, post_quant_conv, and
#         other user-specific modules.
#     """
#     import torch
#     import torch.nn as nn
#     import math

#     # 1) Patch the model with safe_conv3d_module (which no longer overrides conv forward)
#     model = safe_conv3d_module(model)

#     # 2) Now do additional shape fixes for autoencoder quant_conv, post_quant_conv, etc.
#     if hasattr(model, 'quant_conv') and isinstance(model.quant_conv, nn.Conv3d):
#         w = model.quant_conv.weight
#         if w.numel() == 0:
#             print("Fixing broken quant_conv weights")
#             in_ch = model.quant_conv.in_channels
#             out_ch = model.quant_conv.out_channels
#             kernel_size = model.quant_conv.kernel_size
#             expected_shape = (out_ch, in_ch, *kernel_size)
#             new_weight = torch.zeros(expected_shape, device=next(model.parameters()).device)
#             nn.init.kaiming_uniform_(new_weight, a=math.sqrt(5))
#             model.quant_conv.weight = nn.Parameter(new_weight)

#         if (model.quant_conv.bias is not None
#             and model.quant_conv.bias.numel() == 0):
#             print("Fixing broken quant_conv bias")
#             out_ch = model.quant_conv.out_channels
#             new_bias = torch.zeros(out_ch, device=next(model.parameters()).device)
#             bound = 1 / math.sqrt(model.quant_conv.in_channels * model.quant_conv.kernel_size[0]**3)
#             nn.init.uniform_(new_bias, -bound, bound)
#             model.quant_conv.bias = nn.Parameter(new_bias)

#     if hasattr(model, 'post_quant_conv') and isinstance(model.post_quant_conv, nn.Conv3d):
#         w = model.post_quant_conv.weight
#         if w.numel() == 0:
#             print("Fixing broken post_quant_conv weights")
#             in_ch = model.post_quant_conv.in_channels
#             out_ch = model.post_quant_conv.out_channels
#             kernel_size = model.post_quant_conv.kernel_size
#             expected_shape = (out_ch, in_ch, *kernel_size)
#             new_weight = torch.zeros(expected_shape, device=next(model.parameters()).device)
#             nn.init.kaiming_uniform_(new_weight, a=math.sqrt(5))
#             model.post_quant_conv.weight = nn.Parameter(new_weight)

#         if (model.post_quant_conv.bias is not None
#             and model.post_quant_conv.bias.numel() == 0):
#             print("Fixing broken post_quant_conv bias")
#             out_ch = model.post_quant_conv.out_channels
#             new_bias = torch.zeros(out_ch, device=next(model.parameters()).device)
#             bound = 1 / math.sqrt(model.post_quant_conv.in_channels * model.post_quant_conv.kernel_size[0]**3)
#             nn.init.uniform_(new_bias, -bound, bound)
#             model.post_quant_conv.bias = nn.Parameter(new_bias)

#     # 3) If your loss object has a "discriminator", do shape checks
#     if hasattr(model, 'loss'):
#         # If there's a 3D discriminator
#         if hasattr(model.loss, 'discriminator'):
#             discriminator = model.loss.discriminator
#             for name, module in discriminator.named_modules():
#                 if isinstance(module, nn.Conv3d):
#                     if module.weight.numel() == 0:
#                         print(f"Fixing discriminator Conv3d weights in {name}")
#                         in_ch = module.in_channels
#                         out_ch = module.out_channels
#                         kernel_size = module.kernel_size
#                         expected_shape = (out_ch, in_ch, *kernel_size)
#                         new_weight = torch.zeros(expected_shape, device=next(model.parameters()).device)
#                         nn.init.normal_(new_weight, 0.0, 0.02)  # typical init for Discriminator
#                         module.weight = nn.Parameter(new_weight)

#                     if module.bias is not None and module.bias.numel() == 0:
#                         print(f"Fixing discriminator Conv3d bias in {name}")
#                         out_ch = module.out_channels
#                         new_bias = torch.zeros(out_ch, device=next(model.parameters()).device)
#                         module.bias = nn.Parameter(new_bias)

#                 elif isinstance(module, nn.BatchNorm3d):
#                     if module.weight.numel() == 0:
#                         print(f"Fixing discriminator BatchNorm3d weights in {name}")
#                         num_features = module.num_features
#                         new_weight = torch.ones(num_features, device=next(model.parameters()).device)
#                         nn.init.normal_(new_weight, 1.0, 0.02)
#                         module.weight = nn.Parameter(new_weight)

#                     if module.bias.numel() == 0:
#                         print(f"Fixing discriminator BatchNorm3d bias in {name}")
#                         num_features = module.num_features
#                         new_bias = torch.zeros(num_features, device=next(model.parameters()).device)
#                         module.bias = nn.Parameter(new_bias)

#         # Possibly fix the 'logvar' param in your LPIPSWithDiscriminator
#         if hasattr(model.loss, 'logvar') and model.loss.logvar.numel() == 0:
#             print("Fixing logvar in loss module")
#             logvar_init = 0.0
#             new_logvar = torch.ones(1, device=next(model.parameters()).device) * logvar_init
#             model.loss.logvar = nn.Parameter(new_logvar)

#         # Possibly fix the "perceptual_loss" submodule if it has empty params
#         if hasattr(model.loss, 'perceptual_loss'):
#             perceptual_loss = model.loss.perceptual_loss
#             for pname, param in perceptual_loss.named_parameters():
#                 if param.numel() == 0:
#                     print(f"Fixing empty parameter in perceptual_loss: {pname}")
#                     if 'weight' in pname:
#                         shape = param.shape
#                         new_param = torch.zeros(shape, device=next(model.parameters()).device)
#                         nn.init.kaiming_uniform_(new_param, a=math.sqrt(5))
#                     elif 'bias' in pname:
#                         shape = param.shape
#                         new_param = torch.zeros(shape, device=next(model.parameters()).device)
#                     else:
#                         # default
#                         shape = param.shape
#                         new_param = torch.zeros(shape, device=next(model.parameters()).device)
#                     param.data = new_param

#     # 4) Final check for any leftover empty parameters
#     empty_params = []
#     def check_empty_params(module, path=""):
#         for cname, cparam in module.named_parameters(recurse=False):
#             if cparam.numel() == 0:
#                 empty_params.append(f"{path}.{cname}")
#         for cname, child in module.named_children():
#             check_empty_params(child, f"{path}.{cname}" if path else cname)

#     check_empty_params(model)
#     if empty_params:
#         print(f"WARNING: Found {len(empty_params)} empty parameters after all fixes:")
#         for p in empty_params:
#             print(f"  {p}")

#     return model


# def create_minimal_fsdp_strategy():
#     from pytorch_lightning.strategies import FSDPStrategy

#     # Define an auto_wrap_policy that excludes Conv3d modules
#     def my_auto_wrap_policy(module, recurse, unwrapped_params):
#         # Return False for Conv3d modules (do not auto-wrap them)
#         return not isinstance(module, torch.nn.Conv3d)

#     fsdp_strategy = FSDPStrategy(
#         state_dict_type="full",  # use full state dict (if desired)
#         cpu_offload=False,
#         limit_all_gathers=True,
#         auto_wrap_policy=my_auto_wrap_policy  # <-- Add this policy
#     )
#     print("Created minimal FSDP strategy with Conv3d compatibility")
#     return fsdp_strategy

# def optimize_memory():
#     """Configure PyTorch for better memory management"""
#     import torch
#     import gc
    
#     # Enable memory efficient operations if available
#     if hasattr(torch.backends, 'cudnn'):
#         torch.backends.cudnn.benchmark = False
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.enabled = True
    
#     # Empty cache before starting
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
    
#     print("Applied memory optimizations")
#     return True

# def initialize_training_environment():
#     """
#     Setup everything needed for stable FSDP training with 3D models
#     Call this after imports but before model initialization
#     """
#     import torch
#     import os
    
#     # Apply memory optimizations
#     optimize_memory()
    
#     # Configure environment for better stability
#     os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'  # More verbose FSDP logs
    
#     # Print diagnostic info
#     if torch.cuda.is_available():
#         device_count = torch.cuda.device_count()
#         print(f"Found {device_count} CUDA devices")
#         for i in range(device_count):
#             props = torch.cuda.get_device_properties(i)
#             mem_gb = props.total_memory / (1024**3)
#             print(f"  Device {i}: {props.name}, {mem_gb:.1f} GB memory")
#     else:
#         print("CUDA not available")
    
#     print("Training environment initialized")
#     return True

# def create_minimal_fsdp_strategy():
#     """Creates minimal FSDP strategy focused on stability"""
#     from pytorch_lightning.strategies import FSDPStrategy
    
#     # Create minimal FSDP strategy with only essential parameters
#     fsdp_strategy = FSDPStrategy(
#         state_dict_type="full",
#         cpu_offload=False,
#         limit_all_gathers=True
#     )
    
#     print("Created minimal FSDP strategy with Conv3d compatibility")
#     return fsdp_strategy