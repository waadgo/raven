import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.utilities import rank_zero_only
from taming.data.utils import AsegDatasetWithAugmentation3D
import math
import signal

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
print(f'Changing working directory to the current script’s absolute path: {dname}')
os.chdir(dname)

# ------------------------------------------------------------------------------
# 1) CUSTOM TQDM PROGRESS BAR
# ------------------------------------------------------------------------------
class CustomTQDMProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, pl_module):
        metrics = super().get_metrics(trainer, pl_module)
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            mem_allocated = torch.cuda.memory_allocated(device)
            mem_reserved  = torch.cuda.memory_reserved(device)
            total_mem = torch.cuda.get_device_properties(device).total_memory
            alloc_GB = mem_allocated / (1024 ** 3)
            resv_GB  = mem_reserved / (1024 ** 3)
            alloc_percent = mem_allocated / total_mem * 100
            resv_percent  = mem_reserved / total_mem * 100
            new_metrics = {
                "vram_alloc": f"Alloc: {alloc_GB:.2f}GB ({alloc_percent:.1f}%)",
                "vram_resv": f"Resv: {resv_GB:.2f}GB ({resv_percent:.1f}%)"
            }
            # Add the rest of the metrics (e.g. loss values)
            for key, val in metrics.items():
                if key not in new_metrics:
                    new_metrics[key] = val
            metrics = new_metrics
        return metrics

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        pb = getattr(self, "main_progress_bar", None) or getattr(self, "_progress_bar", None)
        if pb is not None and torch.cuda.is_available():
            device = torch.device("cuda:0")
            mem_allocated = torch.cuda.memory_allocated(device)
            mem_reserved = torch.cuda.memory_reserved(device)
            total_mem = torch.cuda.get_device_properties(device).total_memory
            alloc_GB = mem_allocated / (1024 ** 3)
            resv_GB = mem_reserved / (1024 ** 3)
            alloc_percent = mem_allocated / total_mem * 100
            resv_percent = mem_reserved / total_mem * 100
            pb.set_postfix({
                'vram_alloc': f"Alloc: {alloc_GB:.2f}GB ({alloc_percent:.1f}%)",
                'vram_resv': f"Resv: {resv_GB:.2f}GB ({resv_percent:.1f}%)"
            })

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        pb = getattr(self, "val_progress_bar", None) or getattr(self, "_progress_bar", None)
        if pb is not None and torch.cuda.is_available():
            device = torch.device("cuda:0")
            mem_allocated = torch.cuda.memory_allocated(device)
            mem_reserved = torch.cuda.memory_reserved(device)
            total_mem = torch.cuda.get_device_properties(device).total_memory
            alloc_GB = mem_allocated / (1024 ** 3)
            resv_GB = mem_reserved / (1024 ** 3)
            alloc_percent = mem_allocated / total_mem * 100
            resv_percent = mem_reserved / total_mem * 100
            pb.set_postfix({
                'vram_alloc': f"Alloc: {alloc_GB:.2f}GB ({alloc_percent:.1f}%)",
                'vram_resv': f"Resv: {resv_GB:.2f}GB ({resv_percent:.1f}%)"
            })
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
    parser.add_argument("-n", "--name", type=str, default="", help="New log folder name to be created (if resuming, will create at same level as resume folder)")
    parser.add_argument("-r", "--resume", type=str, default="", help="Path to resume folder or checkpoint file")
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
        grid1 = torchvision.utils.make_grid(vol[:, :, vol.shape[2] // 2, :, :])
        grid2 = torchvision.utils.make_grid(vol[:, :, :, vol.shape[3] // 2, :])
        grid3 = torchvision.utils.make_grid(vol[:, :, :, :, vol.shape[4] // 2])
        return torch.cat([grid1, grid2, grid3], dim=1)
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
        self.clamp = clamp
        # In this example, we assume a simple logger interface.
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]

    @rank_zero_only
    def log_img3D(self, pl_module, batch, batch_idx, split="train"):
        # Dummy implementation: extract images from batch (assumed to be a dict)
        if "image" not in batch:
            return
        images = batch["image"]
        # Limit the number of images logged.
        images = images[:self.max_images]
        # Create a grid for visualization.
        grid = make_3d_grid(images, clamp=self.clamp)
        # Log grid to your logger; here we simply print a message.
        # print(f"Logging {split} images at step {pl_module.global_step}")
        # For example, if using TensorBoard:
        # pl_module.logger.experiment.add_image(f"{split}/3d_images", grid, global_step=pl_module.global_step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        # Log training images every batch_frequency steps.
        if batch_idx % self.batch_freq == 0:
            self.log_img3D(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        # Log validation images every batch_frequency steps.
        if batch_idx % self.batch_freq == 0:
            self.log_img3D(pl_module, batch, batch_idx, split="val")

# ------------------------------------------------------------------------------
# 6) MAIN TRAINING / SCRIPT ENTRYPOINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    # Case 1: Resuming from a checkpoint/folder.
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError(f"Cannot find {opt.resume}")

        # Determine resume_logdir and checkpoint path.
        if os.path.isfile(opt.resume):
            # If resume is a file, extract the log directory from its path.
            paths = opt.resume.split(os.path.sep)
            if "logs" not in paths:
                raise ValueError("The resume checkpoint does not appear to be under a 'logs' directory.")
            idx = len(paths) - paths[::-1].index("logs") + 1
            resume_logdir = os.path.join(*paths[:idx])
            ckpt = opt.resume
        else:
            # Resume is a folder; assume it's the log directory.
            resume_logdir = opt.resume.rstrip(os.path.sep)
            ckpt = os.path.join(resume_logdir, "checkpoints", "last.ckpt")

        # Determine new log directory.
        if opt.name:
            # Create a new log folder at the same level as the resume_logdir.
            parent_dir = os.path.dirname(resume_logdir)
            logdir = os.path.join(parent_dir, opt.name)
            nowname = now + "_" + opt.name  # Using exactly the provided name.
        else:
            # If no new name provided, reuse the resume log directory.
            logdir = resume_logdir
            nowname = now + "_" + os.path.basename(resume_logdir)

        # Set the resume_from_checkpoint parameter for Lightning.
        opt.resume_from_checkpoint = ckpt

        # Prepend base configs from the resume folder's configs, if any.
        base_configs = sorted(glob.glob(os.path.join(resume_logdir, "configs", "*.yaml")))
        opt.base = base_configs + opt.base

        # Log the resuming start.
        print(f"Starting training: logging at folder {logdir} but resuming weights from checkpoint {ckpt}")

    # Case 2: Fresh run (no resume).
    else:
        if opt.name:
            nowname = now + "_" + opt.name + opt.postfix
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            nowname = now + "_" + cfg_name + opt.postfix
        else:
            nowname = now + opt.postfix
        logdir = os.path.join("logs", nowname)
        print(f"Starting fresh training: logging at folder {logdir}")

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
        trainer_kwargs.pop("gpus", None)

        if "accelerator" not in trainer_kwargs:
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
                "target": "main_hdf5_3d_v2.SetupCallback",
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
                "target": "main_hdf5_3d_v2.ImageLogger3D",
                "params": {
                    "batch_frequency": 100,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "custom_tqdm_bar": {
                "target": "main_hdf5_3d_v2.CustomTQDMProgressBar",
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
        # Compute effective learning rate based on batch size, number of GPUs, etc.
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

        accumulate_grad_batches = trainer_kwargs.get("accumulate_grad_batches", 1)
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

        # Debug signal handlers.
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
        # Run training and then testing
        # ----------------------------------------------------------------------
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