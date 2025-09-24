import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
from taming.data.utils import custom_collate, AsegDatasetWithAugmentation

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


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
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )

    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
                
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
import math

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
import math
import os

# Make sure these imports match your actual code structure
from main_hdf5 import instantiate_from_config, AsegDatasetWithAugmentation

class DataModuleFromConfig(pl.LightningDataModule):
    """
    DataModule for managing training, validation, and test datasets.
    Handles lazy loading of HDF5 files to manage memory usage.
    """
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None, transform_train=None,
                 seed=23):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = {}
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.transform_train = transform_train
        self.seed = seed

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
        """Utility function to read all .txt files and return a combined list of HDF5 file paths."""
        combined_paths = []
        for txt_file in txt_file_list:
            with open(txt_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                combined_paths.extend(lines)
        return combined_paths

    def setup(self, stage=None):
        """
        Sets up the datasets for training, validation, and testing.
        """
        # ---------- Validation Dataset Setup ----------
        if "validation" in self.dataset_configs:
            config = self.dataset_configs["validation"]
            # Check inside config["params"] for 'dataset_paths_files'
            if "params" in config and "dataset_paths_files" in config["params"]:
                val_txt_files = config["params"]["dataset_paths_files"]
                val_paths = self._load_paths_from_txt_files(val_txt_files)

                print("For validation, the following HDF5 files were found:")
                for p in val_paths:
                    print(f"  {p}")
                self.val_dataset = AsegDatasetWithAugmentation(val_paths, transforms=None)
            else:
                # Fallback to instantiate_from_config if no dataset_paths_files provided
                self.val_dataset = instantiate_from_config(config)

        # ---------- Test Dataset Setup ----------
        if "test" in self.dataset_configs:
            config = self.dataset_configs["test"]
            if "params" in config and "dataset_paths_files" in config["params"]:
                test_txt_files = config["params"]["dataset_paths_files"]
                test_paths = self._load_paths_from_txt_files(test_txt_files)

                print("For testing, the following HDF5 files were found:")
                for p in test_paths:
                    print(f"  {p}")
                self.test_dataset = AsegDatasetWithAugmentation(test_paths, transforms=None)
            else:
                self.test_dataset = instantiate_from_config(config)

        # ---------- Training Dataset Setup ----------
        if "train" in self.dataset_configs:
            config = self.dataset_configs["train"]
            if "params" in config and "dataset_paths_files" in config["params"]:
                train_txt_files = config["params"]["dataset_paths_files"]
                self.full_train_filelist = self._load_paths_from_txt_files(train_txt_files)

                print("For training, the following HDF5 files were found:")
                for p in self.full_train_filelist:
                    print(f"  {p}")
                print(f"Total training HDF5 files found: {len(self.full_train_filelist)}")

                self.train_dataset = AsegDatasetWithAugmentation(
                    self.full_train_filelist,
                    transforms=self.transform_train
                )
            else:
                raise ValueError("Train dataset configuration must include a 'dataset_paths_files' list.")

    def train_dataloader(self):
        """
        Returns the DataLoader for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=self.worker_init_fn  # Ensure reproducibility
        )

    def val_dataloader(self):
        """
        Returns the DataLoader for the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )

    def test_dataloader(self):
        """
        Returns the DataLoader for the test dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )

    def worker_init_fn(self, worker_id):
        """
        Ensures that each worker has a unique seed based on the main seed.
        """
        worker_seed = self.seed + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)


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
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            print(self.config.pretty())
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(self.lightning_config.pretty())
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb,
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        raise ValueError("No way wandb")
        grids = dict()
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grids[f"{split}/{k}"] = wandb.Image(grid)
        pl_module.logger.experiment.log(grids)

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w   ### UNCOMMENT THIS IF THE INPUTS ARE NORMALIZED BETWEEN -1 AND 1

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:  # images['inputs'].shape=torch.Size([1, 1, 192, 192])
            grid = torchvision.utils.make_grid(images[k], nrow=4)

            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = f"{k}_gs-{global_step:06}_e-{current_epoch:06}_b-{batch_idx:06}.png"
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) and
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, pl_module=pl_module)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
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

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")



if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # Add current working directory to sys.path
    sys.path.append(os.getcwd())

    # Parse arguments
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()

    # Validate arguments
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both. "
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )

    # Handle resume logic
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

    # Directories
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # Initialize and merge configurations
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)

        # Ensure the "lightning" section exists
        if "lightning" not in config:
            config.lightning = OmegaConf.create()
        lightning_config = config.pop("lightning", OmegaConf.create())

        # Merge trainer CLI arguments
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if "gpus" not in trainer_config:
            trainer_config.pop("distributed_backend", None)
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # Instantiate model
        model = instantiate_from_config(config.model)

        # Trainer kwargs
        trainer_kwargs = {}

        # Default logger configurations
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                },
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                },
            },
        }
        default_logger_cfg = default_logger_cfgs["testtube"]
        if "logger" not in lightning_config:
            lightning_config.logger = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, lightning_config.logger)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # Default ModelCheckpoint configuration
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            },
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3
        if "modelcheckpoint" not in lightning_config:
            lightning_config.modelcheckpoint = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, lightning_config.modelcheckpoint)
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # Default callbacks configuration
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main_hdf5.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                },
            },
            "image_logger": {
                "target": "main_hdf5.ImageLogger",
                "params": {
                    "batch_frequency": 5000,
                    "max_images": 4,
                    "clamp": True,
                },
            },
            "learning_rate_logger": {
                "target": "pytorch_lightning.callbacks.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                },
            },
        }
        if "callbacks" not in lightning_config:
            lightning_config.callbacks = OmegaConf.create()
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, lightning_config.callbacks)
        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        # Create trainer
        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)

        # Instantiate data
        data = instantiate_from_config(config.data)
        data.setup()

        # Configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        ngpu = len(lightning_config.trainer.get("gpus", "").strip(",").split(",")) if not cpu else 1
        accumulate_grad_batches = lightning_config.trainer.get("accumulate_grad_batches", 1)
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print(f"Setting learning rate to {model.learning_rate:.2e} = "
              f"{accumulate_grad_batches} (accumulate_grad_batches) * "
              f"{ngpu} (num_gpus) * {bs} (batchsize) * {base_lr:.2e} (base_lr)")

        # Signal handling
        def melk(*args, **kwargs):
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                trainer.save_checkpoint(os.path.join(ckptdir, "last.ckpt"))

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb; pudb.set_trace()

        import signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # Run training and testing
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)

    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # Move debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
