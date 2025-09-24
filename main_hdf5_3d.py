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
from taming.data.utils import custom_collate, AsegDatasetWithAugmentation3D
import math
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
print(f'Changing working directory to the current scripts absolute path {dname}')
os.chdir(dname)

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
    

class DataModuleFromConfig3D(pl.LightningDataModule):
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
                self.val_dataset = AsegDatasetWithAugmentation3D(val_paths, transforms=None)
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
                self.test_dataset = AsegDatasetWithAugmentation3D(test_paths, transforms=None)
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

                self.train_dataset = AsegDatasetWithAugmentation3D(
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


def make_3d_grid(vol, clamp=True):
    """
    Given a 5D volume [N, C, H, W, Z], extracts middle slices
    along each axis (H, W, Z), creates 3 grids, then stacks them
    vertically into a single 3×(H×W) grid image.

    Returns a single grid as a 3D Tensor [C, combined_H, combined_W].
    """
    if clamp:
        vol = torch.clamp(vol, -1., 1.)

    # Expecting shape [N, C, H, W, Z]
    # Extract mid-slices along H, W, Z
    N, C, H, W, Z = vol.shape
    mid_H = H // 2
    mid_W = W // 2
    mid_Z = Z // 2

    # Slicing
    # plane1 => slice along H-axis => shape [N, C, W, Z]
    plane1 = vol[:, :, mid_H, :, :]      # [N, C, W, Z]
    # plane2 => slice along W-axis => shape [N, C, H, Z]
    plane2 = vol[:, :, :, mid_W, :]      # [N, C, H, Z]
    # plane3 => slice along Z-axis => shape [N, C, H, W]
    plane3 = vol[:, :, :, :, mid_Z]      # [N, C, H, W]

    # Convert plane1 from [N, C, W, Z] to standard [N, C, H, W]
    # For plane1, treat (W, Z) as (H, W).
    plane1_2d = plane1
    # plane2 from [N, C, H, Z] => treat (H, Z) as (H, W).
    plane2_2d = plane2
    # plane3 is already [N, C, H, W].

    # Use torchvision make_grid on each plane with a default nrow=4
    nrow = 4
    grid1 = torchvision.utils.make_grid(plane1_2d, nrow=nrow)
    grid2 = torchvision.utils.make_grid(plane2_2d, nrow=nrow)
    grid3 = torchvision.utils.make_grid(plane3, nrow=nrow)

    # Shift range from [-1,1] to [0,1] if desired
    grid1 = (grid1 + 1.0) / 2.0
    grid2 = (grid2 + 1.0) / 2.0
    grid3 = (grid3 + 1.0) / 2.0

    # Concatenate the three grids vertically along dim=1 (height).
    # grid shapes: [C, H1, W1], etc. So we want dim=1 to stack them "row by row."
    combined_grid = torch.cat([grid1, grid2, grid3], dim=1)  # shape [C, H_total, W_total]

    return combined_grid


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

        # Example wandb code (not implemented here):
        # for key, grid in grids.items():
        #     array = grid.cpu().numpy()
        #     if array.shape[0] in [1, 3]:
        #         array = np.transpose(array, (1, 2, 0))
        #     wandb.log({key: [wandb.Image(array, caption=key)]}, step=pl_module.global_step)

        print("WandB logging not implemented in this example. Please uncomment and add wandb code if needed.")

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

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img3D(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img3D(pl_module, batch, batch_idx, split="val")



if __name__ == "__main__":

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            idx = len(paths)-paths[::-1].index("logs")+1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs+opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("logs")+1]
    else:
        if opt.name:
            name = "_"+opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_"+cfg_name
        else:
            name = ""
        nowname = now+name+opt.postfix
        logdir = os.path.join("logs", nowname)
    print(f'===== logdir is {logdir}')
    print(f'===== nowname is {nowname}')
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        #trainer_config["distributed_backend"] = "ddp"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:
            del trainer_config["distributed_backend"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        # NOTE wandb < 0.10.0 interferes with shutdown
        # wandb >= 0.10.0 seems to fix it but still interferes with pudb
        # debugging (wrongly sized pudb ui)
        # thus prefer testtube for now
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
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["testtube"]
        logger_cfg = lightning_config.logger or OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        modelckpt_cfg = lightning_config.modelcheckpoint or OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main_hdf5_3d.SetupCallback",
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
                "target": "main_hdf5_3d.ImageLogger3D",
                "params": {
                    "batch_frequency": 3000, #Modify this to change the frequency of logging images
                    "max_images": 4,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "main_hdf5_3d.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    #"log_momentum": True
                }
            },
        }
        callbacks_cfg = lightning_config.callbacks or OmegaConf.create()
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)

        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        
        # data.prepare_data()
        data.setup()

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        else:
            ngpu = 1
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches or 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
            model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb; pudb.set_trace()

        import signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except Exception:
        if opt.debug and trainer.global_rank==0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank==0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)