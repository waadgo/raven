# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:08:13 2024

@author: walte
"""
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
from main_hdf5 import instantiate_from_config
from omegaconf import OmegaConf

def argparser_downscale_wizard(argparser):
    in_name = getattr(argparser, "iname")
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
    if getattr(argparser, "oname") is None:
        oname = basename + suffix + ext
        setattr(argparser, "oname", oname)
    return argparser
    

def gzip_this(in_file):
    in_data = open(in_file, "rb").read() # read the file as bytes
    out_gz = in_file + ".gz" # the name of the compressed file
    gzf = gzip.open(out_gz, "wb") # open the compressed file in write mode
    gzf.write(in_data) # write the data to the compressed file
    gzf.close() # close the file
    # If you want to delete the original file after the gzip is done:
    os.unlink(in_file)
    
def is_anisotropic(z1, z2, z3):
    # find the largest value
    largest = max(z1, z2, z3)
    # determine which axis has the largest value
    if largest == z1:
        irr_pos = "Sagittal"
    elif largest == z2:
        irr_pos = "Coronal"
    else:
        irr_pos = "Axial"
    # find the smallest value
    smallest = min(z1, z2, z3)
    # compare the largest and smallest values
    if largest >= 2 * smallest:
        print("WARNING: Voxel size is at least twice as large in the largest dimension than in the smallest dimension. Will perform denoising only using the "+irr_pos+" plane.")
        return True, irr_pos
    else:
        return False, 0
    
def arguments_setup(sel_option):
    in_name = getattr(sel_option, "iname")
    model_name = "neuroldm_"
    irm = getattr(sel_option, "intensity_range_mode")
    rri = getattr(sel_option, "robust_rescale_input")
    order = getattr(sel_option, "order")
    use_scipy = getattr(sel_option, "use_scipy")
    suffix_type = getattr(sel_option, "suffix_type")
    uf_h = getattr(sel_option, "uf_h")
    uf_w = getattr(sel_option, "uf_w")
    uf_z = getattr(sel_option, "uf_z")
    if getattr(sel_option, "ext") is None:
        fname = Path(in_name)
        basename = os.path.join(fname.parent, fname.stem)
        ext = fname.suffix
        if ext == ".gz":
            fname2 = Path(basename)
            basename = os.path.join(fname2.parent, fname2.stem)
            ext = fname2.suffix + ext
        setattr(sel_option, "ext", ext)
    # if not use_scipy:
    #     setattr(sel_option, "order", 1) #enforce order 1 by default if not using zoom
    order = getattr(sel_option, "order")
    #=====Setting up the filename suffix =====
    if irm == 0:
        suffix_3 = "_irm0"
    elif irm == 1:
        suffix_3 = "_irm1"
    elif irm == 2:
        suffix_3 = "_irm2"
    if rri:
        suffix_4 = "_rri1"
    else:
        suffix_4 = "_rri0"
    suffix_5 = f'_order{order}'
    suffix_6 = f'_scipy{int(use_scipy)}'
    suffix_7 = f'_UpscalingFactor_{uf_h}x{uf_w}x{uf_z}'
    settings_suffix = suffix_3 + suffix_4 + suffix_5 + suffix_6 + suffix_7
    if suffix_type == "detailed":
        suffix = model_name + settings_suffix + ext
    else:
        suffix = model_name + ext
    pathname = os.path.dirname(sys.argv[0])
    model_path = os.path.join(pathname,"model_checkpoints",model_name+".pt")
    
    # set the default suffix name if it was not parsed as an argument
    if getattr(sel_option, "suffix") is None:
        setattr(sel_option, "suffix", suffix)
    
    if getattr(sel_option, "oname") is None:
        setattr(sel_option, "oname", basename + "_" + suffix)
        
    if getattr(sel_option, "iname_new") is None:
        setattr(sel_option, "iname_new", basename + "_preprocessed_" + suffix)
    
    if getattr(sel_option, "noise_info_file") is None:
        setattr(sel_option, "noise_info_file", basename + ".txt")
    
    args = sel_option

    # Now, to print all the values
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    return sel_option

def arguments_setup_ae(sel_option):
    in_name = getattr(sel_option, "iname")
    model_name = getattr(sel_option, "name")
    irm = getattr(sel_option, "intensity_range_mode")
    rri = getattr(sel_option, "robust_rescale_input")
    order = getattr(sel_option, "order")
    use_scipy = getattr(sel_option, "use_scipy")
    # suffix_type = getattr(sel_option, "suffix_type")
    uf_h = getattr(sel_option, "uf_h")
    uf_w = getattr(sel_option, "uf_w")
    uf_z = getattr(sel_option, "uf_z")
    eval_planes = getattr(sel_option, "eval_planes")
    if getattr(sel_option, "ext") is None:
        fname = Path(in_name)
        basename = os.path.join(fname.parent, fname.stem)
        ext = fname.suffix
        if ext == ".gz":
            fname2 = Path(basename)
            basename = os.path.join(fname2.parent, fname2.stem)
            ext = fname2.suffix + ext
        setattr(sel_option, "ext", ext)
    # if not use_scipy:
    #     setattr(sel_option, "order", 1) #enforce order 1 by default if not using zoom
    order = getattr(sel_option, "order")
    #=====Setting up the filename suffix =====
    if irm == 0:
        suffix_3 = "_irm0"
    elif irm == 1:
        suffix_3 = "_irm1"
    elif irm == 2:
        suffix_3 = "_irm2"
    if rri:
        suffix_4 = "_rri1"
    else:
        suffix_4 = "_rri0"
    suffix_7 = f'_UF{uf_h}x{uf_w}x{uf_z}'
    suffix_13 = f'_EP{eval_planes}'
    # settings_suffix = suffix_3 + suffix_4 + suffix_5 + suffix_6 + suffix_7
    settings_suffix = suffix_7 + suffix_13
    # if suffix_type == "detailed":
    #     suffix = model_name + settings_suffix + ext
    # else:
    suffix = model_name + settings_suffix + ext
    pathname = os.path.dirname(sys.argv[0])
    # model_path = os.path.join(pathname,"model_checkpoints",model_name+".pt")
    
    # set the default suffix name if it was not parsed as an argument
    if getattr(sel_option, "suffix") is None:
        setattr(sel_option, "suffix", suffix)
    
    if getattr(sel_option, "oname") is None:
        setattr(sel_option, "oname", basename + "_" + suffix)
        
    if getattr(sel_option, "iname_new") is None:
        setattr(sel_option, "iname_new", basename + "_preprocessed_" + suffix)
    
    # if getattr(sel_option, "noise_info_file") is None:
    #     setattr(sel_option, "noise_info_file", basename + ".txt")
    
    args = sel_option

    # Now, to print all the values
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    return sel_option

def add_noise(x, noise='.'):
        noise_type = noise[0]
        noise_value = float(noise[1:])/100
        if noise_type == 'G':
            noises = np.random.normal(scale=noise_value, size=x.shape)
            # noises = noises.round()
        elif noise_type == 'S':
            noises = np.random.poisson(x * noise_value) / noise_value
            noises = noises - noises.mean(axis=0).mean(axis=0)

        x_noise = abs(x.astype(np.float64) + noises.astype(np.float64))
        return x_noise
    
def filename_wizard(img_filename, save_as, save_as_new_orig):
    fname_in = Path(img_filename)
    fname_out = Path(save_as)
    fname_innew = Path(save_as_new_orig)
    
    basename_in = os.path.join(fname_in.parent, fname_in.stem)
    basename_out = os.path.join(fname_out.parent, fname_out.stem)
    basename_innew = os.path.join(fname_innew.parent, fname_innew.stem)
    
    ext_in = fname_in.suffix
    ext_out = fname_out.suffix
    ext_innew = fname_innew.suffix
    
    if ext_in == ".gz":
        fname_in = Path(basename_in)
        basename_in = os.path.join(fname_in.parent, fname_in.stem)
        ext_in = fname_in.suffix   
        is_gzip_in = True
    else:
        is_gzip_in = False
    if ext_out == ".gz":
        fname_out = Path(basename_out)
        basename_out = os.path.join(fname_out.parent, fname_out.stem)
        ext_out = fname_out.suffix  
        is_gzip_out = True
    else:
        is_gzip_out = False
    if ext_innew == ".gz":
        fname_innew = Path(basename_innew)
        basename_innew = os.path.join(fname_out.parent, fname_innew.stem)
        ext_innew = fname_innew.suffix
        is_gzip_innew = True
    else:
        is_gzip_innew = False
    return basename_in, basename_out, basename_innew, ext_in, ext_out, ext_innew, is_gzip_in, is_gzip_out, is_gzip_innew

def model_loading_wizard_ae(args, AutoencoderKL3D, VQModel3D, logger, noising_steps=30):
    # Construct a pattern for glob using f-string for interpolation
    pattern = f'checkpoints/*{args.name}*.ckpt'
    if "kl" in args.name:
        model = AutoencoderKL3D
        mode = 'klae'
    else:
        model = VQModel3D
        mode = 'vqae'
    list_of_files = glob.glob(pattern)
    if not list_of_files:
        raise FileNotFoundError(f"No checkpoint found using pattern: {pattern}")
    
    # Use the first matching checkpoint
    checkpoint_path = list_of_files[0]

    config_pattern = f'configs/custom_{args.name}.yaml'
    config_files = glob.glob(config_pattern)
    if not config_files:
        raise FileNotFoundError(f"No config file found using pattern: {config_pattern}")
    config_file = config_files[0]

    # 2) Load the config via OmegaConf
    logger.info(f"Loading config from {config_file}")
    config = OmegaConf.load(config_file)
    ddconfig = config.model.params.ddconfig
    embed_dim = config.model.params.embed_dim
    lossconfig = config.model.params.lossconfig
    trainconfig = config.model.params.trainconfig
    logger.info(f"Parsed ddconfig={ddconfig}, embed_dim={embed_dim} from YAML")
    # Optionally combine with the directory of the script if needed
    pathname = os.path.dirname(sys.argv[0])
    full_ckp_path = os.path.join(pathname, checkpoint_path)
    print(f'Searching the configuration of the model at {full_ckp_path}')
    if mode == 'vqae':
        n_embed = config.model.params.n_embed
        model = model.load_from_checkpoint(checkpoint_path=full_ckp_path, n_embed = n_embed, ddconfig=ddconfig,embed_dim=embed_dim, lossconfig = lossconfig, trainconfig = trainconfig)
    else:
        model = model.load_from_checkpoint(checkpoint_path=full_ckp_path, ddconfig=ddconfig, embed_dim=embed_dim, lossconfig = lossconfig, trainconfig = trainconfig)
    model.eval()
    # Put it onto the GPU or CPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # device = torch.device("cpu")
    logger.info("Cuda available: {}, # Available GPUS: {}, "
                "Cuda user disabled (--no_cuda flag): {}, "
                "--> Using device: {}".format(torch.cuda.is_available(),
                                              torch.cuda.device_count(),
                                              args.no_cuda, device))

    if use_cuda and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model_parallel = True
    else:
        model_parallel = False
    model.to(device)
    model.use_ema_for_sampling = args.use_ema
    # model.noising_step = args.noising_step
    model.is_eval = True
    params_model = {'device': device, "use_cuda": use_cuda, "batch_size": args.batch_size,
                    "model_parallel": model_parallel,
                    "use_ema_for_sampling": model.use_ema_for_sampling,
                    # "noising_step": model.noising_step,
                    "is_eval": model.is_eval} #modifications needed?
    return model, params_model

def load_model_from_ckpt(model, ckpt, verbose=False, ignore_keys=[]):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"] if 'state_dict' in pl_sd else pl_sd
    keys = list(sd.keys())
    for k in keys:
        for ik in ignore_keys:
            if ik and k.startswith(ik):
                print("Deleting key {} from state_dict.".format(k))
                del sd[k]
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    print(f'Missing {len(m)} keys and unexpecting {len(u)} keys')
    # model.cuda()
    # model.eval()
    return model