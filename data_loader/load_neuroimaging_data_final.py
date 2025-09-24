# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 17:09:52 2022

@author: walte
"""


# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# IMPORTS
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import h5py
import scipy.ndimage.morphology as morphology
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import sys
import random
import torch
from torchvision import transforms as trf
from skimage.measure import label
from torch.utils.data.dataset import Dataset
from .conform import is_conform, is_conform_itk, conform, check_affine_in_nifti, check_affine_in_nifti_itk, std_pos, conform_std, conform_mask, conform_std_mask, conform_keep_dims, conform_itk, conform_itk2, conform_fix
from data_loader import common
from torch.utils.data.sampler import Sampler
from numpy import array
import os, tempfile
# from scipy.interpolate import Rbf 
from scipy.interpolate import RegularGridInterpolator as rgi
from PIL import Image
# from scipy.interpolate import griddata as rgi
# from scipy.interpolate import interpn as rgi
supported_output_file_formats = ['mgz', 'nii', 'nii.gz']
from datetime import datetime
import matplotlib.pyplot as plt
import math

def torchshow(temp, images_batch):
  # Crear un grid de 2 imágenes horizontales
  fig, axes = plt.subplots(1, 2, figsize=(8, 4))
  # Extraer la rebanada 0 del primer eje de temp y image_batch
  slice_temp = temp[0, 0, :, :].cpu() # Asegurarse de que está en la cpu
  slice_image_batch = images_batch[0, 0, :, :].cpu() # Asegurarse de que está en la cpu
  # Convertir los tensores a numpy arrays
  array_temp = slice_temp.numpy()
  array_image_batch = slice_image_batch.numpy()
  # Mostrar los arrays con imshow en cada imagen
  axes[0].imshow(array_temp)
  axes[1].imshow(array_image_batch)
  # Añadir los títulos de las imágenes
  axes[0].set_title("temp")
  axes[1].set_title("image_batch")
  # Ajustar el espacio entre las imágenes
  plt.tight_layout()
  # Mostrar el grid
  plt.show()
  
# Definir la función volshow
def volshow(batch_tensor, n=120, batch_index=0):
    """
    Visualize three slices of a 2D tensor from a batch.
    
    Parameters:
        batch_tensor (torch.Tensor): Tensor of shape (BS, H, W).
        n (int): Index of the slice to visualize (default: 120).
        batch_index (int): Index of the batch element to visualize (default: 0).
    """
    if not isinstance(batch_tensor, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor.")
    
    # Validate batch index
    if batch_index < 0 or batch_index >= batch_tensor.shape[0]:
        raise ValueError(f"Batch index {batch_index} is out of bounds for the tensor shape {batch_tensor.shape}.")
    
    # Extract the selected batch element
    orig = batch_tensor[batch_index]
    
    # Validate slice index for each axis, default to middle if out of bounds
    H, W = orig.shape
    n_H = n if 0 <= n < H else H // 2
    n_W = n if 0 <= n < W else W // 2
    
    # Convert to NumPy for visualization
    orig = orig.cpu().numpy()
    
    # Create a grid of 3 images
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Show the slice in each orientation
    axes[0].imshow(orig[n_H, :], cmap='gray', aspect='auto')  # Slice along axis 0 (row)
    axes[0].set_title(f"Axis 0, Slice {n_H}")
    axes[1].imshow(orig[:, n_W], cmap='gray', aspect='auto')  # Slice along axis 1 (column)
    axes[1].set_title(f"Axis 1, Slice {n_W}")
    
    # Adjust layout
    plt.tight_layout()
    
    # Calculate statistics
    max_val = np.max(orig)
    min_val = np.min(orig)
    median_val = np.median(orig)
    mean_val = np.mean(orig)
    
    # Add a legend with the statistics
    legend = f"Max: {max_val:.2f}, Min: {min_val:.2f}, Median: {median_val:.2f}, Mean: {mean_val:.2f}"
    fig.suptitle(legend, fontsize=16)
    
    # Show the grid with the legend
    plt.show()
def slice_img(orig, slice = 20):
    """
    Function that receives a 3D freesurfer image and returns a PIL image of the slice position "slice",
    which is the slicing position at the dim=2

    Parameters
    ----------
    orig : TYPE
        DESCRIPTION.
    slice : TYPE, optional
        DESCRIPTION. The default is 20.

    Returns
    -------
    None.

    """
    sample = orig.get_fdata()
    sample = sample[:,:,slice]
    sample = (sample - sample.min()) / (sample.max() - sample.min())*255
    sample = sample.astype(np.uint8)
    sample = Image.fromarray(sample)
    return sample

def reorient_standard_RAS(img):
    ornt = np.array([[0, 1], [1, -1], [2, -1]])
    img_orient = img.as_reoriented(ornt) # re-orient the image
    return img

def chunk(indices, size):
    return torch.split(torch.tensor(indices), size)


# def add_rician_varying(x):
#     shape = x.shape
#     h = shape[0]
#     w = shape[1]
#     c = shape[2]
    
#     map1 = np.ones([3,3,3])
#     map1[1,1,1] = 3
#     nx, ny, nz = (3, 3, 3)
#     xi = np.linspace(1, nx, nx)
#     yi = np.linspace(1, ny, ny)
#     zi = np.linspace(1, nz, nz)
    
#     x1, y1, z1 = np.meshgrid(xi, yi, zi)
    
#     xi2 = np.linspace(1, nx, h)
#     yi2 = np.linspace(1, ny, w)
#     zi2 = np.linspace(1, nz, c)
    
#     x2, y2, z2 = np.meshgrid(xi2, yi2, zi2)
#     interp3 = rgi((y1,x1,z1), map1, np.array([x2,y2,z2]).T, method='cubic')
#     # interp3 = Rbf(x1, y1, z1, map1, x2, y2, z2 function="cubic")
#     final_map = interp3(array([x1, y1, z1]).T)
    
#     return final_map

def add_rician_varying(im):
    def f(x,y,z):
        if x==y and y==z:
            return 3
        else: 
            return 1
        shape = im.shape
        h = shape[0]
        w = shape[1]
        c = shape[2]
        
        nx, ny, nz = (3, 3, 3)
        xi = np.linspace(1, nx, nx)
        yi = np.linspace(1, ny, ny)
        zi = np.linspace(1, nz, nz)
        xg, yg, zg = np.meshgrid(xi, yi, zi, indexing='ij', sparse=True)
        data = np.zeros((3,3,3))
        data = f(xg, yg, zg)
        interp3 = rgi((xi, yi, zi), data, method="cubic")
        
        xi2 = np.linspace(1, nx, h)
        yi2 = np.linspace(1, ny, w)
        zi2 = np.linspace(1, nz, c)
        
        x2, y2, z2 = np.meshgrid(xi2, yi2, zi2)
        final_map = interp3(array([x2, y2, z2]).T)
        
        return final_map
        
def load_and_conform_image(img_filename, interpol=3, logger=None, is_eval = False, intensity_rescaling=True, conform_type = 0, keep_dims = True):
    """
    Function to load MRI image and conform it to UCHAR, RAS orientation and 1mm isotropic voxels size
    (if it does not already have this format)
    :param str img_filename: path and name of volume to read
    :param str conform_type: (0=min_vox_size+max_im_size, 1=std(1.0/256)) 
    :param int interpol: interpolation order for image conformation (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return: nibabel.MGHImage header_info: header information of the conformed image
    :return: np.ndarray affine_info: affine information of the conformed image
    :return: nibabel.MGHImage orig: conformed image
    """
    orig = nib.load(img_filename)
    zoom = orig.header.get_zooms()
    max_orig = orig.get_fdata().max()
    min_orig = orig.get_fdata().min()
    # orig = (orig - orig.min) / (orig.max-orig.min)
    ishape = orig.shape
    if len(orig.shape) == 4:
        orig = orig.slicer[:,:,:,0]
    max_shape = max(ishape)
    if not is_conform(orig):

        if logger is not None:
            if conform_type == 0:
                logger.info('Conforming image to UCHAR, RAS orientation, and '+str(zoom)+'mm isotropic voxels. Final Volume Shape: '+str(max_shape)+'x'+str(max_shape)+'x'+str(max_shape)+' voxels')
            else:
                logger.info('Conforming image to UCHAR, RAS orientation, and 1.0mm isotropic voxels. Final Volume Shape: 256x256x256 voxels')
        else:
            if conform_type == 0:
                print('Conforming image to UCHAR, RAS orientation, and '+str(zoom)+'mm isotropic voxels. Final Volume Shape: '+str(max_shape)+'x'+str(max_shape)+'x'+str(max_shape)+' voxels')
            else:
                print('Conforming image to UCHAR, RAS orientation, and 1.0mm isotropic voxels. Final Volume Shape: 256x256x256 voxels')
        if len(orig.shape) > 3 and orig.shape[3] != 1:
            sys.exit('ERROR: Multiple input frames (' + format(orig.shape[3]) + ') not supported!')

        # Check affine if image is nifti image
        if img_filename[-7:] == ".nii.gz" or img_filename[-4:] == ".nii":
            if not check_affine_in_nifti(orig, logger=logger):
                sys.exit("ERROR: inconsistency in nifti-header. Exiting now.\n")
        
        # conform
        if conform_type == 0:
            orig = conform_fix(orig, interpol, intensity_rescaling=intensity_rescaling, keep_dims = keep_dims)
        elif conform_type == 1:
            orig = conform_std(orig, interpol)
        elif conform_type == 2:
            orig = conform_fix(orig, interpol, intensity_rescaling=intensity_rescaling, keep_dims = keep_dims)

    # Collect header and affine information
    header_info = orig.header
    affine_info = orig.affine
    orig = np.asanyarray(orig.dataobj)
    if is_eval:
        return header_info, affine_info, orig, zoom, max_orig, min_orig
    else:
        return header_info, affine_info, orig
    
def load_and_conform_image_sitk(img_filename, interpol=3, logger=None, is_eval = True, conform_type = 2, intensity_rescaling = True):
    """
    Function to load MRI image and conform it to UCHAR, RAS orientation and 1mm isotropic voxels size
    (if it does not already have this format)
    :param str img_filename: path and name of volume to read
    :param str conform_type: (0: Cubic shape of dims = max(img_filename.shape) and voxdim of minimum voxdim. 1: Cubic shape of dims 256^3. 2: Keep dimensions) 
    :param int interpol: interpolation order for image conformation (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return: nibabel.MGHImage header_info: header information of the conformed image
    :return: np.ndarray affine_info: affine information of the conformed image
    :return: nibabel.MGHImage orig: conformed image
    """
    orig = sitk.ReadImage(img_filename)
    logger.info('Conforming image to RAS orientation... ')
    orig = sitk.DICOMOrient(orig, 'RAS')
    zoom = orig.GetSpacing()
    orig_img = sitk.GetArrayFromImage(orig)
    max_orig = orig_img.max()
    min_orig = orig_img.min()
    orig_img = np.transpose(orig_img, (2, 1, 0)) 
    interp_sitk, _, _ = conform_itk(orig, intensity_rescaling = intensity_rescaling, order=interpol)
    if is_eval:
        return orig, interp_sitk, zoom, max_orig, min_orig
    else:
        return orig, interp_sitk

def load_and_rescale_image_sitk(img_filename, interpol =3, logger=None, is_eval = True,
                                intensity_rescaling = True, 
                                uf_h = 2.0, uf_w = 2.0, uf_z = 2.0, use_scipy = True):
    

    """
    

    Parameters
    ----------
    img_filename : TYPE
        DESCRIPTION.
    interpol : TYPE, optional
        DESCRIPTION. The default is 3.
    logger : TYPE, optional
        DESCRIPTION. The default is None.
    is_eval : TYPE, optional
        DESCRIPTION. The default is True.
    conform_type : TYPE, optional
        DESCRIPTION. The default is 2.
    intensity_rescaling : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    orig = sitk.ReadImage(img_filename)
    # logger.info('Conforming image to RAS orientation... ')
    # orig = sitk.DICOMOrient(orig, 'RAS')
    zoom = orig.GetSpacing()
    orig_img = sitk.GetArrayFromImage(orig)
    max_orig = orig_img.max()
    min_orig = orig_img.min()
    orig_img = np.transpose(orig_img, (2, 1, 0)) 
    interp_sitk = conform_itk(orig, volshow, order=interpol, intensity_rescaling = intensity_rescaling, 
                              uf_h = uf_h, uf_w = uf_w, uf_z = uf_z, use_scipy = use_scipy)
    interp_sitk_data = sitk.GetArrayFromImage(interp_sitk)
    interp_sitk_data = np.transpose(interp_sitk_data, (2, 1, 0)) 
    if is_eval:
        return interp_sitk_data, interp_sitk, zoom, max_orig, min_orig
    else:
        return orig, interp_sitk
    
def load_and_keep_dims(img_filename, interpol=1, logger=None, is_eval = False, conform_type = 0, intensity_rescaling = False):
    """
    Function to load MRI image and conform it to UCHAR, RAS orientation and 1mm isotropic voxels size
    (if it does not already have this format)
    :param str img_filename: path and name of volume to read
    :param str conform_type: (0=min_vox_size+max_im_size, 1=std(1.0/256)) 
    :param int interpol: interpolation order for image conformation (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return: nibabel.MGHImage header_info: header information of the conformed image
    :return: np.ndarray affine_info: affine information of the conformed image
    :return: nibabel.MGHImage orig: conformed image
    """
    orig = nib.load(img_filename)
    
    max_orig = orig.get_fdata().max()
    min_orig = orig.get_fdata().min()
    zoom = orig.header.get_zooms()
    # orig = (orig - orig.min) / (orig.max-orig.min)
    
    
    # orig = conform_keep_dims(orig, interpol, intensity_rescaling)

    # Collect header and affine information
    header_info = orig.header
    affine_info = orig.affine
    orig = np.asanyarray(orig.dataobj)
    if is_eval:
        return header_info, affine_info, orig, zoom, max_orig, min_orig
    else:
        return header_info, affine_info, orig

def load_and_conform_image_mask(img_filename, mask_filename, interpol=1, logger=None, is_eval = False, conform_type = 0):
    """
    Function to load MRI image and conform it to UCHAR, RAS orientation and 1mm isotropic voxels size
    (if it does not already have this format)
    :param str img_filename: path and name of volume to read
    :param str conform_type: (0=min_vox_size+max_im_size, 1=std(1.0/256)) 
    :param int interpol: interpolation order for image conformation (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return: nibabel.MGHImage header_info: header information of the conformed image
    :return: np.ndarray affine_info: affine information of the conformed image
    :return: nibabel.MGHImage orig: conformed image
    """
    
    orig = nib.load(img_filename)
    # affine = orig.affine
    # header = orig.header
    # orig = orig.get_data()
    # if len(orig.shape) == 4:
    #     orig = np.squeeze(orig)
    # orig = nib.MGHImage(orig, affine, header)
    zoom = orig.header.get_zooms()
    if len(zoom) == 4:
        zoom = zoom[:-1]
    mask = nib.load(mask_filename)
    ishape = orig.shape
    if len(ishape) == 4:
        ishape = ishape[:-1]
    max_shape = max(ishape)
    
    # if not is_conform(orig):

        # if logger is not None:
        #     if conform_type == 0:
        #         logger.info('Conforming image to UCHAR, RAS orientation, and '+str(zoom)+'mm isotropic voxels. Final Volume Shape: '+str(max_shape)+'x'+str(max_shape)+'x'+str(max_shape)+' voxels')
        #     else:
        #         logger.info('Conforming image to UCHAR, RAS orientation, and 1.0mm isotropic voxels. Final Volume Shape: 256x256x256 voxels')
        # else:
        #     if conform_type == 0:
        #         print('Conforming image to UCHAR, RAS orientation, and '+str(zoom)+'mm isotropic voxels. Final Volume Shape: '+str(max_shape)+'x'+str(max_shape)+'x'+str(max_shape)+' voxels')
        #     else:
        #         print('Conforming image to UCHAR, RAS orientation, and 1.0mm isotropic voxels. Final Volume Shape: 256x256x256 voxels')
        # if len(orig.shape) > 3 and orig.shape[3] != 1:
        #     sys.exit('ERROR: Multiple input frames (' + format(orig.shape[3]) + ') not supported!')

        # # Check affine if image is nifti image
        # if img_filename[-7:] == ".nii.gz" or img_filename[-4:] == ".nii":
        #     if not check_affine_in_nifti(orig, logger=logger):
        #         sys.exit("ERROR: inconsistency in nifti-header. Exiting now.\n")

        # # conform
        # if conform_type == 0:
        #     orig = conform(orig, interpol)
        #     mask = conform_mask(mask, interpol)
        # elif conform_type == 1:
        #     orig = conform_std(orig, interpol)
        #     mask = conform_std_mask(mask, interpol)
    
    if logger is not None:
        if conform_type == 0:
            logger.info('Conforming image to UCHAR, RAS orientation, and '+str(zoom)+'mm isotropic voxels. Final Volume Shape: '+str(max_shape)+'x'+str(max_shape)+'x'+str(max_shape)+' voxels')
        else:
            logger.info('Conforming image to UCHAR, RAS orientation, and 1.0mm isotropic voxels. Final Volume Shape: 256x256x256 voxels')
    else:
        if conform_type == 0:
            print('Conforming image to UCHAR, RAS orientation, and '+str(zoom)+'mm isotropic voxels. Final Volume Shape: '+str(max_shape)+'x'+str(max_shape)+'x'+str(max_shape)+' voxels')
        else:
            print('Conforming image to UCHAR, RAS orientation, and 1.0mm isotropic voxels. Final Volume Shape: 256x256x256 voxels')
    if len(orig.shape) > 3 and orig.shape[3] != 1:
        sys.exit('ERROR: Multiple input frames (' + format(orig.shape[3]) + ') not supported!')

    # Check affine if image is nifti image
    if img_filename[-7:] == ".nii.gz" or img_filename[-4:] == ".nii":
        if not check_affine_in_nifti(orig, logger=logger):
            sys.exit("ERROR: inconsistency in nifti-header. Exiting now.\n")

    # conform
    if conform_type == 0:
        orig = conform(orig, interpol)
        mask = conform_mask(mask, interpol)
    elif conform_type == 1:
        orig = conform_std(orig, interpol)
        mask = conform_std_mask(mask, interpol)

    # Collect header and affine information
    
    header_info = orig.header
    affine_info = orig.affine
    orig = np.asanyarray(orig.dataobj)
    mask = np.asanyarray(mask.dataobj)
    
    mask = mask>0
    mask = mask.astype(np.uint8)
    
    orig = orig*mask
    
    if is_eval:
        return header_info, affine_info, orig, zoom
    else:
        return header_info, affine_info, orig
    
    

def load_image(img_filename, interpol=1, logger=None, is_eval = False):
    """
    Function to load MRI image and conform it to UCHAR, RAS orientation and 1mm isotropic voxels size
    (if it does not already have this format)
    :param str img_filename: path and name of volume to read
    :param int interpol: interpolation order for image conformation (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return: nibabel.MGHImage header_info: header information of the conformed image
    :return: np.ndarray affine_info: affine information of the conformed image
    :return: nibabel.MGHImage orig: conformed image
    """
    orig = nib.load(img_filename)
    zoom = orig.header.get_zooms()
    orig = std_pos(orig)
    header_info = orig.header
    affine_info = orig.affine
    orig = np.asanyarray(orig.dataobj)
    if is_eval:
        return header_info, affine_info, orig, zoom
    else:
        return header_info, affine_info, orig

def save_image(img_array, affine_info, header_info, save_as):
    """
    Save an image (nibabel MGHImage), according to the desired output file format.
    Supported formats are defined in supported_output_file_formats.

    :param numpy.ndarray img_array: an array containing image data
    :param numpy.ndarray affine_info: image affine information
    :param nibabel.freesurfer.mghformat.MGHHeader header_info: image header information
    :param str save_as: name under which to save prediction; this determines output file format

    :return None: saves predictions to save_as
    """

    assert any(save_as.endswith(file_ext) for file_ext in supported_output_file_formats), \
            'Output filename does not contain a supported file format (' + ', '.join(file_ext for file_ext in supported_output_file_formats) + ')!'

    mgh_img = None
    if save_as.endswith('mgz'):
        mgh_img = nib.MGHImage(img_array, affine_info, header_info)
    elif any(save_as.endswith(file_ext) for file_ext in ['nii', 'nii.gz']):
        mgh_img = nib.nifti1.Nifti1Pair(img_array, affine_info, header_info)

    if any(save_as.endswith(file_ext) for file_ext in ['mgz', 'nii']):
        nib.save(mgh_img, save_as)
    elif save_as.endswith('nii.gz'):
        ## For correct outputs, nii.gz files should be saved using the nifti1 sub-module's save():
        nib.nifti1.save(mgh_img, save_as)

def transform_sagittal(vol, coronal2sagittal=True):
    if coronal2sagittal:
        return np.moveaxis(vol, [0, 1, 2], [2, 1, 0])
    else:
        return np.moveaxis(vol, [0, 1, 2], [2, 1, 0])
    
def transform_axial(vol, coronal2axial=True):
    if coronal2axial:
        return np.moveaxis(vol, [0, 1, 2], [1, 2, 0])
    else:
        return np.moveaxis(vol, [0, 1, 2], [2, 0, 1])
    
def process_second_axis(vol):
    # This function will put the second axis in the first position
    # np.moveaxis(vol, [0, 1, 2], [1, 0, 2])
    vol2 = np.moveaxis(vol, [0, 1, 2], [2, 0, 1])
    return vol2

def process_third_axis(vol):
    # This function will put the third axis in the first position
    # return np.moveaxis(vol, [0, 1, 2], [2, 1, 0])
    vol2 = np.moveaxis(vol, [0, 1, 2], [1, 2, 0])
    return vol2


def transform_coronal(vol, sagittal2coronal=True):
    """
    Function to transform volume into Sagittal axis and back
    :param np.ndarray vol: image volume to transform
    :param bool coronal2sagittal: transform from coronal to sagittal = True (default),
                                transform from sagittal to coronal = False
    :return: np.ndarray: transformed image volume
    """
    if sagittal2coronal:
        return np.moveaxis(vol, [0, 1, 2], [2, 0, 1])
    else:
        return np.moveaxis(vol, [1, 0, 2], [0, 1, 2])


# def get_thick_slices(img_data, slice_thickness=3, anisotropic = False):
#     """
#     Function to extract thick slices from the image 
#     (feed slice_thickness preceeding and suceeding slices to network, 
#     denoise only middle one) Added a padding stage so all the images are the 
#     :param np.ndarray img_data: 3D MRI image read in with nibabel 
#     :param int slice_thickness: number of slices to stack on top and below slice of interest (default=3) 
#     :param bool anisotropic: whether the image has different resolutions along different axes (default=True)
#     :return: np.ndarray img_data_thick: image array containing the extracted slices
#     """
#     h, w, d = img_data.shape
#     img_data_pad = np.expand_dims(np.pad(img_data, ((0, 0), (0, 0), (slice_thickness, slice_thickness)), mode='edge'),
#                                   axis=3)
#     img_data_thick = np.ndarray((h, w, d, 0), dtype=np.uint8)
    
#     # if anisotropic:
#     #     # If anisotropic is True, use only the middle slice and repeat it
#     #     for slice_idx in range(2 * slice_thickness + 1):
#     #         img_data_thick = np.append(img_data_thick, np.expand_dims(img_data, axis=3), axis=3)

#     # else:
#         # If anisotropic is False, use consecutive slices as usual
#     for slice_idx in range(2 * slice_thickness + 1):
#         img_data_thick = np.append(img_data_thick, img_data_pad[:, :, slice_idx:d + slice_idx, :], axis=3)

#     return img_data_thick

def get_thick_slices(img_data, slice_thickness=3):
    """
    Function to extract thick slices from the image 
    (feed slice_thickness preceding and succeeding slices to network, 
    denoise only middle one).

    :param np.ndarray img_data: 3D MRI image read in with nibabel 
    :param int slice_thickness: number of slices to stack on top and below slice of interest (default=3) 
    :return: np.ndarray img_data_thick: image array containing the extracted slices
    """
    h, w, d = img_data.shape
    padded_img = np.pad(img_data, ((0, 0), (0, 0), (slice_thickness, slice_thickness)), mode='edge')
    
    # Preallocate the thick slices array
    thick_slices = np.zeros((h, w, d, 2 * slice_thickness + 1), dtype=img_data.dtype)
    
    # Fill the thick slices array with the appropriate slices
    for i in range(2 * slice_thickness + 1):
        thick_slices[:, :, :, i] = padded_img[:, :, i:i + d]
    
    return thick_slices

def get_noisy_pre_den_pairs(pre_denoised_image, orig):
    """
    Function to extract thick slices from the image 
    (feed slice_thickness preceeding and suceeding slices to network, 
    denoise only middle one) Added a padding stage so all the images are the 
    :param np.ndarray img_data: 3D MRI image read in with nibabel 
    :param int slice_thickness: number of slices to stack on top and below slice of interest (default=3) 
    :param bool anisotropic: whether the image has different resolutions along different axes (default=True)
    :return: np.ndarray img_data_thick: image array containing the extracted slices
    """
    img_data_pairs = np.concatenate ( (pre_denoised_image[..., None], orig[..., None]), axis=3) # array of size [448, 448, 448, 2]
    return img_data_pairs

def get_thick_slices_ms(img_data, max_size, orig_size, slice_thickness=3):
    """
    Function to extract thick slices from the image 
    (feed slice_thickness preceeding and suceeding slices to network, 
    denoise only middle one) Added a padding stage so all the images are the 
    :param np.ndarray img_data: 3D MRI image read in with nibabel 
    :param int slice_thickness: number of slices to stack on top and below slice of interest (default=3) 
    :return: np.ndarray img_data_thick: image array containing the extracted slices
    """
    h, w, d = img_data.shape
    img_data_pad = np.expand_dims(np.pad(img_data, ((0, 0), (0, 0), (slice_thickness, slice_thickness)), mode='edge'),
                                  axis=3)
    img_data_thick = np.ndarray((h, w, d, 0), dtype=np.uint8)
    
    for slice_idx in range(2 * slice_thickness + 1):
        img_data_thick = np.append(img_data_thick, img_data_pad[:, :, slice_idx:d + slice_idx, :], axis=3)
        
    dif = max_size - orig_size
    to_pad = ((0,dif), (0,dif), (0,dif), (0,0))
    img_data_thick = np.pad(img_data_thick, pad_width=to_pad, mode='constant', constant_values = 0)
    return img_data_thick

def get_thick_slices_vmap(img_data, map_data, max_size, orig_size, slice_thickness=3):
    """
    Function to extract thick slices from the image 
    (feed slice_thickness preceeding and suceeding slices to network, 
    denoise only middle one) Added a padding stage so all the images are the 
    :param np.ndarray img_data: 3D MRI image read in with nibabel 
    :param int slice_thickness: number of slices to stack on top and below slice of interest (default=3) 
    :return: np.ndarray img_data_thick: image array containing the extracted slices
    """
    h, w, d = img_data.shape
    
    img_data_pad = np.expand_dims(np.pad(img_data, ((0, 0), (0, 0), (slice_thickness, slice_thickness)), mode='edge'),axis=3)
    map_data_pad = np.expand_dims(np.pad(map_data, ((0, 0), (0, 0), (slice_thickness, slice_thickness)), mode='edge'), axis=3)
    
    img_data_thick = np.ndarray((h, w, d, 0), dtype=np.uint8)
    map_data_thick = np.ndarray((h, w, d, 0), dtype=np.float64)
    
    for slice_idx in range(2 * slice_thickness + 1):
        img_data_thick = np.append(img_data_thick, img_data_pad[:, :, slice_idx:d + slice_idx, :], axis=3)
        map_data_thick = np.append(map_data_thick, map_data_pad[:, :, slice_idx:d + slice_idx, :], axis=3)
    return img_data_thick, map_data_thick 

def get_thick_slices_maponly(map_data, slice_thickness=3):
    """
    Function to extract thick slices from the image 
    (feed slice_thickness preceeding and suceeding slices to network, 
    denoise only middle one) Added a padding stage so all the images are the 
    :param np.ndarray img_data: 3D MRI image read in with nibabel 
    :param int slice_thickness: number of slices to stack on top and below slice of interest (default=3) 
    :return: np.ndarray img_data_thick: image array containing the extracted slices
    """
    h, w, d = map_data.shape
    
    map_data_pad = np.expand_dims(np.pad(map_data, ((0, 0), (0, 0), (slice_thickness, slice_thickness)), mode='edge'), axis=3)
    
    map_data_thick = np.ndarray((h, w, d, 0), dtype=np.float64)
    
    for slice_idx in range(2 * slice_thickness + 1):
        map_data_thick = np.append(map_data_thick, map_data_pad[:, :, slice_idx:d + slice_idx, :], axis=3)
    return map_data_thick 

def filter_blank_slices_thick(img_vol, label_vol, weight_vol, threshold=50):
    """
    Function to filter blank slices from the volume using the label volume
    :param np.ndarray img_vol: orig image volume
    :param np.ndarray label_vol: label images (ground truth)
    :param np.ndarray weight_vol: weight corresponding to labels
    :param int threshold: threshold for number of pixels needed to keep slice (below = dropped)
    :return:
    :return: np.ndarray img_vol: filtered orig image volume
    :return: np.ndarray label_vol: filtered label images (ground truth)
    :return: np.ndarray weight_vol: filtered weight corresponding to labels
    """
    # Get indices of all slices with more than threshold labels/pixels
    select_slices = (np.sum(label_vol, axis=(0, 1)) > threshold)

    # Retain only slices with more than threshold labels/pixels
    img_vol = img_vol[:, :, select_slices, :]
    label_vol = label_vol[:, :, select_slices]
    weight_vol = weight_vol[:, :, select_slices]

    return img_vol, label_vol, weight_vol

def filter_blank_slices_thick_ms(img_vol, threshold=50):
    """
    Function to filter blank slices from the volume using the label volume
    :param np.ndarray img_vol: orig image volume
    :param np.ndarray label_vol: label images (ground truth)
    :param np.ndarray weight_vol: weight corresponding to labels
    :param int threshold: threshold for number of pixels needed to keep slice (below = dropped)
    :return:
    :return: np.ndarray img_vol: filtered orig image volume
    :return: np.ndarray label_vol: filtered label images (ground truth)
    :return: np.ndarray weight_vol: filtered weight corresponding to labels
    """
    # Get indices of all slices with more than threshold labels/pixels
    select_slices = (np.sum(img_vol, axis=(0, 1)) > threshold)

    # Retain only slices with more than threshold labels/pixels
    img_vol = img_vol[:, :, select_slices[:,1], :]

    return img_vol

def filter_blank_slices_thick_vmap(img_vol, threshold=50):
    """
    Function to filter blank slices from the volume using the label volume
    :param np.ndarray img_vol: orig image volume
    :param np.ndarray label_vol: label images (ground truth)
    :param np.ndarray weight_vol: weight corresponding to labels
    :param int threshold: threshold for number of pixels needed to keep slice (below = dropped)
    :return:
    :return: np.ndarray img_vol: filtered orig image volume
    :return: np.ndarray label_vol: filtered label images (ground truth)
    :return: np.ndarray weight_vol: filtered weight corresponding to labels
    """
    # Get indices of all slices with more than threshold labels/pixels
    select_slices = (np.sum(img_vol, axis=(0, 1)) > threshold)

    # Retain only slices with more than threshold labels/pixels
    img_vol = img_vol[:, :, select_slices[:,1], :]
    
    return img_vol, select_slices


# weight map generator
def create_weight_mask(mapped_aseg, max_weight=5, max_edge_weight=5):
    """
    Function to create weighted mask - with median frequency balancing and edge-weighting
    :param np.ndarray mapped_aseg: label space segmentation
    :param int max_weight: an upper bound on weight values
    :param int max_edge_weight: edge-weighting factor
    :return: np.ndarray weights_mask: generated weights mask
    """
    unique, counts = np.unique(mapped_aseg, return_counts=True)

    # Median Frequency Balancing
    class_wise_weights = np.median(counts) / counts
    class_wise_weights[class_wise_weights > max_weight] = max_weight
    (h, w, d) = mapped_aseg.shape

    weights_mask = np.reshape(class_wise_weights[mapped_aseg.ravel()], (h, w, d))

    # Gradient Weighting
    (gx, gy, gz) = np.gradient(mapped_aseg)
    grad_weight = max_edge_weight * np.asarray(np.power(np.power(gx, 2) + np.power(gy, 2) + np.power(gz, 2), 0.5) > 0,
                                               dtype='float')

    weights_mask += grad_weight

    return weights_mask


# class unknown filler (cortex)
def fill_unknown_labels_per_hemi(gt, unknown_label, cortex_stop):
    """
    Function to replace label 1000 (lh unknown) and 2000 (rh unknown) with closest class for each voxel.
    :param np.ndarray gt: ground truth segmentation with class unknown
    :param int unknown_label: class label for unknown (lh: 1000, rh: 2000)
    :param int cortex_stop: class label at which cortical labels of this hemi stop (lh: 2000, rh: 3000)
    :return: np.ndarray gt: ground truth segmentation with replaced unknown class labels
    """
    # Define shape of image and dilation element
    h, w, d = gt.shape
    struct1 = ndimage.generate_binary_structure(3, 2)

    # Get indices of unknown labels, dilate them to get closest sorrounding parcels
    unknown = gt == unknown_label
    unknown = (morphology.binary_dilation(unknown, struct1) ^ unknown)
    list_parcels = np.unique(gt[unknown])

    # Mask all subcortical structures (fill unknown with closest cortical parcels only)
    mask = (list_parcels > unknown_label) & (list_parcels < cortex_stop)
    list_parcels = list_parcels[mask]

    # For each closest parcel, blur label with gaussian filter (spread), append resulting blurred images
    blur_vals = np.ndarray((h, w, d, 0), dtype=float)

    for idx in range(len(list_parcels)):
        aseg_blur = filters.gaussian_filter(1000 * np.asarray(gt == list_parcels[idx], dtype=float), sigma=5)
        blur_vals = np.append(blur_vals, np.expand_dims(aseg_blur, axis=3), axis=3)

    # Get for each position parcel with maximum value after blurring (= closest parcel)
    unknown = np.argmax(blur_vals, axis=3)
    unknown = np.reshape(list_parcels[unknown.ravel()], (h, w, d))

    # Assign the determined closest parcel to the unknown class (case-by-case basis)
    mask = gt == unknown_label
    gt[mask] = unknown[mask]

    return gt


# Label mapping functions (to aparc (eval) and to label (train))
def map_label2aparc_aseg(mapped_aseg):
    """
    Function to perform look-up table mapping from label space to aparc.DKTatlas+aseg space
    :param np.ndarray mapped_aseg: label space segmentation
    :return: np.ndarray aseg: segmentation in aparc+aseg space
    """
    aseg = np.zeros_like(mapped_aseg)
    labels = np.array([0, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14,
                       15, 16, 17, 18, 24, 26, 28, 31, 41, 43, 44,
                       46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63,
                       77, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011,
                       1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022,
                       1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035,
                       2002, 2005, 2010, 2012, 2013, 2014, 2016, 2017, 2021, 2022, 2023,
                       2024, 2025, 2028])
    h, w, d = aseg.shape

    aseg = labels[mapped_aseg.ravel()]

    aseg = aseg.reshape((h, w, d))

    return aseg


def map_aparc_aseg2label(aseg, aseg_nocc=None):
    """
    Function to perform look-up table mapping of aparc.DKTatlas+aseg.mgz data to label space
    :param np.ndarray aseg: ground truth aparc+aseg
    :param None/np.ndarray aseg_nocc: ground truth aseg without corpus callosum segmentation
    :return: np.ndarray mapped_aseg: label space segmentation (coronal and axial)
    :return: np.ndarray mapped_aseg_sag: label space segmentation (sagittal)
    """
    aseg_temp = aseg.copy()
    aseg[aseg == 80] = 77  # Hypointensities Class
    aseg[aseg == 85] = 0  # Optic Chiasma to BKG
    aseg[aseg == 62] = 41  # Right Vessel to Right GM
    aseg[aseg == 30] = 2  # Left Vessel to Left GM
    aseg[aseg == 72] = 24  # 5th Ventricle to CSF

    # If corpus callosum is not removed yet, do it now
    if aseg_nocc is not None:
        cc_mask = (aseg >= 251) & (aseg <= 255)
        aseg[cc_mask] = aseg_nocc[cc_mask]

    aseg[aseg == 3] = 0  # Map Remaining Cortical labels to background
    aseg[aseg == 42] = 0

    # If ctx-unknowns are not filled yet, do it now
    if np.any(np.in1d([1000, 2000], aseg.ravel())):
        aseg = fill_unknown_labels_per_hemi(aseg, 1000, 2000)
        aseg = fill_unknown_labels_per_hemi(aseg, 2000, 3000)

    cortical_label_mask = (aseg >= 2000) & (aseg <= 2999)
    aseg[cortical_label_mask] = aseg[cortical_label_mask] - 1000

    # Preserve Cortical Labels
    aseg[aseg_temp == 2014] = 2014
    aseg[aseg_temp == 2028] = 2028
    aseg[aseg_temp == 2012] = 2012
    aseg[aseg_temp == 2016] = 2016
    aseg[aseg_temp == 2002] = 2002
    aseg[aseg_temp == 2023] = 2023
    aseg[aseg_temp == 2017] = 2017
    aseg[aseg_temp == 2024] = 2024
    aseg[aseg_temp == 2010] = 2010
    aseg[aseg_temp == 2013] = 2013
    aseg[aseg_temp == 2025] = 2025
    aseg[aseg_temp == 2022] = 2022
    aseg[aseg_temp == 2021] = 2021
    aseg[aseg_temp == 2005] = 2005

    labels = np.array([0, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14,
                       15, 16, 17, 18, 24, 26, 28, 31, 41, 43, 44,
                       46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63,
                       77, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011,
                       1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022,
                       1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035,
                       2002, 2005, 2010, 2012, 2013, 2014, 2016, 2017, 2021, 2022, 2023,
                       2024, 2025, 2028])

    h, w, d = aseg.shape
    lut_aseg = np.zeros(max(labels) + 1, dtype='int')
    for idx, value in enumerate(labels):
        lut_aseg[value] = idx

    # Remap Label Classes - Perform LUT Mapping - Coronal, Axial

    mapped_aseg = lut_aseg.ravel()[aseg.ravel()]

    mapped_aseg = mapped_aseg.reshape((h, w, d))

    # Map Sagittal Labels
    aseg[aseg == 2] = 41
    aseg[aseg == 3] = 42
    aseg[aseg == 4] = 43
    aseg[aseg == 5] = 44
    aseg[aseg == 7] = 46
    aseg[aseg == 8] = 47
    aseg[aseg == 10] = 49
    aseg[aseg == 11] = 50
    aseg[aseg == 12] = 51
    aseg[aseg == 13] = 52
    aseg[aseg == 17] = 53
    aseg[aseg == 18] = 54
    aseg[aseg == 26] = 58
    aseg[aseg == 28] = 60
    aseg[aseg == 31] = 63

    cortical_label_mask = (aseg >= 2000) & (aseg <= 2999)
    aseg[cortical_label_mask] = aseg[cortical_label_mask] - 1000

    labels_sag = np.array([0, 14, 15, 16, 24, 41, 43, 44, 46, 47, 49,
                           50, 51, 52, 53, 54, 58, 60, 63, 77, 1002,
                           1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014,
                           1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025,
                           1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035])

    h, w, d = aseg.shape
    lut_aseg = np.zeros(max(labels_sag) + 1, dtype='int')
    for idx, value in enumerate(labels_sag):
        lut_aseg[value] = idx

    # Remap Label Classes - Perform LUT Mapping - Sagittal

    mapped_aseg_sag = lut_aseg.ravel()[aseg.ravel()]

    mapped_aseg_sag = mapped_aseg_sag.reshape((h, w, d))

    return mapped_aseg, mapped_aseg_sag


def sagittal_coronal_remap_lookup(x):
    """
    Dictionary mapping to convert left labels to corresponding right labels for aseg
    :param int x: label to look up
    :return: dict: left-to-right aseg label mapping dict
    """
    return {
        2: 41,
        3: 42,
        4: 43,
        5: 44,
        7: 46,
        8: 47,
        10: 49,
        11: 50,
        12: 51,
        13: 52,
        17: 53,
        18: 54,
        26: 58,
        28: 60,
        31: 63,
        }[x]


def map_prediction_sagittal2full(prediction_sag, num_classes=79):
    """
    Function to remap the prediction on the sagittal network to full label space used by coronal and axial networks
    (full aparc.DKTatlas+aseg.mgz)
    :param np.ndarray prediction_sag: sagittal prediction (labels)
    :param int num_classes: number of classes (96 for full classes, 79 for hemi split)
    :return: np.ndarray prediction_full: Remapped prediction
    """
    if num_classes == 96:
        idx_list = np.asarray([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1, 2, 3, 14, 15, 4, 16,
                               17, 18, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                               20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 20, 21, 22,
                               23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                               40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50], dtype=np.int16)

    else:
        idx_list = np.asarray([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1, 2, 3, 14, 15, 4, 16,
                               17, 18, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                               20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 20, 22, 27,
                               29, 30, 31, 33, 34, 38, 39, 40, 41, 42, 45], dtype=np.int16)

    prediction_full = prediction_sag[:, idx_list, :, :]
    return prediction_full


# Clean up and class separation
def bbox_3d(img):
    """
    Function to extract the three-dimensional bounding box coordinates.
    :param np.ndarray img: mri image
    :return: float rmin
    :return: float rmax
    :return: float cmin
    :return: float cmax
    :return: float zmin
    :return: float zmax
    """

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax


def get_largest_cc(segmentation):
    """
    Function to find largest connected component of segmentation.
    :param np.ndarray segmentation: segmentation
    :return: np.ndarray largest_cc: largest connected component of the segmentation array
    """
    labels = label(segmentation, connectivity=3, background=0)

    bincount = np.bincount(labels.flat)
    background = np.argmax(bincount)
    bincount[background] = -1

    largest_cc = labels == np.argmax(bincount)

    return largest_cc

class OrigDataThickPatches(Dataset):
    def __init__(self, img_filename, volume, max_patch_size=(96,96,96), stride=(32,32,32), transforms=None):
        """
        Initializes the dataset by extracting cubic patches from a 3D volume.
        
        Args:
            img_filename (str): Path to the image file.
            volume (np.ndarray): 3D volume array of shape (D, H, W).
            max_patch_size (int): Size of the cubic patch (default 96).
            stride (int): Stride for sliding-window patch extraction (default 32).
            transforms (callable, optional): Optional transform to apply.
        """
        self.img_filename = img_filename
        self.transforms = transforms
        self.patch_size = max_patch_size
        self.stride = stride
        
        # Save original shape.
        self.orig_shape = volume.shape  # (D, H, W)
        D, H, W = volume.shape
        
        # Pad the volume so that each dimension is at least max_patch_size and
        # so that the last patch extracted will have full size.
        pad_d = (max_patch_size[0] - (D % max_patch_size[0])) % max_patch_size[0]
        pad_h = (max_patch_size[1] - (H % max_patch_size[1])) % max_patch_size[1]
        pad_w = (max_patch_size[2] - (W % max_patch_size[2])) % max_patch_size[2]
        self.pad_width = ((0, pad_d), (0, pad_h), (0, pad_w))
        # volume_padded = np.pad(volume, self.pad_width, mode='constant', constant_values=0)
        volume_padded = np.pad(volume, self.pad_width, mode='reflect')
        self.volume = volume_padded
        self.D, self.H, self.W = volume_padded.shape
        
        # Compute starting indices so that patches always have size max_patch_size.
        self.d_starts = list(range(0, self.D - max_patch_size[0] + 1, stride[0]))
        self.h_starts = list(range(0, self.H - max_patch_size[1] + 1, stride[1]))
        self.w_starts = list(range(0, self.W - max_patch_size[2] + 1, stride[2]))
        
        # Ensure the last patch covers the border.
        if self.d_starts[-1] + max_patch_size[0] < self.D:
            self.d_starts.append(self.D - max_patch_size[0])
        if self.h_starts[-1] + max_patch_size[1] < self.H:
            self.h_starts.append(self.H - max_patch_size[1])
        if self.w_starts[-1] + max_patch_size[2] < self.W:
            self.w_starts.append(self.W - max_patch_size[2])
        
        # Create list of patch indices (tuples of starting indices).
        self.patch_indices = []
        for d in self.d_starts:
            for h in self.h_starts:
                for w in self.w_starts:
                    self.patch_indices.append((d, h, w))
                    
        print(f"Loaded {len(self.patch_indices)} patches from volume {img_filename}")
        
    def __len__(self):
        return len(self.patch_indices)
    
    def __getitem__(self, idx):
        d, h, w = self.patch_indices[idx]
        # Extract the cubic patch.
        patch = self.volume[d:d+self.patch_size[0], h:h+self.patch_size[1], w:w+self.patch_size[2]]
        # Add a channel dimension so that the patch has shape [1, patch_size, patch_size, patch_size].
        patch = np.expand_dims(patch, axis=0)
        if self.transforms:
            patch = self.transforms(patch)
        else:
            patch = torch.from_numpy(patch).float()
        # Return the patch and its starting indices.
        return {'patch': patch, 'indices': (d, h, w)}
    

class OrigDataThickSlices(Dataset):
    def __init__(self, img_filename, orig, plane='First', slice_thickness=0, transforms=None):
        """
        Initializes the dataset by processing the original 3D data based on the specified plane.
        
        Args:
            img_filename (str): Path to the image file.
            orig (np.ndarray): Original 3D data array.
            plane (str): Plane along which to slice ("First", "Second", "Third").
            slice_thickness (int): Thickness of the slices.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        try:
            self.img_filename = img_filename
            self.plane = plane
            self.slice_thickness = slice_thickness
            self.transforms = transforms

            # Normalize the data
            # orig = orig.astype(np.float32)
            # orig = (orig - orig.min()) / (orig.max() - orig.min())
            # orig = (orig*2.0)-1.0
            # Apply plane-specific transformations
            if self.plane == 'First':
                # No transformation needed if slicing along the first dimension
                self.images = orig  # Shape: (h_out, w_in, z_in)
                print('Loading First Plane (No transformation)')
            elif self.plane == 'Second':
                # For the second plane, transpose to bring the second axis first
                self.images = np.transpose(orig, (1, 0, 2))  # Shape: (w_out, h_in, z_in)
                print('Loading Second Plane (Transposed)')
            elif self.plane == 'Third':
                # For the third plane, transpose to bring the third axis first
                self.images = np.transpose(orig, (2, 0, 1))  # Shape: (z_out, h_in, w_in)
                print('Loading Third Plane (Transposed)')
            else:
                raise ValueError(f"Invalid plane specified: {self.plane}")

            # Add channel dimension
            self.images = np.expand_dims(self.images, axis=1)  # Shape: (num_slices, 1, H, W)
            self.count = self.images.shape[0]

            print(f"Successfully loaded Image from {img_filename} with {self.count} slices.")
        except Exception as e:
            print(f"Loading failed. {e}")

    def apply_thick_slicing(self, images, thickness):
        """
        Applies thick slicing by averaging over a specified number of adjacent slices.
        
        Args:
            images (np.ndarray): Array of images with shape (num_slices, H, W).
            thickness (int): Number of adjacent slices to average.
        
        Returns:
            np.ndarray: Thick-sliced images.
        """
        if thickness <= 1:
            return images
        
        num_slices, H, W = images.shape
        new_num_slices = num_slices - thickness + 1
        thick_images = np.zeros((new_num_slices, H, W), dtype=np.float32)
        
        for i in range(new_num_slices):
            thick_images[i] = np.mean(images[i:i+thickness], axis=0)
        
        return thick_images

    def __getitem__(self, index):
        """
        Retrieves the image at the specified index, applies transformations, and ensures correct shape.
        
        Args:
            index (int): Index of the image to retrieve.
        
        Returns:
            dict: Dictionary containing the image tensor.
        """
        img = self.images[index]  # Shape: (1, H, W)
        # img = img/(img.max())
        # img = (img * 2.0) - 1.0
        if self.transforms:
            img = self.transforms(img)
        else:
            # Convert to torch tensor if no transforms are provided
            img = torch.from_numpy(img)
        
        return {'image': img}

    def __len__(self):
        """
        Returns the total number of slices in the dataset.
        
        Returns:
            int: Number of slices.
        """
        return self.count
    
class OrigDataThickSlices2(Dataset):
    def __init__(self, img_filename, orig, plane='Axial', slice_thickness=3, transforms=None):
        try:
            self.img_filename = img_filename
            self.plane = plane
            self.slice_thickness = slice_thickness
            orig = (orig-orig.min())/(orig.max() - orig.min())
            orig = orig.astype(np.float32)
            orig = np.transpose(orig, (0, 2, 1))
            orig = np.flip(orig, axis=0)
            orig = np.flip(orig, axis=1)
            if plane == 'Sagittal':
                orig = transform_sagittal(orig)
                print('Loading Sagittal')
            elif plane == 'Axial':
                orig = transform_axial(orig)
                print('Loading Axial')
            else:
                print('Loading Coronal.')
            orig_thick = get_thick_slices(orig, self.slice_thickness)
            orig_thick2 = np.transpose(orig_thick, (0, 2, 1, 3))
            
            self.images = orig_thick2
            self.count = self.images.shape[0]
            self.transforms = transforms
            print("Successfully loaded Image from {}".format(img_filename))
        except Exception as e:
            print("Loading failed. {}".format(e))
    def __getitem__(self, index):
        img = self.images[index]
        if self.transforms is not None:
            img = self.transforms(img)
        return {'image': img}
    def __len__(self):
        return self.count
def transform_axial_sitk(vol, sagittal2axial=True):
    if sagittal2axial:
        vol = np.moveaxis(vol, [0, 1, 2], [1, 2, 0])
        return vol
    else:
        vol = np.moveaxis(vol, [1, 2, 0], [0, 1, 2])
        return vol
    
def transform_coronal_sitk(vol, sagittal2coronal=True):
    if sagittal2coronal:
        return np.moveaxis(vol, [0, 1, 2], [2, 0, 1])
    else:
        return np.moveaxis(vol, [1, 0, 2], [0, 1, 2])
    
class OrigDataThickSlicesSitk(Dataset):
    def __init__(self, img_filename, orig, input_channels = 7, plane='Axial', slice_thickness=3, transforms=None):
        try:
            #At this point, we have a Sagittal, Coronal, Axial image.
            self.img_filename = img_filename
            self.plane = plane
            self.slice_thickness = slice_thickness
            orig.astype(np.float32)
            orig = (orig-orig.min())/(orig.max() - orig.min())
            if plane == 'Sagittal':
                print('Loading Sagittal')
            elif plane == 'Axial':
                #Transforms to Axial, Sagittal, Coronal
                orig = transform_axial_sitk(orig)
                print('Loading Axial')
            else:
                #Transforms to Coronal, Axial, Sagittal
                orig = transform_coronal_sitk(orig)
                print('Loading Coronal.')
            slice_thickness = (input_channels - 1)/2
            orig_thick = get_thick_slices(orig, slice_thickness = self.slice_thickness)
            self.images = orig_thick
            self.count = self.images.shape[0]
            self.transforms = transforms
            print("Successfully loaded Image from {}".format(img_filename))
        except Exception as e:
            print("Loading failed. {}".format(e))
    def __getitem__(self, index):
        img = self.images[index]
        if self.transforms is not None:
            img = self.transforms(img)
        return {'image': img}
    def __len__(self):
        return self.count

# class OrigDataTwoSlices(Dataset):
#     """
#     Class to load a given image and segmentation and prepare it
#     for network training.
#     """
#     def __init__(self, pre_denoised_image, orig, plane='Axial', transforms=None, anisotropic = True):

#         try:
#             self.plane = plane
#             self.anisotropic = anisotropic
#             orig.astype(np.float64)
#             # orig = (orig-orig.min())/(orig.max() - orig.min())
#             # pre_denoised_image.astype(np.float64)
#             # pre_denoised_image = (pre_denoised_image - pre_denoised_image.min()) / (pre_denoised_image.max() - pre_denoised_image.min())
#             # Transform Data as needed
#             if plane == 'Sagittal':
#                 orig = transform_sagittal(orig)
#                 pre_denoised_image = transform_sagittal(pre_denoised_image)
#                 print('Loading Sagittal')

#             elif plane == 'Axial':
#                 orig = transform_axial(orig)
#                 pre_denoised_image = transform_axial(pre_denoised_image)
#                 print('Loading Axial')

#             else:
#                 print('Loading Coronal.')

#             # Create Thick Slices
#             orig_pairs = get_noisy_pre_den_pairs(pre_denoised_image, orig)
            
#             # Make 4D
#             orig_pairs = np.transpose(orig_pairs, (2, 0, 1, 3))
#             self.images = orig_pairs

#             self.count = self.images.shape[0]

#             self.transforms = transforms

#             print("Successfully loaded pre-denoised + original image pairs ")

#         except Exception as e:
#             print("Loading failed. {}".format(e))

#     def __getitem__(self, index):

#         img = self.images[index]

#         if self.transforms is not None:
#             img = self.transforms(img)

#         return {'image': img}

#     def __len__(self):
#         return self.count


##
# Dataset loading (for training)
##

# Operator to load hdf5-file for training
class AsegDatasetWithAugmentation(Dataset):
    """
    Class for loading aseg file with augmentations (transforms)
    """
    def __init__(self, params, map_256, map_320, map_448, transforms=None, is_val = False):

        # Load the h5 file and save it to the dataset
        try:
            self.params = params

            # Open file in reading mode
            with h5py.File(self.params['dataset_name'], "r") as hf:
                self.images = np.array(hf.get('orig_dataset')[:])
                self.orig_zooms = np.array(hf.get('orig_zooms')[:])
                self.field_localizer = np.array(hf.get('field_localizer')[:])
                self.subjects = np.array(hf.get("subject")[:])
                self.noise_std = params['noise_std']
                self.patch_size = params['patch_size']
                self.map_256 = map_256
                self.map_320 = map_320
                self.map_448 = map_448
                
                self.is_val = is_val

            self.count = self.images.shape[0]
            self.transforms = transforms

            print("Successfully loaded {} with plane: {}".format(params["dataset_name"], params["plane"]))

        except Exception as e:
            print("Loading failed: {}".format(e))

    def get_subject_names(self):
        return self.subjects
    
    def _get_patch(self, img, localizer):
        map_size = max(img.shape)
        if map_size == 256:
            vmap = self.map_256
        elif map_size ==320:
            vmap = self.map_320
        elif map_size == 448:
            vmap = self.map_448
            
        vmap = vmap[int(localizer)]
        
        img = np.moveaxis(img,(0,1,2),(2,1,0))
        maxi = img.max()
        mini = img.min()
        img = (img - mini) / (maxi-mini)
        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=1)
        hr = img[3]
        sigma = random.choice(self.noise_std)
        noise = 'R' + str(sigma)
        # LR_size = self.patch_size
        #NOISE ADDING
        lr = common.add_noise(img, vmap, noise)
        
        #AUGMENTATION
        # randcrop=trf.RandomCrop(LR_size)
        randflip=trf.RandomHorizontalFlip()
        if self.is_val == True:
            randrot=trf.RandomRotation([0,0])
        else:
            randrot=trf.RandomRotation([-5,5])
        lrhr = self.concatenate_Thick_GT(lr, hr)
        # lrhr = randcrop(lrhr)
        lrhr = randflip(lrhr)
        lrhr = randrot(lrhr)
        lr, hr = self.unconcatenate_Thick_GT(lrhr)
        
        return lr, hr, str(sigma)
    
    def __getitem__(self, index):

        img = self.images[index]
        zoom = self.orig_zooms[index]
        localizer = self.field_localizer[index]
        if self.transforms is not None:
            tx_sample = self.transforms({'img': img})
            img = tx_sample['img']
            #NEW PATCH ADDITION
            
        lr, hr, sigma = self._get_patch(img, localizer)
            
                
            # img, label = common.np2Tensor([lr, hr], self.rgb_range)

        return {'LR': lr, 'HR': hr, 'zoom':zoom, 'sigma':sigma}

    def __len__(self):
        return self.count
    
    

    def concatenate_Thick_GT(self, lr, hr):
        #Here, HR is ok
        hr = torch.from_numpy(hr)
        hr2 = hr.float()
        #hr converts to torch correctly
        hr2 = torch.unsqueeze(hr,0)
        lr = torch.from_numpy(lr)
        # lr = torch.unsqueeze(lr,0)
        lrhr = torch.cat([hr2,lr],0) #first 7 arrays of dim 0 are lr, last is hr
        return lrhr
    
    def unconcatenate_Thick_GT(self, lrhr):
        lr = lrhr[1:8] #This means from index 0 to index 7, but not including index 7
        hr = lrhr[0]
        return lr, hr
    
class MyBatchSampler(Sampler):
    def __init__(self, a_indices, b_indices, c_indices, d_indices, e_indices, f_indices, g_indices,  batch_size): 
        self.a_indices = a_indices
        self.b_indices = b_indices
        self.c_indices = c_indices
        self.d_indices = d_indices
        self.e_indices = e_indices
        self.f_indices = f_indices
        self.g_indices = g_indices
        self.batch_size = batch_size
    
    def __iter__(self):
        random.shuffle(self.a_indices)
        random.shuffle(self.b_indices)
        random.shuffle(self.c_indices)
        random.shuffle(self.d_indices)
        random.shuffle(self.e_indices)
        random.shuffle(self.f_indices)
        random.shuffle(self.g_indices)
        a_batches = chunk(self.a_indices, self.batch_size)
        b_batches = chunk(self.b_indices, self.batch_size)
        c_batches = chunk(self.c_indices, self.batch_size)
        d_batches = chunk(self.d_indices, self.batch_size)
        e_batches = chunk(self.e_indices, self.batch_size)
        f_batches = chunk(self.f_indices, self.batch_size)
        g_batches = chunk(self.g_indices, self.batch_size)
        all_batches = list(a_batches + b_batches + c_batches + d_batches + e_batches + f_batches + g_batches)
        all_batches = [batch.tolist() for batch in all_batches]
        random.shuffle(all_batches)
        return iter(all_batches)
    
    def __len__(self):
        return (len(self.a_indices) + len(self.b_indices) + len(self.c_indices) + len(self.d_indices) + len(self.e_indices) + len(self.f_indices) + len(self.g_indices)) // self.batch_size

def load_and_rescale_image_sitk_v2(img_filename, options, interpol=3, logger=None, is_eval=True,
                                 intensity_rescaling=True,
                                 uf_h=2.0, uf_w=2.0, uf_z=2.0, use_scipy=True):
    import math
    def round_v8(value, n=32):
        return math.ceil(value / n) * n

    def prepare_zooms_tensors(zooms, batch_size):
        zooms = torch.tensor(zooms)
        zooms = zooms.unsqueeze(0).repeat(batch_size, 1)
        return zooms
    """


    Parameters
    ----------
    img_filename : TYPE
        DESCRIPTION.
    interpol : TYPE, optional
        DESCRIPTION. The default is 3.
    logger : TYPE, optional
        DESCRIPTION. The default is None.
    is_eval : TYPE, optional
        DESCRIPTION. The default is True.
    conform_type : TYPE, optional
        DESCRIPTION. The default is 2.
    intensity_rescaling : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    batch_size = options.batch_size
    orig = sitk.ReadImage(img_filename)
    zoom = orig.GetSpacing()
    ishape = orig.GetSize()
    s1, s2, s3 = ishape[0:3]
    z1, z2, z3 = zoom[0:3]
    s1_target = s1 * uf_h
    s2_target = s2 * uf_w
    s3_target = s3 * uf_z
    s1_target_fix = round_v8(s1_target)
    s2_target_fix = round_v8(s2_target)
    s3_target_fix = round_v8(s3_target)
    uf_h_fix = s1_target_fix / s1
    uf_w_fix = s2_target_fix / s2
    uf_z_fix = s3_target_fix / s3
    orig_img = sitk.GetArrayFromImage(orig)
    max_orig = orig_img.max()
    min_orig = orig_img.min()
    orig_img = np.transpose(orig_img, (2, 1, 0))
    source_zooms = [z1, z2, z3]
    target_zooms = [z1 / uf_h_fix, z2 / uf_w_fix, z3 / uf_z_fix]

    source_zooms = prepare_zooms_tensors(source_zooms, batch_size)
    target_zooms = prepare_zooms_tensors(target_zooms, batch_size)
    # Never do intensity rescaling of interp_sitk since this will be used for header info and intensity histogram matching
    interp_sitk = conform_itk(orig, volshow, order=interpol, intensity_rescaling=False,
                              uf_h=uf_h, uf_w=uf_w, uf_z=uf_z, use_scipy=use_scipy)
    interp_sitk_v8 = conform_itk(orig, volshow, order=interpol, intensity_rescaling=intensity_rescaling,
                                 uf_h=uf_h_fix, uf_w=uf_w_fix, uf_z=uf_z_fix, use_scipy=use_scipy)
    interp_v8 = sitk.GetArrayFromImage(interp_sitk_v8)
    interp_sitk_data = sitk.GetArrayFromImage(interp_sitk)
    interp_sitk_data = np.transpose(interp_sitk_data, (2, 1, 0))
    if is_eval:
        return interp_v8, interp_sitk_v8, interp_sitk, max_orig, min_orig, source_zooms, target_zooms
    else:
        return orig, interp_sitk

def load_and_rescale_image_sitk_v3(img_filename, options, interpol=1, logger=None, is_eval=True,
                                 intensity_rescaling=True,
                                 uf_h=2.0, uf_w=2.0, uf_z=2.0, use_scipy=True):
    import math
    def round_v8(value, n=4):
        return math.ceil(value / n) * n

    def prepare_zooms_tensors(zooms, batch_size):
        zooms = torch.tensor(zooms)
        zooms = zooms.unsqueeze(0).repeat(batch_size, 1)
        return zooms
    """


    Parameters
    ----------
    img_filename : TYPE
        DESCRIPTION.
    interpol : TYPE, optional
        DESCRIPTION. The default is 3.
    logger : TYPE, optional
        DESCRIPTION. The default is None.
    is_eval : TYPE, optional
        DESCRIPTION. The default is True.
    conform_type : TYPE, optional
        DESCRIPTION. The default is 2.
    intensity_rescaling : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    batch_size = options.batch_size
    orig = sitk.ReadImage(img_filename)
    zoom = orig.GetSpacing()
    ishape = orig.GetSize()
    s1, s2, s3 = ishape[0:3]
    z1, z2, z3 = zoom[0:3]
    s1_target = s1 * uf_h
    s2_target = s2 * uf_w
    s3_target = s3 * uf_z
    s1_target_fix = round_v8(s1_target)
    s2_target_fix = round_v8(s2_target)
    s3_target_fix = round_v8(s3_target)
    uf_h_fix = s1_target_fix / s1
    uf_w_fix = s2_target_fix / s2
    uf_z_fix = s3_target_fix / s3
    orig_img = sitk.GetArrayFromImage(orig)
    max_orig_final = orig_img.max() # 232.42
    min_orig_final = orig_img.min() # 0.0
    orig_img = np.transpose(orig_img, (2, 1, 0))
    source_zooms = [z1, z2, z3]
    target_zooms = [z1 / uf_h_fix, z2 / uf_w_fix, z3 / uf_z_fix]

    source_zooms = prepare_zooms_tensors(source_zooms, batch_size)
    target_zooms = prepare_zooms_tensors(target_zooms, batch_size)
    # Never do intensity rescaling of interp_sitk since this will be used for header info and intensity histogram matching
    interp_sitk, min_orig1, max_orig1 = conform_itk2(orig, volshow, order=interpol, intensity_rescaling=False,
                              uf_h=uf_h, uf_w=uf_w, uf_z=uf_z, use_scipy=use_scipy)
    interp_sitk_v8, min_orig_final, max_orig_final = conform_itk2(orig, volshow, order=interpol, intensity_rescaling=intensity_rescaling,
                                 uf_h=uf_h_fix, uf_w=uf_w_fix, uf_z=uf_z_fix, use_scipy=use_scipy)
    interp_v8 = sitk.GetArrayFromImage(interp_sitk_v8)
    min_orig, max_orig = interp_v8.min(), interp_v8.max()
    interp_sitk_data = sitk.GetArrayFromImage(interp_sitk)
    interp_sitk_data = np.transpose(interp_sitk_data, (2, 1, 0))
    if is_eval:
        return interp_v8, interp_sitk_v8, interp_sitk, max_orig, min_orig, source_zooms, target_zooms, max_orig_final, min_orig_final
    else:
        return orig, interp_sitk
    
def sitk2sitk(source, target, order=3):
    new_size = target.GetSize()
    new_spacing = target.GetSpacing()
    # Using sitk resampler
    new_direction = target.GetDirection()
    new_origin = target.GetOrigin()
    output_pixel_type = target.GetPixelID()
    interpolator = sitk.sitkBSpline
    spline_order = order
    transform = sitk.BSplineTransform(3, spline_order)
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputDirection(new_direction)
    resampler.SetOutputOrigin(new_origin)
    resampler.SetOutputPixelType(output_pixel_type)
    resampler.SetInterpolator(interpolator)
    resampler.SetTransform(transform)
    mapped_data_itk = resampler.Execute(source)
    return mapped_data_itk