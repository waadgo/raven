
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
import optparse
import sys
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from typing import Optional, Type, Tuple, Union, Iterable, cast
from scipy.ndimage import zoom
HELPTEXT = """
Script to conform an MRI brain image to UCHAR, RAS orientation, and 1mm isotropic voxels


USAGE:
conform.py  -i <input> -o <output>


Dependencies:
    Python 3.5

    Numpy
    http://www.numpy.org

    Nibabel to read and write FreeSurfer data
    http://nipy.org/nibabel/


Original Author: Martin Reuter
Date: Jul-09-2019

"""

h_input = 'path to input image'
h_output = 'path to ouput image'
h_order = 'order of interpolation (0=nearest,1=linear(default),2=quadratic,3=cubic)'


def options_parse():
    """
    Command line option parser
    """
    parser = optparse.OptionParser(version='$Id: conform.py,v 1.0 2019/07/19 10:52:08 mreuter Exp $',
                                   usage=HELPTEXT)
    parser.add_option('--input', '-i', dest='input', help=h_input)
    parser.add_option('--output', '-o', dest='output', help=h_output)
    parser.add_option('--order', dest='order', help=h_order, type="int", default=1)
    (fin_options, args) = parser.parse_args()
    if fin_options.input is None or fin_options.output is None:
        sys.exit('ERROR: Please specify input and output images')
    return fin_options


# def map_image(img, out_affine, out_shape, ras2ras=np.array([[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
#               order=1):
#     """
#     Function to map image to new voxel space (RAS orientation)

#     :param nibabel.MGHImage img: the src 3D image with data and affine set
#     :param np.ndarray out_affine: trg image affine
#     :param np.ndarray out_shape: the trg shape information
#     :param np.ndarray ras2ras: ras2ras an additional maping that should be applied (default=id to just reslice)
#     :param int order: order of interpolation (0=nearest,1=linear(default),2=quadratic,3=cubic)
#     :return: mapped Image data array
#     """
#     from scipy.ndimage import affine_transform
#     from numpy.linalg import inv

#     # compute vox2vox from src to trg
#     vox2vox = inv(out_affine) @ ras2ras @ img.affine

#     # here we apply the inverse vox2vox (to pull back the src info to the target image)
#     new_data = affine_transform(img.get_fdata(), inv(vox2vox), output_shape=out_shape, order=order)
#     return new_data


def getscale(data, dst_min, dst_max, f_low=0.0, f_high=0.99999):
    """
    Function to get offset and scale of image intensities to robustly rescale to range dst_min..dst_max.
    Equivalent to how mri_convert conforms images.

    :param np.ndarray data: Image data (intensity values)
    :param float dst_min: future minimal intensity value
    :param float dst_max: future maximal intensity value
    :param f_low: robust cropping at low end (0.0 no cropping)
    :param f_high: robust cropping at higher end (0.999 crop one thousandths of high intensity voxels)
    :return: returns (adjusted) src_min and scale factor
    """
    # get min and max from source
    src_min = np.min(data)
    src_max = np.max(data)

    if src_min < 0.0:
        # sys.exit('ERROR: Min value in input is below 0.0!')
        data = (data - data.min()) / (data.max() - data.min())
        

    print("Input:    min: " + format(src_min) + "  max: " + format(src_max))

    if f_low == 0.0 and f_high == 1.0:
        return src_min, 1.0

    # compute non-zeros and total vox num
    nz = (np.abs(data) >= 1e-15).sum() #nonzeros 7372799
    voxnum = data.shape[0] * data.shape[1] * data.shape[2] #voxnum 7372800

    # compute histogram
    histosize = 1000
    bin_size = (src_max - src_min) / histosize
    hist, bin_edges = np.histogram(data, histosize)

    # compute cummulative sum
    cs = np.concatenate(([0], np.cumsum(hist)))

    # get lower limit
    nth = int(f_low * voxnum)
    idx = np.where(cs < nth)

    if len(idx[0]) > 0:
        idx = idx[0][-1] + 1

    else:
        idx = 0

    src_min = idx * bin_size + src_min

    # print("bin min: "+format(idx)+"  nth: "+format(nth)+"  passed: "+format(cs[idx])+"\n")
    # get upper limit
    nth = voxnum - int((1.0 - f_high) * nz)
    idx = np.where(cs >= nth)

    if len(idx[0]) > 0:
        idx = idx[0][0] - 2

    else:
        print('ERROR: rescale upper bound not found')

    src_max = idx * bin_size + src_min
    # print("bin max: "+format(idx)+"  nth: "+format(nth)+"  passed: "+format(voxnum-cs[idx])+"\n")

    # scale
    if src_min == src_max:
        scale = 1.0

    else:
        scale = (dst_max - dst_min) / (src_max - src_min)

    print("rescale:  min: " + format(src_min) + "  max: " + format(src_max) + "  scale: " + format(scale))

    return src_min, src_max, scale


def map_image(
        img: nib.analyze.SpatialImage,
        out_affine: np.ndarray,
        out_shape: tuple,
        ras2ras: Optional[np.ndarray] = None,
        order: int = 1,
        dtype: Optional[Type] = None
) -> np.ndarray:
    """Map image to new voxel space (RAS orientation).

    Parameters
    ----------
    img : nib.analyze.SpatialImage
        the src 3D image with data and affine set
    out_affine : np.ndarray
        trg image affine
    out_shape : tuple[int, ...], np.ndarray
        the trg shape information
    ras2ras : Optional[np.ndarray]
        an additional mapping that should be applied (default=id to just reslice)
    order : int
        order of interpolation (0=nearest,1=linear(default),2=quadratic,3=cubic)
    dtype : Optional[Type]
        target dtype of the resulting image (relevant for reorientation, default=same as img)

    Returns
    -------
    np.ndarray
        mapped image data array
    
    """
    from scipy.ndimage import affine_transform
    from numpy.linalg import inv

    if ras2ras is None:
        ras2ras = np.eye(4)

    # compute vox2vox from src to trg
    vox2vox = inv(out_affine) @ ras2ras @ img.affine

    # here we apply the inverse vox2vox (to pull back the src info to the target image)
    image_data = np.asanyarray(img.dataobj)
    # convert frames to single image

    out_shape = tuple(out_shape)
    # if input has frames
    if image_data.ndim > 3:
        # if the output has no frames
        if len(out_shape) == 3:
            if any(s != 1 for s in image_data.shape[3:]):
                raise ValueError(
                    f"Multiple input frames {tuple(image_data.shape)} not supported!"
                )
            image_data = np.squeeze(image_data, axis=tuple(range(3, image_data.ndim)))
        # if the output has the same number of frames as the input
        elif image_data.shape[3:] == out_shape[3:]:
            # add a frame dimension to vox2vox
            _vox2vox = np.eye(5, dtype=vox2vox.dtype)
            _vox2vox[:3, :3] = vox2vox[:3, :3]
            _vox2vox[3:, 4:] = vox2vox[:3, 3:]
            vox2vox = _vox2vox
        else:
            raise ValueError(
                    f"Input image and requested output shape have different frames:"
                    f"{image_data.shape} vs. {out_shape}!"
                )

    if dtype is not None:
        image_data = image_data.astype(dtype)

    return affine_transform(
        image_data, inv(vox2vox), output_shape=out_shape, order=order
    )


def getscale(
        data: np.ndarray,
        dst_min: float,
        dst_max: float,
        f_low: float = 0.00,
        f_high: float = 0.999
) -> Tuple[float, float]:
    """Get offset and scale of image intensities to robustly rescale to range dst_min..dst_max.

    Equivalent to how mri_convert conforms images.

    Parameters
    ----------
    data : np.ndarray
        image data (intensity values)
    dst_min : float
        future minimal intensity value
    dst_max : float
        future maximal intensity value
    f_low : float
        robust cropping at low end (0.0 no cropping, default)
    f_high : float
        robust cropping at higher end (0.9995 crop one thousandth of high intensity voxels, default)

    Returns
    -------
    float src_min
        (adjusted) offset
    float
        scale factor

    """
    # get min and max from source
    src_min = np.min(data)
    src_max = np.max(data)

    if src_min < 0.0:
        print("WARNING: Input image has value(s) below 0.0 !")

    print("Input:    min: " + format(src_min) + "  max: " + format(src_max))

    if f_low == 0.0 and f_high == 1.0:
        return src_min, 1.0

    # compute non-zeros and total vox num
    nz = (np.abs(data) >= 1e-15).sum()
    voxnum = data.shape[0] * data.shape[1] * data.shape[2]

    # compute histogram
    histosize = 1000
    bin_size = (src_max - src_min) / histosize
    hist, bin_edges = np.histogram(data, histosize)

    # compute cumulative sum
    cs = np.concatenate(([0], np.cumsum(hist)))

    # get lower limit
    nth = int(f_low * voxnum)
    idx = np.where(cs < nth)

    if len(idx[0]) > 0:
        idx = idx[0][-1] + 1

    else:
        idx = 0

    src_min = idx * bin_size + src_min

    # get upper limit
    nth = voxnum - int((1.0 - f_high) * nz)
    idx = np.where(cs >= nth)

    if len(idx[0]) > 0:
        idx = idx[0][0] - 2

    else:
        print("ERROR: rescale upper bound not found")

    src_max = idx * bin_size + src_min

    # scale
    if src_min == src_max:
        scale = 1.0

    else:
        scale = (dst_max - dst_min) / (src_max - src_min)

    print(
        "rescale:  min: "
        + format(src_min)
        + "  max: "
        + format(src_max)
        + "  scale: "
        + format(scale)
    )

    return src_min, scale


def scalecrop(
        data: np.ndarray,
        dst_min: float,
        dst_max: float,
        src_min: float,
        scale: float
) -> np.ndarray:
    """Crop the intensity ranges to specific min and max values.

    Parameters
    ----------
    data : np.ndarray
        Image data (intensity values)
    dst_min : float
        future minimal intensity value
    dst_max : float
        future maximal intensity value
    src_min : float
        minimal value to consider from source (crops below)
    scale : float
        scale value by which source will be shifted

    Returns
    -------
    np.ndarray
        scaled image data
    
    """
    data_new = dst_min + scale * (data - src_min)

    # clip
    data_new = np.clip(data_new, dst_min, dst_max)
    print(
        "Output:   min: " + format(data_new.min()) + "  max: " + format(data_new.max())
    )

    return data_new

def getscale2(
        data: np.ndarray,
        dst_min: float,
        dst_max: float,
        f_low: float = 0.00,
        f_high: float = 0.9995
) -> Tuple[float, float]:
    """Get offset and scale of image intensities to robustly rescale to range dst_min..dst_max.

    Equivalent to how mri_convert conforms images.

    Parameters
    ----------
    data : np.ndarray
        image data (intensity values)
    dst_min : float
        future minimal intensity value
    dst_max : float
        future maximal intensity value
    f_low : float
        robust cropping at low end (0.0 no cropping, default)
    f_high : float
        robust cropping at higher end (0.9995 crop one thousandth of high intensity voxels, default)

    Returns
    -------
    float src_min
        (adjusted) offset
    float
        scale factor

    """
    # get min and max from source
    src_min = np.min(data)
    src_max = np.max(data)

    if src_min < 0.0:
        print("WARNING: Input image has value(s) below 0.0 !")

    print("Input:    min: " + format(src_min) + "  max: " + format(src_max))

    if f_low == 0.0 and f_high == 1.0:
        return src_min, 1.0

    # compute non-zeros and total vox num
    nz = (np.abs(data) >= 1e-15).sum()
    voxnum = data.shape[0] * data.shape[1] * data.shape[2]

    # compute histogram
    histosize = 10000
    bin_size = (src_max - src_min) / histosize
    hist, bin_edges = np.histogram(data, histosize)

    # compute cumulative sum
    cs = np.concatenate(([0], np.cumsum(hist)))

    # get lower limit
    nth = int(f_low * voxnum)
    idx = np.where(cs < nth)

    if len(idx[0]) > 0:
        idx = idx[0][-1] + 1

    else:
        idx = 0

    src_min = idx * bin_size + src_min

    # get upper limit
    nth = voxnum - int((1.0 - f_high) * nz)
    idx = np.where(cs >= nth)

    if len(idx[0]) > 0:
        idx = idx[0][0] - 2

    else:
        print("ERROR: rescale upper bound not found")

    src_max = idx * bin_size + src_min

    # scale
    if src_min == src_max:
        scale = 1.0

    else:
        scale = (dst_max - dst_min) / (src_max - src_min)

    print(
        "rescale:  min: "
        + format(src_min)
        + "  max: "
        + format(src_max)
        + "  scale: "
        + format(scale)
    )

    return src_min, src_max, scale


def scalecrop2(
        data: np.ndarray,
        dst_min: float,
        dst_max: float,
        src_min: float,
        scale: float
) -> np.ndarray:
    """Crop the intensity ranges to specific min and max values.

    Parameters
    ----------
    data : np.ndarray
        Image data (intensity values)
    dst_min : float
        future minimal intensity value
    dst_max : float
        future maximal intensity value
    src_min : float
        minimal value to consider from source (crops below)
    scale : float
        scale value by which source will be shifted

    Returns
    -------
    np.ndarray
        scaled image data
    
    """
    data_new = dst_min + scale * (data - src_min)

    # clip
    data_new = np.clip(data_new, dst_min, dst_max)
    print(
        "Output:   min: " + format(data_new.min()) + "  max: " + format(data_new.max())
    )

    return data_new


def std_pos(img, order=1):
    """
    Function to reslice images to standard position.
    :param int order: interpolation order (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return:nibabel.MGHImage new_img: conformed image
    """
    from nibabel.freesurfer.mghformat import MGHHeader
    
    ishape = img.shape
    izoom = img.header.get_zooms()
    
    h = ishape[0]
    w = ishape[1]
    c = ishape[2]
    
    z1 = izoom[0]
    z2 = izoom[1]
    z3 = izoom[2]
    
    h1 = MGHHeader.from_header(img.header)  # may copy some parameters if input was MGH format
    h1.set_data_shape([h, w, c, 1])
    h1.set_zooms([z1, z2, z3])
    h1['Mdc'] = [[-1, 0, 0], [0, 0, -1], [0, 1, 0]]
    h1['fov'] = max(ishape)
    h1['Pxyz_c'] = img.affine.dot(np.hstack((np.array(img.shape[:3]) / 2.0, [1])))[:3]

    # from_header does not compute Pxyz_c (and probably others) when importing from nii
    # Pxyz is the center of the image in world coords

    mapped_data = map_image(img, h1.get_affine(), h1.get_data_shape(), order=order)
    # print("max: "+format(np.max(mapped_data)))


    new_img = nib.MGHImage(mapped_data, h1.get_affine(), h1)

    return new_img

def reorient_standard_RAS(img):
    ornt = np.array([[0, 1], [1, -1], [2, -1]])
    img_orient = img.as_reoriented(ornt) # re-orient the image
    return img

def conform_std_itk(img, order=1, conform_type = 2, intensity_rescaling = True):
    img = sitk.DICOMOrient(img, 'RAS')
    
    max_shape = 256
    min_zoom = 1.0
    new_size = (max_shape, max_shape, max_shape)
    new_spacing = (min_zoom, min_zoom, min_zoom)
    old_size = img.GetSize()
    old_spacing = img.GetSpacing()
    old_origin = img.GetOrigin()
    old_center = [old_origin[i] + old_spacing[i] * (old_size[i] - 1) / 2.0 for i in range(3)]
    new_origin = [old_center[i] - new_spacing[i] * (new_size[i] - 1) / 2.0 for i in range(3)]
    reference_image = sitk.Image(new_size, img.GetPixelIDValue())
    reference_image.SetOrigin(new_origin)
    reference_image.SetSpacing(new_spacing)
    reference_image.SetDirection(img.GetDirection())
    
    # resample the original image to the reference image using linear interpolation
    mapped_data_itk = sitk.Resample(img, reference_image, sitk.Transform(), sitk.sitkLinear, 0.0)
    mapped_data = sitk.GetArrayFromImage(mapped_data_itk)
    if intensity_rescaling:
        if not sitk.GetArrayFromImage(img) == np.dtype(np.uint8) or sitk.GetArrayFromImage(img).max() != 255 or sitk.GetArrayFromImage(img).min() != 0:
            src_min, scale = getscale(sitk.GetArrayFromImage(img), 0, 255)
            mapped_data = scalecrop(mapped_data, 0, 255, src_min, scale)
    else:
        mapped_data = 255*(mapped_data - mapped_data.min()) / (mapped_data.max() - mapped_data.min())
    # new_data = np.uint8(np.rint(mapped_data))
    new_img = sitk.GetImageFromArray(mapped_data, isVector = False)
    new_img.CopyInformation(mapped_data_itk)

    return new_img

def conform_itk(img, volshow, intensity_rescaling=True, uf_h=2, uf_w=2, uf_z=2,
                is_mask=False, order=3, rescale_first=False, use_scipy=True):
    # Get original image parameters
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
    original_direction = img.GetDirection()
    original_origin = img.GetOrigin()

    # Calculate new spacing
    new_spacing = [original_spacing[0] / uf_h, original_spacing[1] / uf_w, original_spacing[2] / uf_z]
    
    # Calculate new size
    new_size = [int(round(original_size[0] * uf_h)),
                int(round(original_size[1] * uf_w)),
                int(round(original_size[2] * uf_z))]

    # Convert SimpleITK image to numpy array
    img_array = sitk.GetArrayFromImage(img)
    
    # Calculate zoom factors
    zoom_factors = [uf_h, uf_w, uf_z]
    
    if rescale_first:
        # Intensity rescaling first
        if intensity_rescaling:
            # Define your getscale and scalecrop functions or use equivalent operations
            src_min, scale = getscale(img_array, 0, 255)
            img_array = scalecrop(img_array, 0, 255, src_min, scale)
        
        # Apply Numpy zoom for resampling
        if use_scipy:
            resampled_img_array = zoom(np.swapaxes(img_array,0,-1), zoom_factors, order=order if not is_mask else 0)
            # Convert numpy array back to SimpleITK image
            resampled_img = sitk.GetImageFromArray(np.swapaxes(resampled_img_array, 0, -1))
            resampled_img.SetSpacing(new_spacing)
            resampled_img.SetOrigin(original_origin)
            resampled_img.SetDirection(original_direction)
        else:
            # Use SimpleITK's resample function if not using scipy
            resample_filter = sitk.ResampleImageFilter()
            resample_filter.SetSize(new_size)
            resample_filter.SetOutputSpacing(new_spacing)
            resample_filter.SetOutputDirection(original_direction)
            resample_filter.SetOutputOrigin(original_origin)
            # resample_filter.SetInterpolator(sitk.sitkLinear if not is_mask else sitk.sitkNearestNeighbor)
            if order == 1:
                interpolator = sitk.sitkLinear
            elif order == 0 or is_mask:
                interpolator = sitk.sitkNearestNeighbor
            else:
                interpolator = sitk.sitkBSpline
            resample_filter.SetInterpolator(interpolator)
            resampled_img = resample_filter.Execute(img)

        return resampled_img

    else:
        # Apply Numpy zoom for resampling
        if use_scipy:
            resampled_img_array = zoom(np.swapaxes(img_array,0,-1), zoom_factors, order=order if not is_mask else 0)
            # Convert numpy array back to SimpleITK image
            rescaled_img = sitk.GetImageFromArray(np.swapaxes(resampled_img_array, 0, -1))
            rescaled_img.SetSpacing(new_spacing)
            rescaled_img.SetOrigin(original_origin)
            rescaled_img.SetDirection(original_direction)
        else:
            # Use SimpleITK's resample function if not using scipy
            resample_filter = sitk.ResampleImageFilter()
            resample_filter.SetSize(new_size)
            resample_filter.SetOutputSpacing(new_spacing)
            resample_filter.SetOutputDirection(original_direction)
            resample_filter.SetOutputOrigin(original_origin)
            if order == 1:
                interpolator = sitk.sitkLinear
            elif order == 0 or is_mask:
                interpolator = sitk.sitkNearestNeighbor
            else:
                interpolator = sitk.sitkBSpline
            resample_filter.SetInterpolator(interpolator)
            rescaled_img = resample_filter.Execute(img)
            
        # Intensity rescaling
        if intensity_rescaling:
            resampled_img_array = sitk.GetArrayFromImage(rescaled_img)
            # Define your getscale and scalecrop functions or use equivalent operations
            src_min, scale = getscale(resampled_img_array, 0, 255)
            resampled_img_array = scalecrop(resampled_img_array, 0, 255, src_min, scale)
            # Convert numpy array back to SimpleITK image
            rescaled_img = sitk.GetImageFromArray(resampled_img_array)
            rescaled_img.SetSpacing(new_spacing)
            rescaled_img.SetOrigin(original_origin)
            rescaled_img.SetDirection(original_direction)
        
        return rescaled_img

def conform_itk2(img, volshow, intensity_rescaling=True, uf_h=2, uf_w=2, uf_z=2,
                is_mask=False, order=1, rescale_first=False, use_scipy=True):
    # Get original image parameters
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
    original_direction = img.GetDirection()
    original_origin = img.GetOrigin()

    # Calculate new spacing
    new_spacing = [original_spacing[0] / uf_h, original_spacing[1] / uf_w, original_spacing[2] / uf_z]
    
    # Calculate new size
    new_size = [int(round(original_size[0] * uf_h)),
                int(round(original_size[1] * uf_w)),
                int(round(original_size[2] * uf_z))]

    # Convert SimpleITK image to numpy array
    img_array = sitk.GetArrayFromImage(img)
    
    # Calculate zoom factors
    zoom_factors = [uf_h, uf_w, uf_z]
    
    # Apply Numpy zoom for resampling
    if use_scipy:
        resampled_img_array = zoom(np.swapaxes(img_array,0,-1), zoom_factors, order=order if not is_mask else 0)
        # Convert numpy array back to SimpleITK image
        rescaled_img = sitk.GetImageFromArray(np.swapaxes(resampled_img_array, 0, -1))
        rescaled_img.SetSpacing(new_spacing)
        rescaled_img.SetOrigin(original_origin)
        rescaled_img.SetDirection(original_direction)
    else:
        # Use SimpleITK's resample function if not using scipy
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetSize(new_size)
        resample_filter.SetOutputSpacing(new_spacing)
        resample_filter.SetOutputDirection(original_direction)
        resample_filter.SetOutputOrigin(original_origin)
        if order == 1:
            interpolator = sitk.sitkLinear
        elif order == 0 or is_mask:
            interpolator = sitk.sitkNearestNeighbor
        else:
            interpolator = sitk.sitkBSpline
        resample_filter.SetInterpolator(interpolator)
        rescaled_img = resample_filter.Execute(img)
        
    # Intensity rescaling
    if intensity_rescaling:
        resampled_img_array = sitk.GetArrayFromImage(rescaled_img)
        # Define your getscale and scalecrop functions or use equivalent operations
        src_min, src_max, scale = getscale2(resampled_img_array, 0, 1)
        resampled_img_array = scalecrop2(resampled_img_array, 0, 1, src_min, scale)
        # Convert numpy array back to SimpleITK image
        rescaled_img = sitk.GetImageFromArray(resampled_img_array)
        rescaled_img.SetSpacing(new_spacing)
        rescaled_img.SetOrigin(original_origin)
        rescaled_img.SetDirection(original_direction)
    else: 
        resampled_img_array = sitk.GetArrayFromImage(rescaled_img)
        src_min, src_max = np.min(resampled_img_array), np.max(resampled_img_array)
    
    return rescaled_img, src_min, src_max
    
# def conform_itk(img, volshow, intensity_rescaling=True, uf_h=2, uf_w=2, uf_z=2, is_mask=False, order=3, rescale_first=False):
#     # Get original image parameters
#     original_spacing = img.GetSpacing()
#     original_size = img.GetSize()
#     original_direction = img.GetDirection()
#     original_origin = img.GetOrigin()

#     # Calculate new spacing
#     new_spacing = [original_spacing[0] / uf_h, original_spacing[1] / uf_w, original_spacing[2] / uf_z]
    
#     # Calculate new size
#     new_size = [int(round(original_size[0] * uf_h)),
#                 int(round(original_size[1] * uf_w)),
#                 int(round(original_size[2] * uf_z))]

#     # Convert SimpleITK image to numpy array
#     img_array = sitk.GetArrayFromImage(img)
    
#     # Calculate zoom factors
#     # zoom_factors = [1/uf_h, 1/uf_w, 1/uf_z]
#     zoom_factors = [uf_h, uf_w, uf_z]
#     if rescale_first:
#         # Intensity rescaling first
#         if intensity_rescaling:
#             src_min, scale = getscale(img_array, 0, 255)
#             img_array = scalecrop(img_array, 0, 255, src_min, scale)
        
#         # Apply Numpy zoom for resampling
#         resampled_img_array = zoom(img_array, zoom_factors, order=order if not is_mask else 0)

#         # Convert numpy array back to SimpleITK image
#         resampled_img = sitk.GetImageFromArray(resampled_img_array)
#         resampled_img.SetSpacing(new_spacing)
#         resampled_img.SetOrigin(original_origin)
#         resampled_img.SetDirection(original_direction)

#         return resampled_img

#     else:
#         # Apply Numpy zoom for resampling
#         resampled_img_array = zoom(img_array, zoom_factors, order=order if not is_mask else 0)
        
#         # Intensity rescaling
#         if intensity_rescaling:
#             src_min, scale = getscale(resampled_img_array, 0, 255)
#             resampled_img_array = scalecrop(resampled_img_array, 0, 255, src_min, scale)
        
#         # Convert numpy array back to SimpleITK image
#         rescaled_img = sitk.GetImageFromArray(resampled_img_array)
#         rescaled_img.SetSpacing(new_spacing)
#         rescaled_img.SetOrigin(original_origin)
#         rescaled_img.SetDirection(original_direction)

#         return rescaled_img

def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False):
    
    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

# def conform_itk(img, intensity_rescaling=True, uf_h=2, uf_w=2, uf_z=2):
#     orig_img = sitk.GetArrayFromImage(img)
#     orig_size = img.GetSize()
#     orig_spacing = img.GetSpacing()

#     # Calculate new spacing based on upscaling factors
#     new_spacing = [orig_spacing[i] / upscale_factor for i, upscale_factor in enumerate((uf_h, uf_w, uf_z))]

#     if intensity_rescaling:
#         src_min, scale = getscale(orig_img, 0, 255)
#         mapped_data = scalecrop(orig_img, 0, 255, src_min, scale)
#     else:
#         mapped_data = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())

#     # Create a SimpleITK image
#     new_img = sitk.GetImageFromArray(mapped_data)
#     new_img.SetSpacing(new_spacing)

#     # Perform resampling directly on the original image to the desired spacing
#     resampled_img = sitk.Resample(img, new_img, sitk.Transform(), sitk.sitkBSpline)

#     return resampled_img

import scipy.ndimage as ndi

def resample_to_output_manual(img, voxel_size, order=1, out_shape=None):
    # Get the input image data, affine, and header
    in_data = img.get_fdata()
    in_affine = img.affine
    in_header = img.header

    # Get the input image shape and voxel size
    in_shape = in_data.shape
    in_voxel_size = nib.affines.voxel_sizes(in_affine)

    # Compute the output image shape and voxel size
    if out_shape is None:
        # Use voxel size to determine output shape
        out_shape = np.round(np.array(in_shape) * in_voxel_size / voxel_size).astype(int)
    else:
        # Use out_shape parameter to specify output shape
        out_shape = np.array(out_shape)

    # Compute the output image affine
    out_voxel_size = voxel_size
    out_affine = nib.affines.from_matvec(nib.affines.to_matvec(in_affine)[0] * in_voxel_size / out_voxel_size, in_affine[:3, 3])

    # Compute the transformation matrix from input to output coordinates
    transform = np.linalg.inv(out_affine) @ in_affine

    # Apply the transformation to the input image data using interpolation
    out_data = ndi.affine_transform(in_data, transform, output_shape=out_shape, order=order)

    # Create a new image with the output data, affine, and header
    out_img = nib.Nifti1Image(out_data, out_affine, in_header)

    return out_img

# def orient_RAS(img, conform_type=2, order=3):
#     # Getting shapes and zooms, and correct if the 3D-volume is unsqueezed in last dim
#     ishape = img.shape
#     if len(ishape) > 3:
#         ishape = ishape[0:3]
#     max_shape = max(ishape)
#     izoom = img.header.get_zooms()
#     if len(izoom) > 3:
#         izoom = izoom[0:3]
#     min_zoom = min(izoom)
    
#     if conform_type == 0:
#         out_shape = (max_shape, max_shape, max_shape)
#         voxel_size = (min_zoom, min_zoom, min_zoom)
#     elif conform_type == 1:
#         out_shape = (256, 256, 256)
#         voxel_size = (1.0, 1.0, 1.0)
#     elif conform_type == 2:
#         out_shape = ishape
#         voxel_size = izoom
    
#     # out_img = nib.processing.resample_to_output(img, voxel_size, order=order, out_shape = out_shape)
#     out_img = resample_to_output_manual(img, voxel_size, order=order, out_shape = out_shape)
#     out_img_reoriented = nib.as_closest_canonical(out_img, enforce_diag=True)
    
#     return out_img_reoriented

# def orient_RAS(img, conform_type=2, order=3):
#     from nibabel.freesurfer.mghformat import MGHHeader
    
#     # Getting shapes and zooms, and correct if the 3D-volume is unsqueezed in last dim
#     ishape = img.shape
#     s1 = ishape[0]
#     s2 = ishape[1]
#     s3 = ishape[2]
#     max_shape = max(s1, s2, s3)
    
#     izoom = img.header.get_zooms()
#     z1 = izoom[0]
#     z2 = izoom[1]
#     z3 = izoom[2]
#     min_zoom = min(z1, z2, z3)
    
    
#     if len(ishape)>3:
#         ishape = ishape[:3]
#     if len(izoom)>3:
#         izoom = izoom[:3]
#     h1 = MGHHeader.from_header(img.header)  # may copy some parameters if input was MGH format
    
#     if conform_type == 0:
#         out_shape = (max_shape, max_shape, max_shape)
#         voxel_size = (min_zoom, min_zoom, min_zoom)
#     elif conform_type == 1:
#         out_shape = (256, 256, 256)
#         voxel_size = (1.0, 1.0, 1.0)
#     elif conform_type == 2:
#         out_shape = ishape
#         voxel_size = izoom
    
#     h1.set_data_shape([out_shape[0], out_shape[1], out_shape[2], 1])
#     h1.set_zooms([voxel_size[0], voxel_size[1], voxel_size[2]])
#     h1['Mdc'] = [[-1, 0, 0], [0, 0, -1], [0, 1, 0]]
#     h1['fov'] = max_shape
#     h1['Pxyz_c'] = img.affine.dot(np.hstack((np.array(img.shape[:3]) / 2.0, [1])))[:3]
#     out_img_reoriented = map_image(img, h1.get_affine(), h1.get_data_shape(), order=order)
#     out_img_reoriented = nib.Nifti1Image(out_img_reoriented, h1.get_affine())
#     return out_img_reoriented

# def conform(img, order=1, conform_type = 2, intensity_rescaling = True):
#     img_reoriented = orient_RAS(img)
#     if intensity_rescaling:
#         if not img.get_data_dtype() == np.dtype(np.uint8) or img.get_fdata().max() != 255 or img.get_fdata().min() != 0:
#             src_min, src_max, scale = getscale(img.get_fdata(), 0, 255)
#             img_reoriented_data = scalecrop(img_reoriented.get_fdata(), 0, 255, src_min, scale)
#     else:
#         img_reoriented_data = img_reoriented.get_fdata()
#         img_reoriented_data = 255*(img_reoriented_data - img_reoriented_data.min()) / (img_reoriented_data.max() - img_reoriented_data.min())
#     img_reoriented_data = np.uint8(np.rint(img_reoriented_data))
#     img_reoriented_data = nib.Nifti1Image(img_reoriented_data, img_reoriented.affine, img_reoriented.header)
    
#     return img_reoriented_data


def conform(img, order=3, conform_type = 0, intensity_rescaling = False):
    """
    Python version of mri_convert -c, which turns image intensity values into UCHAR, reslices images to standard position, fills up
    slices to standard 256x256x256 format and enforces 1 mm isotropic voxel sizes.

    Difference to mri_convert -c is that we first interpolate (float image), and then rescale to uchar. mri_convert is
    doing it the other way. However, we compute the scale factor from the input to be more similar again

    :param nibabel.MGHImage img: loaded source image
    :param int order: interpolation order (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return:nibabel.MGHImage new_img: conformed image
    """
    # from nibabel.freesurfer.mghformat import MGHHeader
    
    # ishape = img.shape
    # s1 = ishape[0]
    # s2 = ishape[1]
    # s3 = ishape[2]
    # max_shape = max(s1, s2, s3)
    
    # izoom = img.header.get_zooms()
    # z1 = izoom[0]
    # z2 = izoom[1]
    # z3 = izoom[2]
    # min_zoom = min(z1, z2, z3)
    
    
    # cwidth = max_shape
    # csize = min_zoom
    
    # if conform_type == 2:
    #     csize = 0.25
    #     cwidth = 880
    
    # if len(ishape)>3:
    #     ishape = ishape[:3]
    # if len(izoom)>3:
    #     izoom = izoom[:3]
    # h1 = MGHHeader.from_header(img.header)  # may copy some parameters if input was MGH format

    # h1.set_data_shape([cwidth, cwidth, cwidth, 1])
    # h1.set_zooms([csize, csize, csize])
    # h1['Mdc'] = [[-1, 0, 0], [0, 0, -1], [0, 1, 0]]
    # h1['fov'] = cwidth
    # h1['Pxyz_c'] = img.affine.dot(np.hstack((np.array(img.shape[:3]) / 2.0, [1])))[:3]
    # mapped_data = map_image(img, h1.get_affine(), h1.get_data_shape(), order=order)
    
    img_ras = nib.funcs.as_closest_canonical(img, enforce_diag = True)
    mapped_data = img_ras.get_fdata()
    # print("max: "+format(np.max(mapped_data)))

    # if not img.get_data_dtype() == np.dtype(np.uint8) or img.get_fdata().max() != 255 or img.get_fdata().min() != 0:
    if intensity_rescaling:
        src_min, src_max, scale = getscale(mapped_data, 0, 255)
        mapped_data = scalecrop(mapped_data, 0, 255, src_min, scale)

    new_data = np.uint8(np.rint(mapped_data))
    # new_img = nib.MGHImage(new_data, h1.get_affine(), h1)+
    new_img = nib.Nifti1Image(new_data, img_ras.affine)

    # make sure we store uchar
    new_img.set_data_dtype(np.uint8)

    return new_img

def conform_fix(
        img, order=3, conform_type = 0, intensity_rescaling = False, keep_dims = False, dtype=None
) -> nib.MGHImage:
    """Python version of mri_convert -c.

    mri_convert -c by default turns image intensity values
    into UCHAR, reslices images to standard position, fills up slices to standard
    256x256x256 format and enforces 1mm or minimum isotropic voxel sizes.

    Parameters
    ----------
    img : nib.analyze.SpatialImage
        loaded source image
    order : int
        interpolation order (0=nearest,1=linear(default),2=quadratic,3=cubic)
    conform_vox_size : VoxSizeOption
        conform image the image to voxel size 1. (default), a
        specific smaller voxel size (0-1, for high-res), or automatically
        determine the 'minimum voxel size' from the image (value 'min').
        This assumes the smallest of the three voxel sizes.
    dtype : Optional[Type]
        the dtype to enforce in the image (default: UCHAR, as mri_convert -c)
    conform_to_1mm_threshold : Optional[float]
        the threshold above which the image is conformed to 1mm
        (default: ignore).

    Returns
    -------
    nib.MGHImage
        conformed image

    Notes
    -----
    Unlike mri_convert -c, we first interpolate (float image), and then rescale
    to uchar. mri_convert is doing it the other way around. However, we compute
    the scale factor from the input to increase similarity.

    """
    from nibabel.freesurfer.mghformat import MGHHeader

    
    ishape = img.shape
    s1 = ishape[0]
    s2 = ishape[1]
    s3 = ishape[2]
    max_shape = max(s1, s2, s3)
    conformed_img_size = max_shape
    izoom = img.header.get_zooms()
    z1 = izoom[0]
    z2 = izoom[1]
    z3 = izoom[2]
    min_zoom = min(z1, z2, z3)
    conformed_vox_size = min_zoom
    
    h1 = MGHHeader.from_header(
        img.header
    )  # may copy some parameters if input was MGH format
    h1.set_data_shape([s1, s2, s3, 1])
    if not keep_dims:
        h1.set_data_shape([conformed_img_size, conformed_img_size, conformed_img_size, 1])
        h1.set_zooms(
            [conformed_vox_size, conformed_vox_size, conformed_vox_size])  # --> h1['delta']  
        h1["Mdc"] = [[-1, 0, 0], [0, 0, -1], [0, 1, 0]]
        h1["fov"] = conformed_img_size * conformed_vox_size
        h1["Pxyz_c"] = img.affine.dot(np.hstack((np.array(img.shape[:3]) / 2.0, [1])))[:3]
    else:
        h1.set_data_shape([s1, s2, s3, 1])
        h1.set_zooms(
            [z1, z2, z3])  # --> h1['delta']  
        h1["Mdc"] = [[-1, 0, 0], [0, 0, -1], [0, 1, 0]]
        h1["fov"] = conformed_img_size 
        h1["Pxyz_c"] = img.affine.dot(np.hstack((np.array(img.shape[:3]) / 2.0, [1])))[:3]
    
        # Here, we are explicitly using MGHHeader.get_affine() to construct the affine as
        # MdcD = np.asarray(h1['Mdc']).T * h1['delta']
        # vol_center = MdcD.dot(hdr['dims'][:3]) / 2
        # affine = from_matvec(MdcD, h1['Pxyz_c'] - vol_center)
    affine = h1.get_affine()

    # from_header does not compute Pxyz_c (and probably others) when importing from nii
    # Pxyz is the center of the image in world coords

    # target scalar type and dtype
    sctype = np.uint8 if dtype is None else np.obj2sctype(dtype, default=np.uint8)
    target_dtype = np.dtype(sctype)

    src_min, scale = 0, 1.0
    # get scale for conversion on original input before mapping to be more similar to
    # mri_convert
    if (
        img.get_data_dtype() != np.dtype(np.uint8)
        or img.get_data_dtype() != target_dtype
    ):
        src_min, scale = getscale(np.asanyarray(img.dataobj), 0, 255)

    kwargs = {}
    if sctype != np.uint:
        kwargs["dtype"] = "float"
    mapped_data = map_image(img, affine, h1.get_data_shape(), order=order, **kwargs)

    if img.get_data_dtype() != np.dtype(np.uint8) or (
        img.get_data_dtype() != target_dtype and scale != 1.0
    ):
        scaled_data = scalecrop(mapped_data, 0, 255, src_min, scale)
        # map zero in input to zero in output (usually background)
        scaled_data[mapped_data == 0] = 0
        mapped_data = scaled_data

    if target_dtype == np.dtype(np.uint8):
        mapped_data = np.clip(np.rint(mapped_data), 0, 255)
    new_img = nib.MGHImage(sctype(mapped_data), affine, h1)

    # make sure we store uchar
    try:
        new_img.set_data_dtype(target_dtype)
    except nib.freesurfer.mghformat.MGHError as e:
        if "not recognized" in e.args[0]:
            codes = set(
                k.name
                for k in nib.freesurfer.mghformat.data_type_codes.code.keys()
                if isinstance(k, np.dtype)
            )
            print(
                f'The data type "{options.dtype}" is not recognized for MGH images, switching '
                f'to "{new_img.get_data_dtype()}" (supported: {tuple(codes)}).'
            )

    return new_img

def conform_keep_dims(img, order=1, conform_type = 0, intensity_rescaling = False):
    """
    Python version of mri_convert -c, which turns image intensity values into UCHAR, reslices images to standard position, fills up
    slices to standard 256x256x256 format and enforces 1 mm isotropic voxel sizes.

    Difference to mri_convert -c is that we first interpolate (float image), and then rescale to uchar. mri_convert is
    doing it the other way. However, we compute the scale factor from the input to be more similar again

    :param nibabel.MGHImage img: loaded source image
    :param int order: interpolation order (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return:nibabel.MGHImage new_img: conformed image
    """
    from nibabel.freesurfer.mghformat import MGHHeader
    
    ishape = img.shape
    izoom = img.header.get_zooms()
    
    h1 = MGHHeader.from_header(img.header)  # may copy some parameters if input was MGH format

    h1.set_data_shape([ishape[0], ishape[1], ishape[2], 1])
    h1.set_zooms([izoom[0], izoom[1], izoom[2]])
    h1['Mdc'] = [[-1, 0, 0], [0, 0, -1], [0, 1, 0]]
    h1['fov'] = max(ishape)
    h1['Pxyz_c'] = img.affine.dot(np.hstack((np.array(img.shape[:3]) / 2.0, [1])))[:3]

    # from_header does not compute Pxyz_c (and probably others) when importing from nii
    # Pxyz is the center of the image in world coords

    # get scale for conversion on original input before mapping to be more similar to mri_convert
    mapped_data = map_image(img, h1.get_affine(), h1.get_data_shape(), order=order)
    # print("max: "+format(np.max(mapped_data)))
    
    if intensity_rescaling:
        if not img.get_data_dtype() == np.dtype(np.uint8) or img.get_fdata().max() != 255 or img.get_fdata().min() != 0:
            src_min, scale = getscale(img.get_fdata(), 0, 255)
            mapped_data = scalecrop(mapped_data, 0, 255, src_min, scale)
    else:
        new_data = (mapped_data - mapped_data.min()) / (mapped_data.max() - mapped_data.min())
    

    # new_data = np.rint(mapped_data)
    new_img = nib.MGHImage(new_data, h1.get_affine(), h1)

    # # make sure we store uchar
    # new_img.set_data_dtype(np.uint8)

    return new_img

def deconform(img, orig_shape, orig_zoom, order=1, orig_max_intens=255, orig_min_intens=0):
    """
    Python version of mri_convert -c, which turns image intensity values into UCHAR, reslices images to standard position, fills up
    slices to standard 256x256x256 format and enforces 1 mm isotropic voxel sizes.

    Difference to mri_convert -c is that we first interpolate (float image), and then rescale to uchar. mri_convert is
    doing it the other way. However, we compute the scale factor from the input to be more similar again

    :param nibabel.MGHImage img: loaded source image
    :param int order: interpolation order (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return:nibabel.MGHImage new_img: conformed image
    """
    from nibabel.freesurfer.mghformat import MGHHeader
    cwidth1, cwidth2, cwidth3 = orig_shape
    csize1, csize2, csize3 = orig_zoom
    
  
        
    h1 = MGHHeader.from_header(img.header)  # may copy some parameters if input was MGH format

    h1.set_data_shape([cwidth1, cwidth2, cwidth3, 1])
    h1.set_zooms([csize1, csize2, csize3])
    h1['Mdc'] = [[-1, 0, 0], [0, 0, -1], [0, 1, 0]]
    h1['fov'] = cwidth1
    h1['Pxyz_c'] = img.affine.dot(np.hstack((np.array(img.shape[:3]) / 2.0, [1])))[:3]

    # from_header does not compute Pxyz_c (and probably others) when importing from nii
    # Pxyz is the center of the image in world coords

    # get scale for conversion on original input before mapping to be more similar to mri_convert

    mapped_data = map_image(img, h1.get_affine(), h1.get_data_shape(), order=order)
    # # print("max: "+format(np.max(mapped_data)))
    # new_data = np.uint8(np.rint(mapped_data))
    # new_data = orig_max_intens*((mapped_data - mapped_data.min()) / (mapped_data.max() - mapped_data.min()))
    new_img = nib.MGHImage(mapped_data, h1.get_affine(), h1)
    
    # make sure we store uchar
    new_img.set_data_dtype(np.single)

    return new_img

def onlyscale(img, ishape, zoom, order=1):
    """
    Python version of mri_convert -c, which turns image intensity values into UCHAR, reslices images to standard position, fills up
    slices to standard 256x256x256 format and enforces 1 mm isotropic voxel sizes.

    Difference to mri_convert -c is that we first interpolate (float image), and then rescale to uchar. mri_convert is
    doing it the other way. However, we compute the scale factor from the input to be more similar again

    :param nibabel.MGHImage img: loaded source image
    :param int order: interpolation order (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return:nibabel.MGHImage new_img: conformed image
    """
    from nibabel.freesurfer.mghformat import MGHHeader
    
    h1 = MGHHeader.from_header(img.header)  # may copy some parameters if input was MGH format
    h1.set_data_shape([ishape[0], ishape[1], ishape[2], 1])
    h1.set_zooms([zoom[0], zoom[1], zoom[2]])
    h1['Mdc'] = [[-1, 0, 0], [0, 0, -1], [0, 1, 0]]
    h1['fov'] = zoom[0]
    h1['Pxyz_c'] = img.affine.dot(np.hstack((np.array(img.shape[:3]) / 2.0, [1])))[:3]
    # get scale for conversion on original input before mapping to be more similar to mri_convert
    src_min, scale = getscale(img.get_fdata(), 0, 255)

    mapped_data = map_image(img, h1.get_affine(), h1.get_data_shape(), order=order)
    # print("max: "+format(np.max(mapped_data)))

    if not img.get_data_dtype() == np.dtype(np.uint8):
        mapped_data = scalecrop(mapped_data, 0, 255, src_min, scale)

    new_data = np.uint8(np.rint(mapped_data))
    new_img = nib.MGHImage(new_data, h1.get_affine(), h1)

    # make sure we store uchar
    new_img.set_data_dtype(np.uint8)

    return new_img

def resize_to(img, dst_s1, dst_s2, dst_s3, dst_zoom1, dst_zoom2, dst_zoom3, order = 1):
    """
    
    :param int order: interpolation order (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return:nibabel.MGHImage new_img: conformed image
    """
    from nibabel.freesurfer.mghformat import MGHHeader
    
    h1 = MGHHeader.from_header(img.header)  # may copy some parameters if input was MGH format

    h1.set_data_shape([dst_s1, dst_s2, dst_s3, 1])
    h1.set_zooms([dst_zoom1, dst_zoom2, dst_zoom3])
    h1['Mdc'] = [[-1, 0, 0], [0, 0, -1], [0, 1, 0]]
    h1['fov'] = dst_s1
    h1['Pxyz_c'] = img.affine.dot(np.hstack((np.array(img.shape[:3]) / 2.0, [1])))[:3]

    # from_header does not compute Pxyz_c (and probably others) when importing from nii
    # Pxyz is the center of the image in world coords

    # get scale for conversion on original input before mapping to be more similar to mri_convert
    src_min, scale = getscale(img.get_fdata(), 0, 255)

    mapped_data = map_image(img, h1.get_affine(), h1.get_data_shape(), order=order)
    # print("max: "+format(np.max(mapped_data)))

    if not img.get_data_dtype() == np.dtype(np.uint8):

        if np.max(mapped_data) > 255:
            mapped_data = scalecrop(mapped_data, 0, 255, src_min, scale)

    new_data = np.uint8(np.rint(mapped_data))
    new_img = nib.MGHImage(new_data, h1.get_affine(), h1)

    # make sure we store uchar
    new_img.set_data_dtype(np.uint8)

    return new_img

def resize_as(img, dst_s1, dst_s2, dst_s3, dst_zoom1, dst_zoom2, dst_zoom3, order = 1):
    """
    
    :param int order: interpolation order (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return:nibabel.MGHImage new_img: conformed image
    """
    from nibabel.freesurfer.mghformat import MGHHeader
    
    h1 = MGHHeader.from_header(img.header)  # may copy some parameters if input was MGH format

    h1.set_data_shape([dst_s1, dst_s2, dst_s3, 1])
    h1.set_zooms([dst_zoom1, dst_zoom2, dst_zoom3])
    h1['Mdc'] = [[-1, 0, 0], [0, 0, -1], [0, 1, 0]]
    h1['fov'] = dst_s1
    h1['Pxyz_c'] = img.affine.dot(np.hstack((np.array(img.shape[:3]) / 2.0, [1])))[:3]

    # from_header does not compute Pxyz_c (and probably others) when importing from nii
    # Pxyz is the center of the image in world coords

    # get scale for conversion on original input before mapping to be more similar to mri_convert
    mapped_data = map_image(img, h1.get_affine(), h1.get_data_shape(), order=order)
    # print("max: "+format(np.max(mapped_data)))

    new_img = nib.MGHImage(mapped_data, h1.get_affine(), h1)

    # make sure we store uchar
    new_img.set_data_dtype(np.uint8)

    return new_img

def conform_mask(mask, order=1):
    """
    Python version of mri_convert -c, which turns image intensity values into UCHAR, reslices images to standard position, fills up
    slices to standard 256x256x256 format and enforces 1 mm isotropic voxel sizes.

    Difference to mri_convert -c is that we first interpolate (float image), and then rescale to uchar. mri_convert is
    doing it the other way. However, we compute the scale factor from the input to be more similar again

    :param nibabel.MGHImage img: loaded source image
    :param int order: interpolation order (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return:nibabel.MGHImage new_img: conformed image
    """
    from nibabel.freesurfer.mghformat import MGHHeader
    
    ishape = mask.shape
    max_shape = max(ishape)
    izoom = mask.header.get_zooms()
    min_zoom = min(izoom)
    
    
    cwidth = max_shape
    csize = min_zoom
    
    h1 = MGHHeader.from_header(mask.header)  # may copy some parameters if input was MGH format

    h1.set_data_shape([cwidth, cwidth, cwidth, 1])
    h1.set_zooms([csize, csize, csize])
    h1['Mdc'] = [[-1, 0, 0], [0, 0, -1], [0, 1, 0]]
    h1['fov'] = cwidth
    h1['Pxyz_c'] = mask.affine.dot(np.hstack((np.array(mask.shape[:3]) / 2.0, [1])))[:3]
    mapped_data = map_image(mask, h1.get_affine(), h1.get_data_shape(), order=order)
    new_img = nib.MGHImage(mapped_data, h1.get_affine(), h1)

    # make sure we store uchar
    new_img.set_data_dtype(np.uint8)

    return new_img

def conform_std(img, order=1, conform_type = 0):
    """
    Python version of mri_convert -c, which turns image intensity values into UCHAR, reslices images to standard position, fills up
    slices to standard 256x256x256 format and enforces 1 mm isotropic voxel sizes.

    Difference to mri_convert -c is that we first interpolate (float image), and then rescale to uchar. mri_convert is
    doing it the other way. However, we compute the scale factor from the input to be more similar again

    :param nibabel.MGHImage img: loaded source image
    :param int order: interpolation order (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return:nibabel.MGHImage new_img: conformed image
    """
    from nibabel.freesurfer.mghformat import MGHHeader
    
    max_shape = 256
    min_zoom = 1.0
    
    
    cwidth = max_shape
    csize = min_zoom
    
    h1 = MGHHeader.from_header(img.header)  # may copy some parameters if input was MGH format

    h1.set_data_shape([cwidth, cwidth, cwidth, 1])
    h1.set_zooms([csize, csize, csize])
    h1['Mdc'] = [[-1, 0, 0], [0, 0, -1], [0, 1, 0]]
    h1['fov'] = cwidth
    h1['Pxyz_c'] = img.affine.dot(np.hstack((np.array(img.shape[:3]) / 2.0, [1])))[:3]

    # from_header does not compute Pxyz_c (and probably others) when importing from nii
    # Pxyz is the center of the image in world coords

    # get scale for conversion on original input before mapping to be more similar to mri_convert
    src_min, src_max, scale = getscale(img.get_fdata(), 0, 255)

    mapped_data = map_image(img, h1.get_affine(), h1.get_data_shape(), order=order)
    # print("max: "+format(np.max(mapped_data)))

    if not img.get_data_dtype() == np.dtype(np.uint8):

        if np.max(mapped_data) > 255:
            mapped_data = scalecrop(mapped_data, 0, 255, src_min, scale)

    new_data = np.uint8(np.rint(mapped_data))
    new_img = nib.MGHImage(new_data, h1.get_affine(), h1)

    # make sure we store uchar
    new_img.set_data_dtype(np.uint8)

    return new_img

def conform_std_mask(mask, order=1):
    """
    Python version of mri_convert -c, which turns image intensity values into UCHAR, reslices images to standard position, fills up
    slices to standard 256x256x256 format and enforces 1 mm isotropic voxel sizes.

    Difference to mri_convert -c is that we first interpolate (float image), and then rescale to uchar. mri_convert is
    doing it the other way. However, we compute the scale factor from the input to be more similar again

    :param nibabel.MGHImage img: loaded source image
    :param int order: interpolation order (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return:nibabel.MGHImage new_img: conformed image
    """
    from nibabel.freesurfer.mghformat import MGHHeader
    
    max_shape = 256
    min_zoom = 1.0
    
    
    cwidth = max_shape
    csize = min_zoom
    
    h1 = MGHHeader.from_header(mask.header)  # may copy some parameters if input was MGH format

    h1.set_data_shape([cwidth, cwidth, cwidth, 1])
    h1.set_zooms([csize, csize, csize])
    h1['Mdc'] = [[-1, 0, 0], [0, 0, -1], [0, 1, 0]]
    h1['fov'] = cwidth
    h1['Pxyz_c'] = mask.affine.dot(np.hstack((np.array(mask.shape[:3]) / 2.0, [1])))[:3]
    mapped_data = map_image(mask, h1.get_affine(), h1.get_data_shape(), order=order)
    new_img = nib.MGHImage(mapped_data, h1.get_affine(), h1)

    # make sure we store uchar
    new_img.set_data_dtype(np.uint8)

    return new_img

def is_conform(img, eps=1e-06):
    """
    Function to check if an image is already conformed or not (Dimensions: 256x256x256, Voxel size: 1x1x1, and
    LIA orientation.

    :param nibabel.MGHImage img: Loaded source image
    :param float eps: allowed deviation from zero for LIA orientation check (default 1e-06).
                      Small inaccuracies can occur through the inversion operation. Already conformed images are
                      thus sometimes not correctly recognized. The epsilon accounts for these small shifts.
    :return: True if image is already conformed, False otherwise
    """
    ishape = img.shape
    max_size = max(ishape)
    if len(ishape) > 3 and ishape[3] != 1:
        sys.exit('ERROR: Multiple input frames (' + format(img.shape[3]) + ') not supported!')

    # check dimensions
    if ishape[0] != max_size or ishape[1] != max_size or ishape[2] != max_size:
        return False

    # check voxel size
    izoom = img.header.get_zooms()
    min_zoom = min(izoom)
    # min_zoom = 1.2
    if izoom[0] != min_zoom or izoom[1] != min_zoom or izoom[2] != min_zoom:
        return False

    # check orientation LIA
    iaffine = img.affine[0:3, 0:3] + [[1, 0, 0], [0, 0, -1], [0, 1, 0]]

    if np.max(np.abs(iaffine)) > 0.0 + eps:
        return False

    return True

def is_conform_itk(img, eps=1e-06):
        # Get the affine transform from the image origin, spacing and direction
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(img.GetDirection())
    affine.SetTranslation(img.GetOrigin())
    affine.SetCenter(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
    
    # Convert the affine transform to a numpy array
    affine_array = np.array(affine.GetParameters()).reshape(3, 4)
    affine_array = np.vstack([affine_array, [0, 0, 0, 1]])
    
    # Apply the same transformation as in the nibabel code
    iaffine = affine_array[0:3, 0:3] + [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    
    # Check the condition
    eps = 1e-6
    if np.max(np.abs(iaffine)) > 0.0 + eps:
        return False

    return True


def check_affine_in_nifti(img, logger=None):
    """
    Function to check affine in nifti Image. Sets affine with qform if it exists and differs from sform.
    If qform does not exist, voxelsizes between header information and information in affine are compared.
    In case these do not match, the function returns False (otherwise returns True.

    :param nibabel.NiftiImage img: loaded nifti-image
    :return bool: True, if: affine was reset to qform
                            voxelsizes in affine are equivalent to voxelsizes in header
                  False, if: voxelsizes in affine and header differ
    """
    check = True
    message = ""

    if img.header['qform_code'] != 0 and np.max(np.abs(img.get_sform() - img.get_qform())) > 0.001:
        message = "#############################################################" \
                  "\nWARNING: qform and sform transform are not identical!\n sform-transform:\n{}\n qform-transform:\n{}\n" \
                  "You might want to check your Nifti-header for inconsistencies!" \
                  "\n!!! Affine from qform transform will now be used !!!\n" \
                  "#############################################################".format(img.header.get_sform(),
                                                                                         img.header.get_qform())
        # Set sform with qform affine and update best affine in header
        img.set_sform(img.get_qform())
        img.update_header()

    else:
        # Check if affine correctly includes voxel information and print Warning/Exit otherwise
        vox_size_head = img.header.get_zooms()
        aff = img.affine
        xsize = np.sqrt(aff[0][0] * aff[0][0] + aff[1][0] * aff[1][0] + aff[2][0] * aff[2][0])
        ysize = np.sqrt(aff[0][1] * aff[0][1] + aff[1][1] * aff[1][1] + aff[2][1] * aff[2][1])
        zsize = np.sqrt(aff[0][2] * aff[0][2] + aff[1][2] * aff[1][2] + aff[2][2] * aff[2][2])

        if (abs(xsize - vox_size_head[0]) > .001) or (abs(ysize - vox_size_head[1]) > .001) or (abs(zsize - vox_size_head[2]) > 0.001):
            message = "#############################################################\n" \
                      "ERROR: Invalid Nifti-header! Affine matrix is inconsistent with Voxel sizes. " \
                      "\nVoxel size (from header) vs. Voxel size in affine: " \
                      "({}, {}, {}), ({}, {}, {})\nInput Affine----------------\n{}\n" \
                      "#############################################################".format(vox_size_head[0],
                                                                                             vox_size_head[1],
                                                                                             vox_size_head[2],
                                                                                             xsize, ysize, zsize,
                                                                                             aff)
            check = False

    if logger is not None:
        logger.info(message)

    else:
        print(message)

    return check

def check_affine_in_nifti_itk(img, logger=None):
    """
    Function to check affine in nifti Image. Sets affine with qform if it exists and differs from sform.
    If qform does not exist, voxelsizes between header information and information in affine are compared.
    In case these do not match, the function returns False (otherwise returns True.

    :param sitk.Image img: loaded nifti-image
    :return bool: True, if: affine was reset to qform
                            voxelsizes in affine are equivalent to voxelsizes in header
                  False, if: voxelsizes in affine and header differ
    """
    check = True
    message = ""
    
    # Get the direction, origin and spacing of the image
    direction = np.array(img.GetDirection()).reshape(3,3)
    origin = np.array(img.GetOrigin())
    spacing = np.array(img.GetSpacing())
    
    # Calculate the affine matrix from the direction, origin and spacing
    aff = np.eye(4)
    aff[:3,:3] = direction * spacing[:,np.newaxis]
    aff[:3,3] = origin
    
    # Check if affine correctly includes voxel information and print Warning/Exit otherwise
    xsize, ysize, zsize = spacing

    if (abs(xsize - spacing[0]) > .001) or (abs(ysize - spacing[1]) > .001) or (abs(zsize - spacing[2]) > 0.001):
        message = "#############################################################\n" \
                  "ERROR: Invalid Nifti-header! Affine matrix is inconsistent with Voxel sizes. " \
                  "\nVoxel size (from header) vs. Voxel size in affine: " \
                  "({}, {}, {}), ({}, {}, {})\nInput Affine----------------\n{}\n" \
                  "#############################################################".format(spacing[0],
                                                                                         spacing[1],
                                                                                         spacing[2],
                                                                                         xsize, ysize, zsize,
                                                                                         aff)
        check = False

    if logger is not None:
        logger.info(message)

    else:
        print(message)

    return check

if __name__ == "__main__":
    # Command Line options are error checking done here
    options = options_parse()

    print("Reading input: {} ...".format(options.input))
    image = nib.load(options.input)

    if len(image.shape) > 3 and image.shape[3] != 1:
        sys.exit('ERROR: Multiple input frames (' + format(image.shape[3]) + ') not supported!')

    if is_conform(image):
        sys.exit("Input " + format(options.input) + " is already conform! No output created.\n")

    # If image is nifti image
    if options.input[-7:] == ".nii.gz" or options.input[-4:] == ".nii":

        if not check_affine_in_nifti(image):
            sys.exit("ERROR: inconsistency in nifti-header. Exiting now.\n")

    new_image = conform(image, options.order)
    print ("Writing conformed image: {}".format(options.output))

    nib.save(new_image, options.output)

    sys.exit(0)


