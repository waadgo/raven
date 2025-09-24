import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
import warnings
from pathlib import Path
from math import floor

# Suppress the specific Matplotlib warning about Z-ordering
warnings.filterwarnings(
    "ignore",
    message="Attempting to set identical bottom == top results in singular transformations",
    category=UserWarning
)

def create_mri_cube_figure(
    image_path: str,
    output_path: str,
    patch_size_mm: tuple = (64, 64, 64),
    intensity_window: tuple = None,
    patch_center: tuple = None,
    dpi: int = 600,
    view_angles: tuple = (30, -45),
    figsize: tuple = (8, 8)
):
    """
    Creates a clean 3D visualization of an MRI sub-volume and saves it to a file.

    This function extracts a cubic patch from a 3D MRI based on physical dimensions (mm),
    normalizes it using a specified intensity window, and displays three faces
    (axial, sagittal, coronal) as a texture-mapped cube. The output is a clean
    image with no axes, titles, or padding.

    Args:
        image_path (str): Path to the MRI file (e.g., .nii.gz, .mha, .mnc).
        output_path (str): Path to save the output PNG image.
        patch_size_mm (tuple, optional): The (depth, height, width) of the cube in millimeters.
        intensity_window (tuple, optional): The (min, max) intensity values for visualization.
                                            If None, the patch is auto-scaled.
        patch_center (tuple, optional): The (z, y, x) Voxel center of the cube.
                                        If None, the image center is used.
        dpi (int, optional): Resolution of the saved figure in dots per inch.
        view_angles (tuple, optional): The (elevation, azimuth) for the 3D plot.
        figsize (tuple, optional): The size of the figure in inches.
    """
    # 1. Read the image and its metadata using SimpleITK
    try:
        image = sitk.ReadImage(image_path, sitk.sitkFloat32)
        image_array = sitk.GetArrayFromImage(image)
        spacing = image.GetSpacing()  # Returns (x, y, z)
    except Exception as e:
        print(f"❌ Error reading image file: {e}")
        return

    # 2. Convert patch size from millimeters to voxels
    # patch_size_mm is (depth, height, width) which corresponds to (z, y, x)
    # spacing is (x_spacing, y_spacing, z_spacing)
    patch_size_voxels = (
        int(round(patch_size_mm[0] / spacing[2])),  # Depth (z)
        int(round(patch_size_mm[1] / spacing[1])),  # Height (y)
        int(round(patch_size_mm[2] / spacing[0])),  # Width (x)
    )
    print(f"ℹ️ Image spacing (x, y, z) mm/voxel: {np.round(spacing, 2)}")
    print(f"ℹ️ Requested patch size (depth, height, width) in mm: {patch_size_mm}")
    print(f"ℹ️ Calculated patch size (depth, height, width) in voxels: {patch_size_voxels}")

    # 3. Define the patch (cube) to be visualized
    img_dims = image_array.shape
    if patch_center is None:
        patch_center = (img_dims[0] // 2, img_dims[1] // 2, img_dims[2] // 2)

    z_start = max(0, patch_center[0] - patch_size_voxels[0] // 2)
    z_end = min(img_dims[0], z_start + patch_size_voxels[0])
    y_start = max(0, patch_center[1] - patch_size_voxels[1] // 2)
    y_end = min(img_dims[1], y_start + patch_size_voxels[1])
    x_start = max(0, patch_center[2] - patch_size_voxels[2] // 2)
    x_end = min(img_dims[2], x_start + patch_size_voxels[2])

    patch_data = image_array[z_start:z_end, y_start:y_end, x_start:x_end]

    if patch_data.size == 0:
        print("❌ Error: The calculated patch is empty. Check patch_center and patch_size_mm.")
        return

    # 4. Normalize patch based on the specified intensity window
    if intensity_window:
        min_val, max_val = intensity_window
    else:
        min_val, max_val = np.min(patch_data), np.max(patch_data)
        print(f"ℹ️ Auto-scaling intensity to window: ({min_val:.2f}, {max_val:.2f})")

    clipped_patch = np.clip(patch_data, min_val, max_val)
    if max_val > min_val:
        norm_patch = (clipped_patch - min_val) / (max_val - min_val)
    else:
        norm_patch = np.zeros_like(clipped_patch)

    # 5. Create the 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Define coordinates for the cube faces. Using voxel indices for plotting.
    x_coords = np.arange(patch_size_voxels[2])
    y_coords = np.arange(patch_size_voxels[1])
    z_coords = np.arange(patch_size_voxels[0])

    if any(arr.size == 0 for arr in [x_coords, y_coords, z_coords]):
        print("❌ Error: One of the patch dimensions is zero. Cannot create plot.")
        return

    # Meshgrids for the faces
    xx, yy = np.meshgrid(x_coords, y_coords)
    xx_z, zz_y = np.meshgrid(x_coords, z_coords)
    yy_z, zz_x = np.meshgrid(y_coords, z_coords)

    # 6. Plot the three visible faces using plot_surface
    # Axial face (top)
    ax.plot_surface(xx, yy, np.full_like(xx, z_coords[-1]), facecolors=plt.cm.gray(norm_patch[-1, :, :]), rstride=1, cstride=1, shade=False)
    # Coronal face (front)
    ax.plot_surface(xx_z, np.full_like(xx_z, y_coords[-1]), zz_y, facecolors=plt.cm.gray(norm_patch[:, -1, :]), rstride=1, cstride=1, shade=False)
    # Sagittal face (side)
    ax.plot_surface(np.full_like(yy_z, x_coords[-1]), yy_z, zz_x, facecolors=plt.cm.gray(norm_patch[:, :, -1]), rstride=1, cstride=1, shade=False)

    # 7. Customize the plot for clean, publication-ready output
    ax.set_box_aspect((np.ptp(x_coords), np.ptp(y_coords), np.ptp(z_coords)))
    ax.view_init(elev=view_angles[0], azim=view_angles[1])
    ax.set_axis_off()  # Remove axes, labels, and background panes

    # 8. Save the figure and close the plot
    try:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        print(f"✔️ Figure successfully saved to {output_path}")
    except Exception as e:
        print(f"❌ Error saving figure: {e}")
    finally:
        plt.close(fig)  # Prevent the plot from being displayed interactively

if __name__ == '__main__':
    # --- Create a Dummy MRI File for Demonstration ---
    # In your actual use, you will replace this with the path to your file.
    # dummy_shape = (256, 256, 256)
    # dummy_image = sitk.GetImageFromArray(np.random.rand(*dummy_shape) * 1000)
    # dummy_image.SetSpacing([0.8, 0.8, 1.0]) # Example spacing in mm
    # dummy_image_path = "dummy_mri_volume.nii.gz"
    # sitk.WriteImage(dummy_image, dummy_image_path)
    # mri_file_path = dummy_image_path
    # ---------------------------------------------------

    # 🔹 --- Parameters You Can Modify --- 🔹

    # 1. Path to your MRI file (e.g., .nii.gz, .mha, .mnc)
    # mri_file_path = "/mnt/c/Users/walte/Downloads/PRESENTATION_CANDIDACY/HCP_103212_v01_t1w_pp0.nii.gz"
    mri_file_path = "/mnt/c/Users/walte/Downloads/PRESENTATION_CANDIDACY/HCP_103212_v01_t1w_pp1/original/original.nii.gz"

    # 3. Size of the cubic patch in MILLIMETERS: (depth, height, width)
    patch_dimensions_mm = (150, 50, 150)

    # 3. Intensity window for visualization: (min_intensity, max_intensity).
    #    Set to None for auto-scaling based on the patch's min/max values.

    # intensity_range = (0, 1200)  # Example for a T1-weighted brain scan
    intensity_range = (0, 80)  # Example for a T1-weighted brain scan

    # 4. (Optional) Voxel coordinates for the center of the patch: (z, y, x).
    #    Set to None to automatically use the center of the whole image.
    patch_center_coords = None

    # 5. Viewing angle: (elevation, azimuth)
    viewing_perspective = (25, 30)

    # 6. Quality of the output figure in Dots Per Inch (DPI)
    figure_resolution_dpi = 1000

    # 7. Figure size in inches (affects rendering resolution)
    figure_size_inches = (20, 20)

    # --- Automatic Output Path Generation ---
    if "PATH/TO/YOUR" not in mri_file_path:
        # Create a Path object from the input file path
        input_path = Path(mri_file_path)

        # Get the base name of the file by removing extensions (handles .nii.gz)
        file_stem = input_path.name
        while Path(file_stem).suffix:
            file_stem = Path(file_stem).stem
        
        # Get the full path including directory, but without the extension
        base_path = input_path.with_name(file_stem)

        # Create informative strings from parameters for the filename
        patch_str = f"p{'x'.join(map(str, patch_dimensions_mm))}mm"
        
        if intensity_range:
            intensity_str = f"i{intensity_range[0]}-{intensity_range[1]}"
        else:
            intensity_str = "iAUTO"
            
        view_str = f"v{viewing_perspective[0]}el{viewing_perspective[1]}az"

        # Combine parts to create the final output path with a .png extension
        output_figure_path = f"{base_path}_{patch_str}_{intensity_str}_{view_str}.png"
        
        print(f"🖼️  Generated output path: {output_figure_path}")

    else:
        # Default fallback path if the input path is still the placeholder
        output_figure_path = "mri_patch_visualization.png"

    # --- End of Parameter Setup ---

    if "PATH/TO/YOUR" in mri_file_path:
        print("="*60)
        print("⚠️  PLEASE UPDATE THE 'mri_file_path' VARIABLE WITH A REAL PATH")
        print("="*60)
    else:
        create_mri_cube_figure(
            image_path=mri_file_path,
            output_path=output_figure_path,
            patch_size_mm=patch_dimensions_mm,
            intensity_window=intensity_range,
            patch_center=patch_center_coords,
            view_angles=viewing_perspective,
            dpi=figure_resolution_dpi,
            figsize=figure_size_inches,
        )