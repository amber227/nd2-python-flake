import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology, segmentation

def process_image_and_get_diameters(image, block_size=51, min_blob_area=30, debug_visualize=True):
    """
    - image: 2D numpy array (grayscale, float)
    - block_size: *odd* integer, size of region for local thresholding
    - min_blob_area: minimum number of pixels for a blob to be considered valid
    - debug_visualize: display diagnostics overlay
    Returns:
        - diameters: list of diameters (float, in pixels)
        - boundaries: list of (N,2) ndarray coords for blob contours
    """
    # 1. Local thresholding
    # Normalize to [0, 1]
    img_norm = (image - image.min()) / (image.max() - image.min() + 1e-6)
    # Adaptive thresholding
    local_thresh = filters.threshold_local(img_norm, block_size)
    binary_mask = img_norm > local_thresh
    # Remove small objects, fill holes
    binary_mask = morphology.remove_small_objects(binary_mask, min_size=min_blob_area)
    binary_mask = morphology.remove_small_holes(binary_mask, area_threshold=64)

    # 2. Label blobs
    label_img = measure.label(binary_mask)
    props = measure.regionprops(label_img)

    diameters = []
    boundaries = []

    for prop in props:
        if prop.area >= min_blob_area:
            # Equivalent diameter = diameter of a circle with same area as the region
            diam = prop.equivalent_diameter
            diameters.append(diam)
            # Get boundary (as (row, col) pixel coords)
            boundary = measure.find_contours(label_img == prop.label, 0.5)
            # find_contours can return multiple contours, pick the longest (main boundary)
            if boundary:
                max_boundary = max(boundary, key=lambda arr: arr.shape[0])
                boundaries.append(max_boundary)
            else:
                boundaries.append(np.zeros((0,2)))

    if debug_visualize:
        plt.figure(figsize=(8,8))
        plt.imshow(image, cmap='gray')
        for b in boundaries:
            plt.plot(b[:,1], b[:,0], color='red', linewidth=2)
        plt.title(f"")
        plt.axis("off")
        plt.savefig("test.png")

    return diameters, boundaries

import nd2

# Replace with your ND2 file path
file_path = "./AL001c2_3_mwm_009.nd2"

img = nd2.imread(file_path)

# EXAMPLE USAGE:
# Assuming 'img' is your loaded 2D numpy array (float, grayscale)
diameters = process_image_and_get_diameters(
    img,
    block_size=201,
    min_blob_area=20
)
print(diameters)
