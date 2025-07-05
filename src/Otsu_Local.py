import numpy as np
import os
import sys

# add project root
script_dir = os.getcwd()
project_root = os.path.abspath(script_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the global Otsu implementation from the project source
from src.Complete_Otsu_Global import otsu_threshold_skimage_like

def local_otsu(image: np.ndarray, radius: int = 15) -> np.ndarray:
    """
    Compute a local Otsu threshold map for a 2D grayscale image by calculating a separate Otsu threshold
    for each pixel within a square window centered on that pixel.

    This implementation preserves the original intensity range of the input and mimics the behavior
    of skimage.filters.threshold_local, but replaces the standard mean or median-based thresholds
    with Otsu's method, which maximizes inter-class variance locally.

    For each pixel (i, j), the method:
        1. Extracts a square neighborhood of size (2*radius + 1) around the pixel, with reflection
           padding at the borders to avoid artifacts.
        2. Computes the Otsu threshold on the local window.
        3. Assigns the computed threshold to the corresponding position in the output threshold map.

    Args:
        image (np.ndarray): Input 2D grayscale image, either floating-point or integer.
        radius (int): Radius of the local window; the window size is (2*radius + 1) x (2*radius + 1).

    Returns:
        t_map (np.ndarray): 2D array with the local threshold at each pixel, having the same dtype as the input image.
    """
    H, W = image.shape
    t_map = np.zeros((H, W), dtype=image.dtype)

    # Pad the image using reflection to handle edge regions gracefully
    pad = radius
    padded = np.pad(image, pad, mode="reflect")
    w = 2 * radius + 1

    # Iterate over each pixel and calculate the local Otsu threshold
    for i in range(H):
        if i % 50 == 0 or i == H - 1:
            print(f"Processing row {i+1}/{H}...")
        for j in range(W):
            block = padded[i : i + w, j : j + w]
            t = otsu_threshold_skimage_like(block)  # compute threshold directly on original intensity values
            t_map[i, j] = t

    return t_map

from skimage import util
from skimage.filters import rank, threshold_otsu
from skimage.morphology import footprint_rectangle

# Defining a quicker otsu local for less computational load
def local_otsu_fast(image: np.ndarray, radius: int = 15) -> np.ndarray:
    """
    Efficient local Otsu threshold map.

    Args:
        image: 2D grayscale array (float or int).
        radius: local neighborhood radius.

    Returns:
        threshold_map: same shape and dtype as input, per-pixel Otsu thresholds.
    """
    # Must be uint8 for rank filters
    img_u8 = util.img_as_ubyte(image)

    # Define local neighborhood
    selem = footprint_rectangle((2*radius+1, 2*radius+1))

    # Compute local thresholds (C-optimized)
    local_thresh = rank.otsu(img_u8, selem)

    # Convert thresholds back to same dtype as input
    return local_thresh.astype(image.dtype)
