import numpy as np
import os
import sys

# Determine the script directory and project root for robust relative imports
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))  # one level above 'plots'

# Add the project root to the system path for consistent relative imports
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
