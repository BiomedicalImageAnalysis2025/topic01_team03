# src/finde_image.py

import os
from pathlib import Path
from skimage import io
import numpy as np

print("Working Directory:", os.getcwd())

def find_image_in_data_git(filename: str, 
                           data_root: str = "data-git") -> Path:
    """
    Searches the 'data-git' directory and its three expected subdirectories
    ('N2DH-GOWT1', 'N2DL-HeLa', 'NIH3T3') for an image file with the specified
    `filename`. The function searches exclusively in the 'img/' folder within each
    subdirectory.

    Args:
        filename (str): Name of the image file to locate (e.g., 't01.tif').
        data_root (str, optional): Path to the root data directory (default: "data-git").

    Returns:
        Path: Absolute path to the found image file.

    Raises:
        FileNotFoundError: If the file is not found in any of the expected
                           subdirectory locations.
    """
    root = Path(data_root)
    # List of the expected subdirectories
    subdirs = ["N2DH-GOWT1", "N2DL-HeLa", "NIH3T3"]

    for sd in subdirs:
        img_dir = root / sd / "img"
        if not img_dir.is_dir():
            continue
        # Search only in img/ of the current subdirectory
        candidate = img_dir / filename
        if candidate.exists():
            return candidate.resolve()
    
    raise FileNotFoundError(
        f"File '{filename}' not found in any 'img/' folder within '{data_root}'."
    )


def load_image_gray(filename: str,
                    data_root: str = "data-git",
                    as_gray: bool = True) -> np.ndarray:
    """
    Locates an image file within the data directory structure using
    `find_image_in_data_git`, and loads it as a grayscale image (or in color
    if `as_gray=False`). Returns the image as a NumPy array.

    This utility is intended to simplify loading of dataset images for
    segmentation evaluation.

    Args:
        filename (str): Name of the image file to load.
        data_root (str, optional): Path to the root data directory (default: "data-git").
        as_gray (bool, optional): If True, loads the image as a single-channel grayscale
                                  array. If False, loads in original color channels
                                  (default: True).

    Returns:
        np.ndarray: Loaded image as a NumPy array, in grayscale or color.
    """
    img_path = find_image_in_data_git(filename, data_root)
    return io.imread(str(img_path), as_gray=as_gray)
