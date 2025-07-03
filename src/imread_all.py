import os
from skimage.io import imread
from glob import glob
import sys

# Determine the directory of the current script and set the project root
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))  # one level above this script

# Add the project root to the system path for consistent imports across modules
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_n2dh_gowt1_images(base_path=None):
    """
    Load grayscale images and corresponding ground-truth masks from the N2DH-GOWT1 dataset.

    If no base_path is provided, a default path relative to the project root is used.
    The function reads all .tif files from the 'img' and 'gt' subdirectories of the dataset.

    This function ensures reproducibility by returning the list of images and masks in a
    consistently sorted order, matching each image with its corresponding ground truth.

    Args:
        base_path (str or None): Path to the dataset root directory. If None, the default
            path <project_root>/data-git/N2DH-GOWT1 is used.

    Returns:
        imgs (list of np.ndarray): List of grayscale input images.
        gts (list of np.ndarray): List of ground-truth masks.
        img_paths (list of str): File paths of the input images.
        gt_paths (list of str): File paths of the ground-truth masks.
    """
    if base_path is None:
        base_path = os.path.join(project_root, "data-git", "N2DH-GOWT1")

    img_dir = os.path.join(base_path, "img")
    gt_dir = os.path.join(base_path, "gt")

    img_paths = sorted(glob(os.path.join(img_dir, "*.tif")))
    gt_paths = sorted(glob(os.path.join(gt_dir, "*.tif")))

    imgs = [imread(path, as_gray=True) for path in img_paths]
    gts = [imread(path, as_gray=True) for path in gt_paths]

    return imgs, gts, img_paths, gt_paths


def load_n2dl_hela_images(base_path=None):
    """
    Load grayscale images and corresponding ground-truth masks from the N2DL-HeLa dataset.

    If no base_path is provided, a default path relative to the project root is used.
    The function reads all .tif files from the 'img' and 'gt' subdirectories of the dataset.

    This implementation guarantees consistent matching of images and ground truths by sorting
    the file lists before loading.

    Args:
        base_path (str or None): Path to the dataset root directory. If None, the default
            path <project_root>/data-git/N2DL-HeLa is used.

    Returns:
        imgs (list of np.ndarray): List of grayscale input images.
        gts (list of np.ndarray): List of ground-truth masks.
        img_paths (list of str): File paths of the input images.
        gt_paths (list of str): File paths of the ground-truth masks.
    """
    if base_path is None:
        base_path = os.path.join(project_root, "data-git", "N2DL-HeLa")

    img_dir = os.path.join(base_path, "img")
    gt_dir = os.path.join(base_path, "gt")

    img_paths = sorted(glob(os.path.join(img_dir, "*.tif")))
    gt_paths = sorted(glob(os.path.join(gt_dir, "*.tif")))

    imgs = [imread(path, as_gray=True) for path in img_paths]
    gts = [imread(path, as_gray=True) for path in gt_paths]

    return imgs, gts, img_paths, gt_paths


def load_nih3t3_images(base_path=None):
    """
    Load grayscale images and corresponding ground-truth masks from the NIH3T3 dataset.

    If no base_path is provided, a default path relative to the project root is used.
    The function reads all .png files from the 'img' and 'gt' subdirectories of the dataset.

    The matching of images and masks relies on consistent file naming and sorting to ensure
    correct pairing for subsequent analysis.

    Args:
        base_path (str or None): Path to the dataset root directory. If None, the default
            path <project_root>/data-git/NIH3T3 is used.

    Returns:
        imgs (list of np.ndarray): List of grayscale input images.
        gts (list of np.ndarray): List of ground-truth masks.
        img_paths (list of str): File paths of the input images.
        gt_paths (list of str): File paths of the ground-truth masks.
    """
    if base_path is None:
        base_path = os.path.join(project_root, "data-git", "NIH3T3")

    img_dir = os.path.join(base_path, "img")
    gt_dir = os.path.join(base_path, "gt")

    img_paths = sorted(glob(os.path.join(img_dir, "*.png")))
    gt_paths = sorted(glob(os.path.join(gt_dir, "*.png")))

    imgs = [imread(path, as_gray=True) for path in img_paths]
    gts = [imread(path, as_gray=True) for path in gt_paths]

    return imgs, gts, img_paths, gt_paths
