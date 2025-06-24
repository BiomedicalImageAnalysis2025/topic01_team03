import os
from skimage.io import imread
from glob import glob
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import numpy as np
import sys

# Set the current working directory as the project root
project_root = os.getcwd()
src_dir = os.path.join(project_root, "src")

# Add the src directory to the Python path if it's not already present
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Create an output directory if it doesn't already exist
output_dir = os.path.join(project_root, "output")
os.makedirs(output_dir, exist_ok=True)

# --------------------------------------------------------------
def load_n2dh_gowt1_images(base_path="data-git/N2DH-GOWT1"):
    """
    Load N2DH-GOWT1 image data and its corresponding ground-truth masks.

    Args:
        base_path (str): Base directory containing "img" and "gt" subdirectories.
                         Default is "data-git/N2DH-GOWT1".

    Returns:
        imgs_N2DH_GOWT1 (list[np.ndarray]): List of loaded grayscale image arrays.
        gts_N2DH_GOWT1 (list[np.ndarray]): List of loaded grayscale ground-truth arrays.
        img_paths_N2DH_GOWT1 (list[str]): List of file paths for each image.
        gt_paths_N2DH_GOWT1 (list[str]): List of file paths for each ground-truth image.
    """
    img_dir_N2DH_GOWT1 = os.path.join(base_path, "img")
    gt_dir_N2DH_GOWT1 = os.path.join(base_path, "gt")

    img_paths_N2DH_GOWT1 = sorted(glob(os.path.join(img_dir_N2DH_GOWT1, "*.tif")))
    gt_paths_N2DH_GOWT1 = sorted(glob(os.path.join(gt_dir_N2DH_GOWT1, "*.tif")))

    imgs_N2DH_GOWT1 = [imread(path, as_gray=True) for path in img_paths_N2DH_GOWT1]
    gts_N2DH_GOWT1 = [imread(path, as_gray=True) for path in gt_paths_N2DH_GOWT1]

    return imgs_N2DH_GOWT1, gts_N2DH_GOWT1, img_paths_N2DH_GOWT1, gt_paths_N2DH_GOWT1

# Load N2DH-GOWT1 data
imgs_N2DH_GOWT1, gts_N2DH_GOWT1, img_paths_N2DH_GOWT1, gt_paths_N2DH_GOWT1 = load_n2dh_gowt1_images()

# --------------------------------------------------------------
def load_n2dl_hela_images(base_path="data-git/N2DL-HeLa"):
    """
    Load N2DL-HeLa image data and its corresponding ground-truth masks.

    Args:
        base_path (str): Base directory containing "img" and "gt" subdirectories.
                         Default is "data-git/N2DL-HeLa".

    Returns:
        imgs_N2DL_HeLa (list[np.ndarray]): List of loaded grayscale image arrays.
        gts_N2DL_HeLa (list[np.ndarray]): List of loaded grayscale ground-truth arrays.
        img_paths_N2DL_HeLa (list[str]): List of file paths for each image.
        gt_paths_N2DL_HeLa (list[str]): List of file paths for each ground-truth image.
    """
    img_dir_N2DL_HeLa = os.path.join(base_path, "img")
    gt_dir_N2DL_HeLa = os.path.join(base_path, "gt")

    img_paths_N2DL_HeLa = sorted(glob(os.path.join(img_dir_N2DL_HeLa, "*.tif")))
    gt_paths_N2DL_HeLa = sorted(glob(os.path.join(gt_dir_N2DL_HeLa, "*.tif")))

    imgs_N2DL_HeLa = [imread(path, as_gray=True) for path in img_paths_N2DL_HeLa]
    gts_N2DL_HeLa = [imread(path, as_gray=True) for path in gt_paths_N2DL_HeLa]

    return imgs_N2DL_HeLa, gts_N2DL_HeLa, img_paths_N2DL_HeLa, gt_paths_N2DL_HeLa

# Load N2DL-HeLa data
imgs_N2DL_HeLa, gts_N2DL_HeLa, img_paths_N2DL_HeLa, gt_paths_N2DL_HeLa = load_n2dl_hela_images()

# --------------------------------------------------------------
def load_nih3t3_images(base_path="data-git/NIH3T3"):
    """
    Load NIH3T3 image data and its corresponding ground-truth masks.

    Args:
        base_path (str): Base directory containing "img" and "gt" subdirectories.
                         Default is "data-git/NIH3T3".

    Returns:
        imgs_NIH3T3 (list[np.ndarray]): List of loaded grayscale image arrays.
        gts_NIH3T3 (list[np.ndarray]): List of loaded grayscale ground-truth arrays.
        img_paths_NIH3T3 (list[str]): List of file paths for each image.
        gt_paths_NIH3T3 (list[str]): List of file paths for each ground-truth image.
    """
    img_dir_NIH3T3 = os.path.join(base_path, "img")
    gt_dir_NIH3T3 = os.path.join(base_path, "gt")

    img_paths_NIH3T3 = sorted(glob(os.path.join(img_dir_NIH3T3, "*.png")))
    gt_paths_NIH3T3 = sorted(glob(os.path.join(gt_dir_NIH3T3, "*.png")))

    imgs_NIH3T3 = [imread(path, as_gray=True) for path in img_paths_NIH3T3]
    gts_NIH3T3 = [imread(path, as_gray=True) for path in gt_paths_NIH3T3]

    return imgs_NIH3T3, gts_NIH3T3, img_paths_NIH3T3, gt_paths_NIH3T3

# Load NIH3T3 data
imgs_NIH3T3, gts_NIH3T3, img_paths_NIH3T3, gt_paths_NIH3T3 = load_nih3t3_images()

# --------------------------------------------------------------
from Dice_Score import dice_score

def berechne_dice_scores(imgs, gts):
    """
    Binarize the images using Otsu's threshold and compute the Dice-score
    against the corresponding ground-truth masks.

    Args:
        imgs (list[np.ndarray]): List of grayscale images.
        gts (list[np.ndarray]): List of ground-truth masks.

    Returns:
        scores (list[float]): List of Dice-scores for each image/ground-truth pair.
    """
    # Compute a binary version of each image using Otsu's threshold
    otsu_imgs = [img > threshold_otsu(img) for img in imgs]

    # Binarize the ground-truths (assuming they are already masks)
    gt_binaries = [gt > 0 for gt in gts]

    scores = []
    for i in range(min(len(otsu_imgs), len(gt_binaries))):
        score = dice_score(otsu_imgs[i], gt_binaries[i])
        scores.append(score)

    return scores

# --------------------------------------------------------------
# Compute the Dice-scores for all datasets
dice_gowt1 = berechne_dice_scores(imgs_N2DH_GOWT1, gts_N2DH_GOWT1)
dice_hela = berechne_dice_scores(imgs_N2DL_HeLa, gts_N2DL_HeLa)
dice_nih = berechne_dice_scores(imgs_NIH3T3, gts_NIH3T3)

# Convert numpy types to plain float
dice_gowt1 = [float(score) for score in dice_gowt1]
dice_hela = [float(score) for score in dice_hela]
dice_nih = [float(score) for score in dice_nih]

# Print all Dice-scores
print("GOWT1 Scores:", [f"{score}" for score in dice_gowt1])
print("HeLa Scores:", [f"{score}" for score in dice_hela])
print("NIH3T3 Scores:", [f"{score}" for score in dice_nih])
