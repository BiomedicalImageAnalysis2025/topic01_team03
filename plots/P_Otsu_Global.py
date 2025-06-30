import os
from skimage.io import imread
from glob import glob
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import numpy as np
import sys

# --------------------------------------------------------------
def load_n2dh_gowt1_images(base_path="data-git/N2DH-GOWT1"):
    """
    Load the N2DH-GOWT1 image dataset and its corresponding ground-truth masks.

    Args:
        base_path (str): Base directory containing the "img" and "gt" subfolders.
                         Default is "data-git/N2DH-GOWT1".

    Returns:
        imgs_N2DH_GOWT1 (list[np.ndarray]): List of loaded grayscale images.
        gts_N2DH_GOWT1 (list[np.ndarray]): List of loaded grayscale ground-truth masks.
        img_paths_N2DH_GOWT1 (list[str]): List of file paths to the images.
        gt_paths_N2DH_GOWT1 (list[str]): List of file paths to the ground-truth masks.
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
    Load the N2DL-HeLa image dataset and its corresponding ground-truth masks.

    Args:
        base_path (str): Base directory containing the "img" and "gt" subfolders.
                         Default is "data-git/N2DL-HeLa".

    Returns:
        imgs_N2DL_HeLa (list[np.ndarray]): List of loaded grayscale images.
        gts_N2DL_HeLa (list[np.ndarray]): List of loaded grayscale ground-truth masks.
        img_paths_N2DL_HeLa (list[str]): List of file paths to the images.
        gt_paths_N2DL_HeLa (list[str]): List of file paths to the ground-truth masks.
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
    Load the NIH3T3 image dataset and its corresponding ground-truth masks.

    Args:
        base_path (str): Base directory containing the "img" and "gt" subfolders.
                         Default is "data-git/NIH3T3".

    Returns:
        imgs_NIH3T3 (list[np.ndarray]): List of loaded grayscale images.
        gts_NIH3T3 (list[np.ndarray]): List of loaded grayscale ground-truth masks.
        img_paths_NIH3T3 (list[str]): List of file paths to the images.
        gt_paths_NIH3T3 (list[str]): List of file paths to the ground-truth masks.
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

# import all needed funktions
#from imread_all import load_n2dh_gowt1_images, load_n2dl_hela_images, load_nih3t3_images
#from Dice_Score import dice_score
def dice_score(otsu_img: np.ndarray, otsu_gt: np.ndarray) -> float:
    """
    Berechnet den Dice-Koeffizienten zwischen zwei binären Bildern.

    Args:
        otsu_img: binäres Vorhersagebild (np.ndarray, dtype=bool)
        otsu_gt: binäres Ground-Truth-Bild (np.ndarray, dtype=bool)

    Returns:
        Dice Score als float (0.0 bis 1.0)
    """
    # control if the Pictures have the same Size
    if len(otsu_img) != len(otsu_gt):
        raise ValueError("Images don't have the same length!")

    # defining the variables for the Dice Score equation for TRUE FALSE images
    sum_img = np.sum(otsu_img)
    sum_gt = np.sum(otsu_gt)
    positive_overlap = np.sum(np.logical_and(otsu_img, otsu_gt))

    if sum_img + sum_gt == 0:
        return 1.0

    return 2 * positive_overlap / (sum_img + sum_gt)
# --------------------------------------------------------------

# --------------------------------------------------------------

def calculate_dice_scores_OP(imgs, gts):
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
dice_gowt1 = calculate_dice_scores_OP(imgs_N2DH_GOWT1, gts_N2DH_GOWT1)
dice_hela = calculate_dice_scores_OP(imgs_N2DL_HeLa, gts_N2DL_HeLa)
dice_nih = calculate_dice_scores_OP(imgs_NIH3T3, gts_NIH3T3)

# Convert numpy types to plain float
dice_gowt1 = [float(scores) for scores in dice_gowt1]
dice_hela = [float(scores) for scores in dice_hela]
dice_nih = [float(scores) for scores in dice_nih]

# Print all Dice-scores
print("GOWT1 Scores:", [f"{scores}" for scores in dice_gowt1])
print("HeLa Scores:", [f"{scores}" for scores in dice_hela])
print("NIH3T3 Scores:", [f"{scores}" for scores in dice_nih])