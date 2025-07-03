import os
import sys
import numpy as np

# add project root
script_dir = os.getcwd()
project_root = os.path.abspath(script_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports from project-specific src/ directory
from src.imread_all import load_n2dh_gowt1_images, load_n2dl_hela_images, load_nih3t3_images
from src.Dice_Score import dice_score
from src.Otsu_Local import local_otsu

# --------------------------------------------------------------------------
# Load images and ground-truth masks for all dataset
imgs_N2DH_GOWT1, gts_N2DH_GOWT1, img_paths_N2DH_GOWT1, gt_paths_N2DH_GOWT1 = load_n2dh_gowt1_images()

imgs_N2DL_HeLa, gts_N2DL_HeLa, img_paths_N2DL_HeLa, gt_paths_N2DL_HeLa = load_n2dl_hela_images()

imgs_NIH3T3, gts_NIH3T3, img_paths_NIH3T3, gt_paths_NIH3T3 = load_nih3t3_images()
# --------------------------------------------------------------------------

def calculate_dice_scores_local(imgs, gts):
    """
    Computes Dice scores for a list of images using local Otsu thresholding.

    Args:
        imgs (list[np.ndarray]): List of grayscale input images.
        gts (list[np.ndarray]): Corresponding ground-truth binary masks.

    Returns:
        list[float]: Computed Dice scores.
    """
    scores = []
    for img, gt in zip(imgs, gts):
        # Compute local Otsu threshold map
        t_map = local_otsu(img, radius=15)
        # Apply local threshold
        mask = img > t_map
        # Binarize ground-truth mask
        gt_binary = gt > 0
        # Calculate Dice score
        score = dice_score(mask, gt_binary)
        scores.append(score)
    return scores

# --------------------------------------------------------------------------
# Compute Dice scores for all datasets
dice_gowt1 = calculate_dice_scores_local(imgs_N2DH_GOWT1, gts_N2DH_GOWT1)
dice_hela = calculate_dice_scores_local(imgs_N2DL_HeLa, gts_N2DL_HeLa)
dice_nih = calculate_dice_scores_local(imgs_NIH3T3, gts_NIH3T3)


# Convert scores to regular Python floats
dice_gowt1 = [float(score) for score in dice_gowt1]
dice_hela = [float(score) for score in dice_hela]
dice_nih = [float(score) for score in dice_nih]


# Print Dice scores in a clear format
print("GOWT1_Scores =", ", ".join(f"{score}" for score in dice_gowt1))
print("HeLa_Scores =", ", ".join(f"{score}" for score in dice_hela))
print("NIH3T3_Scores =", ", ".join(f"{score}" for score in dice_nih))

