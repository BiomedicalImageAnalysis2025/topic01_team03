# All Dice Scores with our Otsu Global

import os
import sys

# add project root
script_dir = os.getcwd()
project_root = os.path.abspath(script_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Imports from project-specific src/ directory
from src.imread_all import load_n2dh_gowt1_images, load_n2dl_hela_images, load_nih3t3_images
from src.Dice_Score import dice_score
from src.Complete_Otsu_Global import otsu_threshold_skimage_like

# --------------------------------------------------------------------------
# Load images and ground-truth masks from the datasets
imgs_N2DH_GOWT1, gts_N2DH_GOWT1, img_paths_N2DH_GOWT1, gt_paths_N2DH_GOWT1 = load_n2dh_gowt1_images()
imgs_N2DL_HeLa, gts_N2DL_HeLa, img_paths_N2DL_HeLa, gt_paths_N2DL_HeLa = load_n2dl_hela_images()
imgs_NIH3T3, gts_NIH3T3, img_paths_NIH3T3, gt_paths_NIH3T3 = load_nih3t3_images()

# --------------------------------------------------------------------------
def calculate_dice_scores_OO(imgs, gts):
    """
    Calculates Dice scores between Otsu-binarized images and ground-truth masks.

    For each image, computes the global Otsu threshold, binarizes the image,
    and evaluates the Dice coefficient against the corresponding ground-truth.

    Args:
        imgs (list[np.ndarray]): List of grayscale input images.
        gts (list[np.ndarray]): Corresponding ground-truth binary masks.

    Returns:
        list[float]: Computed Dice scores.
    """
    scores = []
    for img, gt in zip(imgs, gts):
        # Compute global Otsu threshold on the input image
        t = otsu_threshold_skimage_like(img)

        # Apply threshold to binarize the input image
        otsu_img = img > t

        # Binarize the ground-truth (any positive value is foreground)
        gt_binary = gt > 0

        # Compute Dice score
        score = dice_score(otsu_img, gt_binary)
        scores.append(score)
    return scores

# --------------------------------------------------------------------------
# Compute Dice scores for each dataset
dice_gowt1 = calculate_dice_scores_OO(imgs_N2DH_GOWT1, gts_N2DH_GOWT1)
dice_hela = calculate_dice_scores_OO(imgs_N2DL_HeLa, gts_N2DL_HeLa)
dice_nih = calculate_dice_scores_OO(imgs_NIH3T3, gts_NIH3T3)

# Convert numpy floats to plain Python floats for clearer output
dice_gowt1 = [float(score) for score in dice_gowt1]
dice_hela = [float(score) for score in dice_hela]
dice_nih = [float(score) for score in dice_nih]

# Print scores in a readable format
print("GOWT1_Scores =", ", ".join(f"{score}" for score in dice_gowt1))
print("HeLa_Scores =", ", ".join(f"{score}" for score in dice_hela))
print("NIH3T3_Scores =", ", ".join(f"{score}" for score in dice_nih))

