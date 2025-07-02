import os
import sys
from medpy.metric import binary

# Add project root to sys.path for imports from src/
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))  # three levels above 'Method_Comparison'

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports from project-specific src/ directory
from src.imread_all import load_n2dh_gowt1_images, load_n2dl_hela_images, load_nih3t3_images
from src.Complete_Otsu_Global import otsu_threshold_skimage_like

# --------------------------------------------------------------------------
# Load images and ground-truth masks for all datasets
imgs_N2DH_GOWT1, gts_N2DH_GOWT1, img_paths_N2DH_GOWT1, gt_paths_N2DH_GOWT1 = load_n2dh_gowt1_images()
imgs_N2DL_HeLa, gts_N2DL_HeLa, img_paths_N2DL_HeLa, gt_paths_N2DL_HeLa = load_n2dl_hela_images()
imgs_NIH3T3, gts_NIH3T3, img_paths_NIH3T3, gt_paths_NIH3T3 = load_nih3t3_images()

# --------------------------------------------------------------------------
def calculate_package_dice_scores(imgs, gts):
    """
    Calculates Dice scores between Otsu-thresholded images and ground-truth masks using
    the medpy.metric Dice implementation (package Dice Score).

    Args:
        imgs (list[np.ndarray]): List of input grayscale images.
        gts (list[np.ndarray]): Corresponding ground-truth binary masks.

    Returns:
        list[float]: Calculated Dice scores.
    """
    scores = []
    for img, gt in zip(imgs, gts):
        # Compute Otsu threshold
        t = otsu_threshold_skimage_like(img)

        # Binarize the input image with the computed threshold
        otsu_img = img > t

        # Binarize the ground-truth mask
        gt_binary = gt > 0

        # Compute Dice score using medpy's implementation
        score = binary.dc(otsu_img, gt_binary)
        scores.append(score)
    return scores

# --------------------------------------------------------------------------
# Compute Dice scores for all three datasets
dice_gowt1 = calculate_package_dice_scores(imgs_N2DH_GOWT1, gts_N2DH_GOWT1)
dice_hela = calculate_package_dice_scores(imgs_N2DL_HeLa, gts_N2DL_HeLa)
dice_nih = calculate_package_dice_scores(imgs_NIH3T3, gts_NIH3T3)

# Convert scores to regular Python floats
dice_gowt1 = [float(score) for score in dice_gowt1]
dice_hela = [float(score) for score in dice_hela]
dice_nih = [float(score) for score in dice_nih]

# Print Dice scores in a clear format
print("GOWT1 Scores:", [f"{score:.6f}" for score in dice_gowt1])
print("HeLa Scores:", [f"{score:.6f}" for score in dice_hela])
print("NIH3T3 Scores:", [f"{score:.6f}" for score in dice_nih])