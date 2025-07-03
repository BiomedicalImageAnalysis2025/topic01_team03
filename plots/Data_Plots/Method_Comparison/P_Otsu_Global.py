import os
import sys
from skimage.filters import threshold_otsu

# Add project root to sys.path for imports from src/
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))  # three levels above 'Method_Comparison'

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports from project-specific src/ directory
from src.imread_all import load_n2dh_gowt1_images, load_n2dl_hela_images, load_nih3t3_images
from src.Dice_Score import dice_score

# --------------------------------------------------------------------------
# Load images and ground-truth masks for all datasets
imgs_N2DH_GOWT1, gts_N2DH_GOWT1, img_paths_N2DH_GOWT1, gt_paths_N2DH_GOWT1 = load_n2dh_gowt1_images()
imgs_N2DL_HeLa, gts_N2DL_HeLa, img_paths_N2DL_HeLa, gt_paths_N2DL_HeLa = load_n2dl_hela_images()
imgs_NIH3T3, gts_NIH3T3, img_paths_NIH3T3, gt_paths_NIH3T3 = load_nih3t3_images()

# --------------------------------------------------------------------------
def calculate_otsu_package_dice_scores(imgs, gts):
    """
    Binarizes each image using skimage's global Otsu threshold and computes the Dice score
    against the corresponding ground-truth mask.

    Args:
        imgs (list[np.ndarray]): List of grayscale input images.
        gts (list[np.ndarray]): Corresponding ground-truth masks.

    Returns:
        list[float]: Dice scores for each image/ground-truth pair.
    """
    scores = []
    for img, gt in zip(imgs, gts):
        # Binarize image using global Otsu threshold
        otsu_img = img > threshold_otsu(img)

        # Ensure ground-truth is binary
        gt_binary = gt > 0

        # Compute Dice score
        score = dice_score(otsu_img, gt_binary)
        scores.append(score)
    return scores

# --------------------------------------------------------------------------
# Compute Dice scores for all datasets using skimage's Otsu threshold
dice_gowt1 = calculate_otsu_package_dice_scores(imgs_N2DH_GOWT1, gts_N2DH_GOWT1)
dice_hela = calculate_otsu_package_dice_scores(imgs_N2DL_HeLa, gts_N2DL_HeLa)
dice_nih = calculate_otsu_package_dice_scores(imgs_NIH3T3, gts_NIH3T3)

# Convert scores to regular Python floats
dice_gowt1 = [float(score) for score in dice_gowt1]
dice_hela = [float(score) for score in dice_hela]
dice_nih = [float(score) for score in dice_nih]

# Print Dice scores in a clear, formatted output
print("GOWT1 Scores =", ", ".join(f"{score}" for score in dice_gowt1))
print("HeLa Scores =", ", ".join(f"{score}" for score in dice_hela))
print("NIH3T3 Scores =", ", ".join(f"{score}" for score in dice_nih))