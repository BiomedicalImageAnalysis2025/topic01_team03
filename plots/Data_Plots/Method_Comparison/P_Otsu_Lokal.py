import os
import sys
from skimage.filters import threshold_local

# add project root
script_dir = os.getcwd()
project_root = os.path.abspath(script_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Project-specific imports
from src.imread_all import load_n2dh_gowt1_images, load_n2dl_hela_images, load_nih3t3_images
from src.Dice_Score import dice_score

# --------------------------------------------------------------------------
# Load images and ground-truth masks for all datasets
imgs_N2DH_GOWT1, gts_N2DH_GOWT1, img_paths_N2DH_GOWT1, gt_paths_N2DH_GOWT1 = load_n2dh_gowt1_images()
imgs_N2DL_HeLa, gts_N2DL_HeLa, img_paths_N2DL_HeLa, gt_paths_N2DL_HeLa = load_n2dl_hela_images()
imgs_NIH3T3, gts_NIH3T3, img_paths_NIH3T3, gt_paths_NIH3T3 = load_nih3t3_images()

# --------------------------------------------------------------------------
def calculate_local_otsu_package_dice_scores(imgs, gts):
    """
    Computes Dice scores by binarizing each image with skimage's local threshold method
    (threshold_local) and comparing the result to its corresponding ground-truth mask.

    Args:
        imgs (list[np.ndarray]): List of grayscale input images.
        gts (list[np.ndarray]): Corresponding ground-truth masks.

    Returns:
        list[float]: Dice scores for each image/ground-truth pair.
    """
    # Apply local Otsu thresholding
    otsu_imgs = [img > threshold_local(img, block_size=31, offset=0) for img in imgs]
    
    # Convert ground-truth masks to binary
    gt_binaries = [gt > 0 for gt in gts]

    # Compute Dice scores
    scores = []
    for otsu_img, gt_binary in zip(otsu_imgs, gt_binaries):
        score = dice_score(otsu_img, gt_binary)
        scores.append(score)

    return scores

# --------------------------------------------------------------------------
# Compute Dice scores for all datasets
dice_gowt1 = calculate_local_otsu_package_dice_scores(imgs_N2DH_GOWT1, gts_N2DH_GOWT1)
dice_hela = calculate_local_otsu_package_dice_scores(imgs_N2DL_HeLa, gts_N2DL_HeLa)
dice_nih = calculate_local_otsu_package_dice_scores(imgs_NIH3T3, gts_NIH3T3)

# Convert to plain Python floats for consistent output
dice_gowt1 = [float(score) for score in dice_gowt1]
dice_hela = [float(score) for score in dice_hela]
dice_nih = [float(score) for score in dice_nih]

# Print Dice scores in a readable format
print("GOWT1_Scores =", ", ".join(f"{score}" for score in dice_gowt1))
print("HeLa_Scores =", ", ".join(f"{score}" for score in dice_hela))
print("NIH3T3_Scores =", ", ".join(f"{score}" for score in dice_nih))
