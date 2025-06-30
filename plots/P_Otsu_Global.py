import os
from skimage.io import imread
from glob import glob
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import numpy as np
import sys

# Dynamisch src-Verzeichnis zur Pfadsuche hinzufügen
current_dir = os.path.dirname(os.path.abspath(__file__))               # z. B. .../topic01_team03/plots
project_root = os.path.abspath(os.path.join(current_dir, ".."))       # → .../topic01_team03
src_dir = os.path.join(project_root, "src")                            # → .../topic01_team03/src

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Create output directory
output_dir = os.path.join(project_root, "output")
os.makedirs(output_dir, exist_ok=True)

# import all needed funktions
from imread_all import load_n2dh_gowt1_images, load_n2dl_hela_images, load_nih3t3_images
from Dice_Score import dice_score

# --------------------------------------------------------------

# Load N2DH-GOWT1 data
imgs_N2DH_GOWT1, gts_N2DH_GOWT1, img_paths_N2DH_GOWT1, gt_paths_N2DH_GOWT1 = load_n2dh_gowt1_images()

# Load N2DL-HeLa data
imgs_N2DL_HeLa, gts_N2DL_HeLa, img_paths_N2DL_HeLa, gt_paths_N2DL_HeLa = load_n2dl_hela_images()

# Load NIH3T3 data
imgs_NIH3T3, gts_NIH3T3, img_paths_NIH3T3, gt_paths_NIH3T3 = load_nih3t3_images()

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
