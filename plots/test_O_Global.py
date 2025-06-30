import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

# Dynamisch src-Verzeichnis zur Pfadsuche hinzufügen
current_dir = os.path.dirname(os.path.abspath(__file__))               # z. B. .../topic01_team03/plots
project_root = os.path.abspath(os.path.join(current_dir, ".."))       # → .../topic01_team03
src_dir = os.path.join(project_root, "src")                            # → .../topic01_team03/src

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Create output directory
output_dir = os.path.join(project_root, "output")
os.makedirs(output_dir, exist_ok=True)

# Import src modules
from imread_all import load_n2dh_gowt1_images, load_n2dl_hela_images, load_nih3t3_images
from Dice_Score import dice_score

# Load datasets
imgs_N2DH_GOWT1, gts_N2DH_GOWT1, _, _ = load_n2dh_gowt1_images()
imgs_N2DL_HeLa, gts_N2DL_HeLa, _, _ = load_n2dl_hela_images()
imgs_NIH3T3, gts_NIH3T3, _, _ = load_nih3t3_images()

# Dice Score calculation using Otsu
def calculate_dice_scores_OP(imgs, gts):
    """
    Binarize images using Otsu and calculate Dice scores vs. ground truth.
    """
    otsu_imgs = [img > threshold_otsu(img) for img in imgs]
    gt_binaries = [gt > 0 for gt in gts]
    return [float(dice_score(otsu, gt)) for otsu, gt in zip(otsu_imgs, gt_binaries)]

# Run Dice calculations
dice_gowt1 = calculate_dice_scores_OP(imgs_N2DH_GOWT1, gts_N2DH_GOWT1)
dice_hela = calculate_dice_scores_OP(imgs_N2DL_HeLa, gts_N2DL_HeLa)
dice_nih = calculate_dice_scores_OP(imgs_NIH3T3, gts_NIH3T3)

# Output results
print("GOWT1 Scores:", [f"{score:.3f}" for score in dice_gowt1])
print("HeLa Scores:", [f"{score:.3f}" for score in dice_hela])
print("NIH3T3 Scores:", [f"{score:.3f}" for score in dice_nih])
