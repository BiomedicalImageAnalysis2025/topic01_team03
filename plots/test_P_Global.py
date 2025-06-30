import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

# Dynamisch src-Verzeichnis zur Pfadsuche hinzufügen
current_dir = os.path.dirname(os.path.abspath(__file__))               # z. B. .../plots
project_root = os.path.abspath(os.path.join(current_dir, ".."))       # → .../topic01_team03
src_dir = os.path.join(project_root, "src")                            # → .../topic01_team03/src

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Imports aus src
from imread_all import load_n2dh_gowt1_images, load_n2dl_hela_images, load_nih3t3_images
from Dice_Score import dice_score

# --------------------------------------------------------------
# Datensätze laden
imgs_N2DH_GOWT1, gts_N2DH_GOWT1, _, _ = load_n2dh_gowt1_images()
imgs_N2DL_HeLa, gts_N2DL_HeLa, _, _ = load_n2dl_hela_images()
imgs_NIH3T3, gts_NIH3T3, _, _ = load_nih3t3_images()

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
# Dice-Scores berechnen
dice_gowt1 = calculate_dice_scores_OP(imgs_N2DH_GOWT1, gts_N2DH_GOWT1)
dice_hela = calculate_dice_scores_OP(imgs_N2DL_HeLa, gts_N2DL_HeLa)
dice_nih = calculate_dice_scores_OP(imgs_NIH3T3, gts_NIH3T3)

# Ausgabe
print("GOWT1 Scores:", [f"{score:.3f}" for score in dice_gowt1])
print("HeLa Scores:", [f"{score:.3f}" for score in dice_hela])
print("NIH3T3 Scores:", [f"{score:.3f}" for score in dice_nih])
