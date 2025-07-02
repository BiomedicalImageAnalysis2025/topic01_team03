import os
from skimage.filters import threshold_local
import sys

# for imports from src
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))  # eine Ebene über 'plots'

if project_root not in sys.path:
    sys.path.insert(0, project_root)


# imports from src
from src.imread_all import load_n2dh_gowt1_images, load_n2dl_hela_images, load_nih3t3_images
from src.Dice_Score import dice_score

# --------------------------------------------------------------------------
 
# Funktion aufrufen und Daten speichern
imgs_N2DH_GOWT1, gts_N2DH_GOWT1, img_paths_N2DH_GOWT1, gt_paths_N2DH_GOWT1 = load_n2dh_gowt1_images()

imgs_N2DL_HeLa, gts_N2DL_HeLa, img_paths_N2DL_HeLa, gt_paths_N2DL_HeLa = load_n2dl_hela_images()

imgs_NIH3T3, gts_NIH3T3, img_paths_NIH3T3, gt_paths_NIH3T3 = load_nih3t3_images()


# --------------------------------------------------------------

def calculate_O_dice_scores_OLP(imgs, gts):
    """
    Compute Dice-scores between binarized images and their corresponding ground-truth masks.

    Args:
        imgs (list[np.ndarray]): List of grayscale images to threshold.
        gts (list[np.ndarray]): List of ground-truth masks.

    Returns:
        scores (list[float]): List of Dice-scores for each image/ground-truth pair.
    """
    # Compute binary masks using local Otsu threshold
    otsu_imgs = [img > threshold_local(img, block_size=31, offset=0) for img in imgs]
    # Binarize ground-truths
    gt_binaries = [gt > 0 for gt in gts]

    # Compute Dice score for each image/ground-truth pair
    scores = []
    for i in range(min(len(otsu_imgs), len(gt_binaries))):
        score = dice_score(otsu_imgs[i], gt_binaries[i])
        scores.append(score)

    return scores

# --------------------------------------------------------------
# Compute Dice-scores for all datasets
dice_gowt1 = calculate_O_dice_scores_OLP(imgs_N2DH_GOWT1, gts_N2DH_GOWT1)
dice_hela = calculate_O_dice_scores_OLP(imgs_N2DL_HeLa, gts_N2DL_HeLa)
dice_nih = calculate_O_dice_scores_OLP(imgs_NIH3T3, gts_NIH3T3)

# Convert scores to regular Python floats
dice_gowt1 = [float(score) for score in dice_gowt1]
dice_hela = [float(score) for score in dice_hela]
dice_nih = [float(score) for score in dice_nih]

# Print all Dice-scores
print("GOWT1 Scores:", [f"{score}" for score in dice_gowt1])
print("HeLa Scores:", [f"{score}" for score in dice_hela])
print("NIH3T3 Scores:", [f"{score}" for score in dice_nih])