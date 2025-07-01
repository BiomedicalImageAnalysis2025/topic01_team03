# does not work yet
import os
import sys

# for imports from src
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))  # eine Ebene über 'plots'

if project_root not in sys.path:
    sys.path.insert(0, project_root)


# imports from src
from src.imread_all import load_n2dh_gowt1_images, load_n2dl_hela_images, load_nih3t3_images
from src.Dice_Score import dice_score
from src.Complete_Otsu_Global import otsu_threshold_skimage_like

import numpy as np

def local_otsu(image: np.ndarray, radius: int = 15) -> np.ndarray:
    """
    Computes a local Otsu threshold map of an image: for each pixel, calculates the Otsu threshold
    in a square window around that pixel. Works in the original image value range,
    similar to skimage.filters.threshold_local (but with Otsu instead of mean/median).
    
    Args:
        image (np.ndarray): 2D grayscale image (float or integer).
        radius (int): Radius of the local window (window size is (2*radius + 1)^2).
    
    Returns:
        t_map (np.ndarray): 2D array with the local threshold at each pixel (same dtype as input).
    """
    H, W = image.shape
    t_map = np.zeros((H, W), dtype=image.dtype)

    pad = radius
    padded = np.pad(image, pad, mode="reflect")
    w = 2 * radius + 1

    for i in range(H):
        # Fortschrittsanzeige alle 50 Zeilen
        if i % 50 == 0:
            print(f"Processing row {i}/{H}")
        for j in range(W):
            block = padded[i : i + w, j : j + w]
            t = otsu_threshold_skimage_like(block)  # directly on original values
            t_map[i, j] = t

    return t_map

# --------------------------------------------------------------------------
 
# Funktion aufrufen und Daten speichern
#imgs_N2DH_GOWT1, gts_N2DH_GOWT1, img_paths_N2DH_GOWT1, gt_paths_N2DH_GOWT1 = load_n2dh_gowt1_images()

#imgs_N2DL_HeLa, gts_N2DL_HeLa, img_paths_N2DL_HeLa, gt_paths_N2DL_HeLa = load_n2dl_hela_images()

imgs_NIH3T3, gts_NIH3T3, img_paths_NIH3T3, gt_paths_NIH3T3 = load_nih3t3_images()
#from skimage.io import imread

#imgs1 = imread("data-git/N2DH-GOWT1/img/t01.tif", as_gray=True)
#gts1 = imread("data-git/N2DH-GOWT1/gt/man_seg01.tif", as_gray=True)

#imgs = [imgs1]
#gts = [gts1]

def calculate_dice_scores_local(imgs, gts):
    scores = []
    for img, gt in zip(imgs, gts):
        t_map = local_otsu(img, radius=15)
        mask = img > t_map
        gt_binary = gt > 0
        scores.append(dice_score(mask, gt_binary))
    return scores



# --------------------------------------------------------------
# Compute Dice-scores for all datasets
#dice_gowt1 = calculate_dice_scores_local(imgs_N2DH_GOWT1, gts_N2DH_GOWT1)
#dice_hela = calculate_dice_scores_local(imgs_N2DL_HeLa, gts_N2DL_HeLa)
dice_nih = calculate_dice_scores_local(imgs_NIH3T3, gts_NIH3T3)

#dice = calculate_dice_scores_local(imgs, gts)

# Convert scores to regular Python floats
#dice_gowt1 = [float(score) for score in dice_gowt1]
#dice_hela = [float(score) for score in dice_hela]
dice_nih = [float(score) for score in dice_nih]

#dice = [float(score) for score in dice]

# Print all Dice-scores
#print("GOWT1 Scores:", [f"{score}" for score in dice_gowt1])
#print("HeLa Scores:", [f"{score}" for score in dice_hela])
print("NIH3T3 Scores:", [f"{score}" for score in dice_nih])

#print("Scores:", [f"{score}" for score in dice])