import os
import sys

# add project root
script_dir = os.getcwd()
project_root = os.path.abspath(script_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import importlib

import src.Dice_Score


importlib.reload(src.Dice_Score)
# Import the global Otsu implementation from the project source
from src.Complete_Otsu_Global import otsu_threshold_skimage_like
from src.Otsu_Local import local_otsu
from src.Dice_Score import dice_score
from skimage.filters import threshold_local

def calculate_dice_scores_global(imgs, gts):
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

def calculate_dice_scores_local_package(imgs, gts):
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