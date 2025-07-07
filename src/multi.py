from skimage.filters import threshold_multiotsu
from skimage import img_as_ubyte
import numpy as np
from skimage.morphology import remove_small_objects
import sys
import os

# Add the current folder to the system path so local modules can be imported
script_dir = os.getcwd()
project_root = os.path.abspath(script_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.Dice_Score import dice_score

# ------------------- Multi-Otsu mask using class 1 -------------------

def apply_multiotsu_mask_class1(img):
    """
    Applies Multi-Otsu thresholding and uses only class 1 (middle intensity)
    as foreground. Class 0 (dark) and class 2+ (bright) are treated as background.
    """
    img_u8 = img_as_ubyte(img)
    thresholds = threshold_multiotsu(img_u8, classes=3)
    regions = np.digitize(img_u8, bins=thresholds)

    # Keep only class 1 as mask
    mask = (regions == 1)
    return mask

# ------------------- Calculate Dice scores -------------------

def calculate_multiotsu_dice_scores_class1(imgs, gts):
    """
    Calculates Dice scores using the basic Multi-Otsu mask (class 1 only).
    """
    scores = []
    for img, gt in zip(imgs, gts):
        mask = apply_multiotsu_mask_class1(img)
        gt_binary = gt > 0
        scores.append(dice_score(mask, gt_binary))
    return scores

# ------------------- Multi-Otsu with simple cleaning -------------------

def apply_multiotsu_mask_class1_cleaned(img):
    """
    Same as before, but removes very bright pixels (intensity > 230)
    even if they fall into class 1.
    """
    img_u8 = img_as_ubyte(img)
    thresholds = threshold_multiotsu(img_u8, classes=3)
    regions = np.digitize(img_u8, bins=thresholds)
    mask = (regions == 1)

    # Remove bright pixels from the mask
    mask[img_u8 > 230] = 0
    return mask

def calculate_multiotsu_dice_scores_class1_cleaned(imgs, gts):
    """
    Calculates Dice scores using cleaned Multi-Otsu masks
    (class 1 only, very bright pixels removed).
    """
    scores = []
    for img, gt in zip(imgs, gts):
        mask = apply_multiotsu_mask_class1_cleaned(img)
        gt_binary = gt > 0
        scores.append(dice_score(mask, gt_binary))
    return scores

# ------------------- Multi-Otsu with cleaning + small object removal -------------------

def apply_multiotsu_mask_class1_cleaned_remove(img):
    """
    Same as cleaned version, but also removes small objects from the mask
    (e.g. noise or artifacts smaller than 50 pixels).
    """
    img_u8 = img_as_ubyte(img)
    thresholds = threshold_multiotsu(img_u8, classes=3)
    regions = np.digitize(img_u8, bins=thresholds)
    mask = (regions == 1)

    # Remove bright pixels
    mask[img_u8 > 230] = 0

    # Remove small objects (noise)
    mask = remove_small_objects(mask, min_size=50)
    return mask

def calculate_multiotsu_dice_scores_class1_cleaned_remove(imgs, gts):
    """
    Calculates Dice scores using the cleaned Multi-Otsu mask with small
    object removal.
    """
    scores = []
    for img, gt in zip(imgs, gts):
        mask = apply_multiotsu_mask_class1_cleaned_remove(img)
        gt_binary = gt > 0
        scores.append(dice_score(mask, gt_binary))
    return scores