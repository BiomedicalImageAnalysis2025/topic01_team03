from skimage.filters import threshold_multiotsu
from skimage import img_as_ubyte
import numpy as np
from skimage.morphology import remove_small_objects
import sys
import os

# add project root
script_dir = os.getcwd()
project_root = os.path.abspath(script_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.Dice_Score import dice_score


# ------------------- Multi-Otsu-basierte Maske -------------------

def apply_multiotsu_mask_class1(img):
    """
    Nutzt Multi-Otsu und weist nur Klasse 1 (zwischen den ersten beiden Schwellen)
    als Vordergrund zu. Klasse 0 (dunkel) und Klassen >=2 (heller) werden Hintergrund.
    """
    img_u8 = img_as_ubyte(img)
    thresholds = threshold_multiotsu(img_u8, classes=3)
    regions = np.digitize(img_u8, bins=thresholds)
    
    # Nur Klasse 1 wird Vordergrund
    mask = (regions == 1)
    return mask

# ------------------- Dice-Scores berechnen -------------------
def calculate_multiotsu_dice_scores_class1(imgs, gts):
    scores = []
    for img, gt in zip(imgs, gts):
        mask = apply_multiotsu_mask_class1(img)
        gt_binary = gt > 0
        scores.append(dice_score(mask, gt_binary))
    return scores



def apply_multiotsu_mask_class1_cleaned(img):
    img_u8 = img_as_ubyte(img)
    thresholds = threshold_multiotsu(img_u8, classes=3)
    regions = np.digitize(img_u8, bins=thresholds)
    mask = (regions == 1)

    # Entferne sehr helle Pixel (>230), selbst wenn sie in Klasse 1 gelandet sind
    mask[img_u8 > 230] = 0
    return mask

def calculate_multiotsu_dice_scores_class1_cleaned(imgs, gts):
    scores = []
    for img, gt in zip(imgs, gts):
        mask = apply_multiotsu_mask_class1_cleaned(img)
        gt_binary = gt > 0
        scores.append(dice_score(mask, gt_binary))
    return scores


def apply_multiotsu_mask_class1_cleaned_remove(img):
    img_u8 = img_as_ubyte(img)
    thresholds = threshold_multiotsu(img_u8, classes=3)
    regions = np.digitize(img_u8, bins=thresholds)
    mask = (regions == 1)
    
    # Entferne sehr helle Pixel (>230)
    mask[img_u8 > 230] = 0

    # Optional: Kleine Artefakte entfernen (z.B. unter 50 Pixel)
    mask = remove_small_objects(mask, min_size=50)
    return mask

def calculate_multiotsu_dice_scores_class1_cleaned_remove(imgs, gts):
    scores = []
    for img, gt in zip(imgs, gts):
        mask = apply_multiotsu_mask_class1_cleaned_remove(img)
        gt_binary = gt > 0
        scores.append(dice_score(mask, gt_binary))
    return scores