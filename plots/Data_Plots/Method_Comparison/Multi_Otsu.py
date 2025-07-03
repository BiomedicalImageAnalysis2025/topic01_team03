import numpy as np
import os
import sys
from skimage.filters import threshold_multiotsu
from skimage import img_as_ubyte

# Project root hinzufügen
script_dir = os.getcwd()
project_root = os.path.abspath(script_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.imread_all import load_nih3t3_images
from src.Dice_Score import dice_score

# ------------------- Multi-Otsu-basierte Maske -------------------

def apply_multiotsu_mask_class1_foreground(img):
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
def calculate_multiotsu_dice_scores(imgs, gts):
    scores = []
    for img, gt in zip(imgs, gts):
        mask = apply_multiotsu_mask_class1_foreground(img)
        gt_binary = gt > 0
        scores.append(dice_score(mask, gt_binary))
    return scores

# ------------------- Bilder laden -------------------

imgs_NIH3T3, gts_NIH3T3, *_ = load_nih3t3_images()

# ------------------- Dice-Scores berechnen -------------------

dice_nih   = calculate_multiotsu_dice_scores(imgs_NIH3T3, gts_NIH3T3)

# Als normale Floats ausgeben
dice_nih   = [float(score) for score in dice_nih]

# Ergebnisse ausgeben

print("NIH3T3_Scores =", ", ".join(f"{score}" for score in dice_nih))
