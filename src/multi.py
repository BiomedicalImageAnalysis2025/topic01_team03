from skimage.filters import threshold_multiotsu
from skimage import img_as_ubyte
import numpy as np

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