# does not work yet
import os
import sys
from medpy.metric import binary

# for imports from src
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))  # eine Ebene über 'plots'

if project_root not in sys.path:
    sys.path.insert(0, project_root)


# imports from srd
from src.imread_all import load_n2dh_gowt1_images, load_n2dl_hela_images, load_nih3t3_images
from src.Complete_Otsu_Global import custom_histogram, otsu_threshold_skimage_like

# --------------------------------------------------------------------------
 
# Funktion aufrufen und Daten speichern
imgs_N2DH_GOWT1, gts_N2DH_GOWT1, img_paths_N2DH_GOWT1, gt_paths_N2DH_GOWT1 = load_n2dh_gowt1_images()

imgs_N2DL_HeLa, gts_N2DL_HeLa, img_paths_N2DL_HeLa, gt_paths_N2DL_HeLa = load_n2dl_hela_images()

imgs_NIH3T3, gts_NIH3T3, img_paths_NIH3T3, gt_paths_NIH3T3 = load_nih3t3_images()

# --------------------------------------------------------------
def calculate_P_dice_scores(imgs, gts):
    """
    Berechnet die Dice-Scores zwischen Otsu-binarisierten Bildern und den Ground-Truth-Masken.

    Args:
        imgs (list[np.ndarray]): Die Eingabebilder.
        gts (list[np.ndarray]): Die zugehörigen Ground-Truth-Bilder.

    Returns:
        list[float]: Die berechneten Dice-Scores.
    """
    scores = []
    for img, gt in zip(imgs, gts):
        
        # Otsu-Schwelle berechnen
        t = otsu_threshold_skimage_like(img)

        # Bild binarisieren
        otsu_img = img > t

        # GT binarisieren
        gt_binary = gt > 0

        # Dice-Score berechnen
        score = binary.dc(otsu_img, gt_binary)
        scores.append(score)
    return scores

# --------------------------------------------------------------
# Berechne die Dice-Scores für die drei Datensätze
dice_gowt1 = calculate_P_dice_scores(imgs_N2DH_GOWT1, gts_N2DH_GOWT1)
dice_hela = calculate_P_dice_scores(imgs_N2DL_HeLa, gts_N2DL_HeLa)
dice_nih = calculate_P_dice_scores(imgs_NIH3T3, gts_NIH3T3)

# Als einfache Floats statt np.float64
dice_gowt1 = [float(score) for score in dice_gowt1]
dice_hela = [float(score) for score in dice_hela]
dice_nih = [float(score) for score in dice_nih]

# Schön formatiert ausgeben
print("GOWT1 Scores:", [f"{score}" for score in dice_gowt1])
print("HeLa Scores:", [f"{score}" for score in dice_hela])
print("NIH3T3 Scores:", [f"{score}" for score in dice_nih])