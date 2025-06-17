from skimage.io import imread
from skimage.filters import threshold_otsu
from medpy.metric import binary
import time

start = time.time()

# --- Otsu-Maske erstellen ---
bild = imread("data-git/N2DH-GOWT1/img/t01.tif", as_gray=True)
otsu_threshold = threshold_otsu(bild)
otsu_mask = (bild > otsu_threshold).astype(bool)

# --- Ground Truth laden ---
ground_truth = imread("data-git/N2DH-GOWT1/gt/man_seg01.tif", as_gray=True)
gt_mask = (ground_truth > 0).astype(bool)

# --- Dice Score berechnen ---
dice_score = binary.dc(otsu_mask, gt_mask)

print("Dice Score:", dice_score)

end = time.time()

print(f"Laufzeit: {end - start:.4f} Sekunden")