from skimage.io import imread
from skimage.filters import threshold_otsu
from medpy.metric import binary
import time



# --- Otsu-Maske erstellen ---
bild = imread("data-git/N2DH-GOWT1/img/t01.tif", as_gray=True)
otsu_threshold = threshold_otsu(bild)
otsu_mask = (bild > otsu_threshold).astype(bool)

# --- Ground Truth laden ---
ground_truth = imread("data-git/N2DH-GOWT1/gt/man_seg01.tif", as_gray=True)
gt_mask = (ground_truth > 0).astype(bool)


zeiten = []
score = []
for i in range(10000):
    start = time.perf_counter()

    # --- Dice Score berechnen ---
    dice_score = binary.dc(otsu_mask, gt_mask)

    #print("Dice Score:", dice_score)
    score.append(dice_score)
    ende = time.perf_counter()
    zeiten.append(ende - start)

mittelwert = sum(zeiten) / len(zeiten)
scoremittel = sum(score) / len(score)
print(f"package_Dice.py Durchschnittliche Laufzeit: {mittelwert:.10f} Sekunden")
print(scoremittel)