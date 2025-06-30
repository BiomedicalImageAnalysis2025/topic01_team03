import numpy as np
from skimage.io import imread
from skimage.filters import threshold_otsu  # otsu-global pakage
import time

start = time.time()

# skimage Otsu-Global
image = imread("data-git/N2DH-GOWT1/img/t01.tif", as_gray=True)
otsu_threshold = threshold_otsu(image)
otsu = (image > otsu_threshold).astype(int).flatten()  # binary & 1D

otsu_img = otsu   # output of otsu

ground_truth = imread("data-git/N2DH-GOWT1/gt/man_seg01.tif", as_gray=True)
otsu_gt = (ground_truth > 0).astype(int).flatten()  # gt binary & 1D

zeiten = []

for _ in range(10000):
    start = time.perf_counter()

    def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
        """
        Berechnet den Dice-Koeffizienten zwischen zwei binären Bildern (dtype=bool).
        """
        if pred.shape != target.shape:
            raise ValueError("Die Eingabebilder haben unterschiedliche Formen.")

        intersection = np.logical_and(pred, target).sum()
        total = pred.sum() + target.sum()

        if total == 0:
            return 1.0  # Sonderfall: beide leer → perfekte Übereinstimmung

        return 2 * intersection / total

    #print("Dice Score:", dice_score(otsu_img, otsu_gt))

    ende = time.perf_counter()
    zeiten.append(ende - start)

mittelwert = sum(zeiten) / len(zeiten)

print(f"Dice_Score_v4.py Durchschnittliche Laufzeit: {mittelwert:.10f} Sekunden")