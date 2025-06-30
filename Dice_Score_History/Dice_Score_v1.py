from skimage.io import imread
from skimage.filters import threshold_otsu  # otsu-global pakage
import time
import os


# skimage Otsu-Global
image = imread("data-git/N2DH-GOWT1/img/t01.tif", as_gray=True)
otsu_threshold = threshold_otsu(image)
otsu = (image > otsu_threshold).astype(int).flatten()  # binary & 1D

otsu_img = otsu   # output of otsu

ground_truth = imread("data-git/N2DH-GOWT1/gt/man_seg01.tif", as_gray=True)
otsu_gt = (ground_truth > 0).astype(int).flatten()  # gt binary & 1D

zeiten = []

for i in range(1000):
    start = time.perf_counter()

    def dice_score(otsu_img, otsu_gt):
    
        # control if the Pictures have the same Size
        if len(otsu_img) != len(otsu_gt):
            raise ValueError("Images don't have the same length!")

        # defining the variables for the Dice Score equation
        positive_overlap = 0
        sum_img = 0
        sum_gt = 0

        for t, p in zip(otsu_img, otsu_gt):
            if t == 1:
                sum_img += 1
            if p == 1:
                sum_gt += 1
            if t == 1 and p == 1:
                positive_overlap += 1

        if sum_img + sum_gt == 0:
            return 1.0

        return 2 * positive_overlap / (sum_img + sum_gt)

    #print("Dice Score:", dice_score(otsu_img, otsu_gt))

    ende = time.perf_counter()
    zeiten.append(ende - start)

mittelwert = sum(zeiten) / len(zeiten)

print(f"Dice_Score_v1.py Durchschnittliche Laufzeit: {mittelwert:.10f} Sekunden")
