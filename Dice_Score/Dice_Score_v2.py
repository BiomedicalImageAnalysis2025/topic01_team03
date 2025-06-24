import numpy as np
from skimage.io import imread
from skimage.filters import threshold_otsu  # otsu-global pakage
import time

start = time.time()

# skimage Otsu-Global
image = imread("data-git/N2DH-GOWT1/img/t01.tif", as_gray=True)
otsu_threshold = threshold_otsu(image)
otsu = image > otsu_threshold  # TRUE FALSE Picture

otsu_img = otsu   # output of otsu

ground_truth = imread("data-git/N2DH-GOWT1/gt/man_seg01.tif", as_gray=True)
otsu_gt = ground_truth > 0  # TRUE FALSE images


def dice_score(otsu_img, otsu_gt):

    # control if the Pictures have the same Size
    if len(otsu_img) != len(otsu_gt):
        raise ValueError("Images don't have the same length!")

    # defining the variables for the Dice Score equation for TRUE FALSE images
    sum_img = np.sum(otsu_img)
    sum_gt = np.sum(otsu_gt)
    positive_overlap = np.sum(np.logical_and(otsu_img, otsu_gt))

    if sum_img + sum_gt == 0:
        return 1.0

    return 2 * positive_overlap / (sum_img + sum_gt)

print("Dice Score:", dice_score(otsu_img, otsu_gt))

end = time.time()

print(f"Laufzeit: {end - start:.4f} Sekunden")