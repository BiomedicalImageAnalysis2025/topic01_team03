import os
from skimage.io import imread
from glob import glob
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import numpy as np

def load_n2dh_gowt1_images(base_path="data-git/N2DH-GOWT1"):
    img_dir_N2DH_GOWT1 = os.path.join(base_path, "img")
    gt_dir_N2DH_GOWT1 = os.path.join(base_path, "gt")

    img_paths_N2DH_GOWT1 = sorted(glob(os.path.join(img_dir_N2DH_GOWT1, "*.tif")))
    gt_paths_N2DH_GOWT1 = sorted(glob(os.path.join(gt_dir_N2DH_GOWT1, "*.tif")))

    imgs_N2DH_GOWT1 = [imread(path, as_gray=True) for path in img_paths_N2DH_GOWT1]
    gts_N2DH_GOWT1 = [imread(path, as_gray=True) for path in gt_paths_N2DH_GOWT1]

    return imgs_N2DH_GOWT1, gts_N2DH_GOWT1, img_paths_N2DH_GOWT1, gt_paths_N2DH_GOWT1

# Funktion aufrufen und Daten speichern
imgs_N2DH_GOWT1, gts_N2DH_GOWT1, img_paths_N2DH_GOWT1, gt_paths_N2DH_GOWT1 = load_n2dh_gowt1_images()

def load_n2dl_hela_images(base_path="data-git/N2DL-HeLa"):
    img_dir_N2DL_HeLa = os.path.join(base_path, "img")
    gt_dir_N2DL_HeLa = os.path.join(base_path, "gt")

    img_paths_N2DL_HeLa = sorted(glob(os.path.join(img_dir_N2DL_HeLa, "*.tif")))
    gt_paths_N2DL_HeLa = sorted(glob(os.path.join(gt_dir_N2DL_HeLa, "*.tif")))

    imgs_N2DL_HeLa = [imread(path, as_gray=True) for path in img_paths_N2DL_HeLa]
    gts_N2DL_HeLa = [imread(path, as_gray=True) for path in gt_paths_N2DL_HeLa]

    return imgs_N2DL_HeLa, gts_N2DL_HeLa, img_paths_N2DL_HeLa, gt_paths_N2DL_HeLa

# Funktion aufrufen und Daten speichern
imgs_N2DL_HeLa, gts_N2DL_HeLa, img_paths_N2DL_HeLa, gt_paths_N2DL_HeLa = load_n2dl_hela_images()

def load_nih3t3_images(base_path="data-git/NIH3T3"):
    img_dir_NIH3T3 = os.path.join(base_path, "img")
    gt_dir_NIH3T3 = os.path.join(base_path, "gt")

    img_paths_NIH3T3 = sorted(glob(os.path.join(img_dir_NIH3T3, "*.tif")))
    gt_paths_NIH3T3 = sorted(glob(os.path.join(gt_dir_NIH3T3, "*.tif")))

    imgs_NIH3T3 = [imread(path, as_gray=True) for path in img_paths_NIH3T3]
    gts_NIH3T3 = [imread(path, as_gray=True) for path in gt_paths_NIH3T3]

    return imgs_NIH3T3, gts_NIH3T3, img_paths_NIH3T3, gt_paths_NIH3T3

# Funktion aufrufen und Daten speichern
imgs_NIH3T3, gts_NIH3T3, img_paths_NIH3T3, gt_paths_NIH3T3 = load_nih3t3_images()



# Beispiel Anwendung an N2DH-GOWT1

# Otsu anwenden
otsu_imgs = []
for img in imgs_N2DH_GOWT1:
    otsu_thresh = threshold_otsu(img)
    otsu_binary = img > otsu_thresh
    otsu_imgs.append(otsu_binary)


# GT zu binären Masken
for gt in gts_N2DH_GOWT1:
    gt_binaries = [gt > 0 for gt in gts_N2DH_GOWT1]


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

# Berechne Dice Scores für alle Bildpaare
print("Dice Scores:")
for i in range(min(len(otsu_imgs), len(gt_binaries))):
    score = dice_score(otsu_imgs[i], gt_binaries[i])
    print(f"Bild {i}: {score}")
