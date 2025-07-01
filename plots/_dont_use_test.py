# does not work yet
import os
from skimage.io import imread
from glob import glob
import matplotlib.pyplot as plt
#from skimage.filters import threshold_otsu
import numpy as np
import sys

# Aktuelles Arbeitsverzeichnis als Projekt-Root
project_root = os.getcwd()
src_dir      = os.path.join(project_root, "src")

# src-Verzeichnis ins Python-Modulverzeichnis aufnehmen
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Ausgabe-Ordner erstellen
output_dir = os.path.join(project_root, "output")
os.makedirs(output_dir, exist_ok=True)

#   
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

    img_paths_NIH3T3 = sorted(glob(os.path.join(img_dir_NIH3T3, "*.png")))
    gt_paths_NIH3T3 = sorted(glob(os.path.join(gt_dir_NIH3T3, "*.png")))

    imgs_NIH3T3 = [imread(path, as_gray=True) for path in img_paths_NIH3T3]
    gts_NIH3T3 = [imread(path, as_gray=True) for path in gt_paths_NIH3T3]

    return imgs_NIH3T3, gts_NIH3T3, img_paths_NIH3T3, gt_paths_NIH3T3

# Funktion aufrufen und Daten speichern
imgs_NIH3T3, gts_NIH3T3, img_paths_NIH3T3, gt_paths_NIH3T3 = load_nih3t3_images()


# --------------------------------------------------------------------------
# Importiere deine Otsu-Funktionen und Dice-Score
from Dice_Score import dice_score
#from otsu_global import otsu_threshold
#from gray_hist import compute_gray_histogram
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional

from skimage.exposure import histogram

def custom_histogram(image: np.ndarray, nbins: int=256) -> tuple[np.ndarray, np.ndarray]:
    """
    Reimplementation of skimage.exposure.histogram using only numpy.

    Computes histogram and bin centers over the full range of the input image.

    Args:
        image (np.ndarray): Input image.
        nbins (int): Number of histogram bins.

    Returns:
        hist (np.ndarray): Histogram frequencies.
        bin_centers (np.ndarray): Centers of histogram bins.
    """
    # Determine min and max of the image
    img_min, img_max = image.min(), image.max()

    # Compute histogram
    hist, bin_edges = np.histogram(image.ravel(), bins=nbins, range=(img_min, img_max))

    # Compute bin centers as the average of edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return hist, bin_centers

def custom_histogram1(image: np.ndarray, nbins: int=256) -> tuple[np.ndarray, np.ndarray]:
    """
    Reimplementation of skimage.exposure.histogram identically to skimage,
    including scaling the image to [0,255].
    """
    # Determine original min and max of the image
    img_min, img_max = image.min(), image.max()

    # Scale image to [0,255], like skimage does
    image_scaled = (image - img_min) / (img_max - img_min) * 255

    # Compute histogram over fixed range [0,255]
    hist, bin_edges = np.histogram(image_scaled.ravel(), bins=nbins, range=(0, 255))

    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return hist, bin_centers


def otsu_threshold_skimage_like(image: np.ndarray) -> float:
    """
    Compute the Otsu threshold identically to skimage.filters.threshold_otsu.
    """
    # Compute histogram and bin centers using skimage's exposure.histogram
    hist, bin_centers = histogram(image, nbins=256)
    hist = hist.astype(np.float64)

    # Normalize histogram to get probabilities
    hist_norm = hist / hist.sum()
    # Cumulative sums of class probabilities and class means
    weight1 = np.cumsum(hist_norm)
    weight2 = np.cumsum(hist_norm[::-1])[::-1]
    mean1 = np.cumsum(hist_norm * bin_centers)
    mean2 = np.cumsum((hist_norm * bin_centers)[::-1])[::-1]

    # Inter-class variance
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] / weight1[:-1] - mean2[1:] / weight2[1:])**2

    # Index of maximum variance gives threshold bin
    idx = np.argmax(variance12)
    threshold = bin_centers[idx]

    return threshold

def otsu_threshold_skimage_like1(image: np.ndarray) -> float:
    """
    Compute the Otsu threshold identically to skimage.filters.threshold_otsu.
    """
    # Compute histogram and bin centers using skimage's exposure.histogram
    hist, bin_centers = custom_histogram1(image, nbins=256)
    hist = hist.astype(np.float64)

    # Normalize histogram to get probabilities
    hist_norm = hist / hist.sum()
    # Cumulative sums of class probabilities and class means
    weight1 = np.cumsum(hist_norm)
    weight2 = np.cumsum(hist_norm[::-1])[::-1]
    mean1 = np.cumsum(hist_norm * bin_centers)
    mean2 = np.cumsum((hist_norm * bin_centers)[::-1])[::-1]

    # Inter-class variance
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] / weight1[:-1] - mean2[1:] / weight2[1:])**2

    # Index of maximum variance gives threshold bin
    idx = np.argmax(variance12)
    threshold = bin_centers[idx]

    return threshold

def otsu_threshold_skimage_like2(image: np.ndarray) -> float:
    """
    Compute the Otsu threshold identically to skimage.filters.threshold_otsu.
    """
    # Compute histogram and bin centers using skimage's exposure.histogram
    hist, bin_centers = custom_histogram1(image, nbins=256)
    hist = hist.astype(np.float64)

    # Normalize histogram to get probabilities
    hist_norm = hist / hist.sum()
    # Cumulative sums of class probabilities and class means
    weight1 = np.cumsum(hist_norm)
    weight2 = np.cumsum(hist_norm[::-1])[::-1]
    mean1 = np.cumsum(hist_norm * bin_centers)
    mean2 = np.cumsum((hist_norm * bin_centers)[::-1])[::-1]

    # Inter-class variance
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] / weight1[:-1] - mean2[1:] / weight2[1:])**2

    # Index of maximum variance gives threshold bin
    idx = np.argmax(variance12)
    threshold_scaled = bin_centers[idx]

    # Rescale threshold back to original image range!
    img_min, img_max = image.min(), image.max()
    threshold_original = threshold_scaled / 255 * (img_max - img_min) + img_min

    return threshold_original



# --------------------------------------------------------------
def berechne_dice_scores(imgs, gts):
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
        # Histogramm und Wahrscheinlichkeit berechnen
        #hist, _ = compute_gray_histogram(img)
        #p = hist / hist.sum()

        # Otsu-Schwelle berechnen
        t = otsu_threshold_skimage_like(img)

        # Bild binarisieren
        otsu_img = img > t

        # GT binarisieren
        gt_binary = gt > 0

        # Dice-Score berechnen
        score = dice_score(otsu_img, gt_binary)
        scores.append(score)
    return scores

def berechne_dice_scores1(imgs, gts):
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
        # Histogramm und Wahrscheinlichkeit berechnen
        #hist, _ = compute_gray_histogram(img)
        #p = hist / hist.sum()

        # Otsu-Schwelle berechnen
        t = otsu_threshold_skimage_like2(img)

        # Bild binarisieren
        otsu_img = img > t

        # GT binarisieren
        gt_binary = gt > 0

        # Dice-Score berechnen
        score = dice_score(otsu_img, gt_binary)
        scores.append(score)
    return scores

from skimage.filters import threshold_otsu

def calculate_dice_scores_OP(imgs, gts):
    """
    Binarize the images using Otsu's threshold and compute the Dice-score
    against the corresponding ground-truth masks.

    Args:
        imgs (list[np.ndarray]): List of grayscale images.
        gts (list[np.ndarray]): List of ground-truth masks.

    Returns:
        scores (list[float]): List of Dice-scores for each image/ground-truth pair.
    """
    # Compute a binary version of each image using Otsu's threshold
    otsu_imgs = [img > threshold_otsu(img) for img in imgs]

    # Binarize the ground-truths (assuming they are already masks)
    gt_binaries = [gt > 0 for gt in gts]

    scores = []
    for i in range(min(len(otsu_imgs), len(gt_binaries))):
        score = dice_score(otsu_imgs[i], gt_binaries[i])
        scores.append(score)

    return scores

# --------------------------------------------------------------
# Berechne die Dice-Scores für die drei Datensätze
#dice_gowt1 = berechne_dice_scores(imgs_N2DH_GOWT1, gts_N2DH_GOWT1)
#dice_hela = berechne_dice_scores(imgs_N2DL_HeLa, gts_N2DL_HeLa)
#dice_nih = berechne_dice_scores(imgs_NIH3T3, gts_NIH3T3)

ndice_gowt1 = berechne_dice_scores1(imgs_N2DH_GOWT1, gts_N2DH_GOWT1)
ndice_hela = berechne_dice_scores1(imgs_N2DL_HeLa, gts_N2DL_HeLa)
ndice_nih = berechne_dice_scores1(imgs_NIH3T3, gts_NIH3T3)

#sdice_gowt1 = calculate_dice_scores_OP(imgs_N2DH_GOWT1, gts_N2DH_GOWT1)
#sdice_hela = calculate_dice_scores_OP(imgs_N2DL_HeLa, gts_N2DL_HeLa)
#sdice_nih = calculate_dice_scores_OP(imgs_NIH3T3, gts_NIH3T3)

# Als einfache Floats statt np.float64
dice_gowt1 = [float(score) for score in ndice_gowt1]
dice_hela = [float(score) for score in ndice_hela]
dice_nih = [float(score) for score in ndice_nih]

# Schön formatiert ausgeben
#print("GOWT1 Scores =", [f"{score}" for score in dice_gowt1])
#print("HeLa Scores =", [f"{score}" for score in dice_hela])
#print("NIH3T3 Scores =", [f"{score}" for score in dice_nih])

print("GOWT1 Scores =", [f"{score}" for score in ndice_gowt1])
print("HeLa Scores =", [f"{score}" for score in ndice_hela])
print("NIH3T3 Scores =", [f"{score}" for score in ndice_nih])

#a = dice_gowt1 == ndice_gowt1
#b = dice_hela == ndice_hela
#c = dice_nih == ndice_nih

#print(a,b,c)

#d = dice_gowt1 == sdice_gowt1
#e = dice_hela == sdice_hela
#f = dice_nih == sdice_nih

#print(d,e,f)
 
# after testing we chose otsu_threshold_skimage_like2