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
from typing import Tuple

def custom_histogram(image: np.ndarray, nbins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the histogram and corresponding bin centers of a grayscale image,
    replicating the behavior of skimage.exposure.histogram, including normalization
    to the [0, 255] range. This ensures consistent behavior with Otsu implementations
    that assume 8-bit images.

    Args:
        image (np.ndarray): Input image as a 2D array of grayscale values.
        nbins (int): Number of bins for the histogram (default: 256).

    Returns:
        hist (np.ndarray): Array of histogram frequencies for each bin.
        bin_centers (np.ndarray): Array of bin center values.
    """
    # Determine the minimum and maximum pixel intensity in the image
    img_min, img_max = image.min(), image.max()

    # Normalize the image intensities to the range [0, 255], as in skimage
    image_scaled = (image - img_min) / (img_max - img_min) * 255

    # Compute the histogram of the scaled image within [0, 255]
    hist, bin_edges = np.histogram(
        image_scaled.ravel(),
        bins=nbins,
        range=(0, 255)
    )

    # Compute bin centers as the average of adjacent bin edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return hist, bin_centers


def otsu_threshold_skimage_like(image: np.ndarray) -> float:
    """
    Computes the global Otsu threshold of an input grayscale image in a way that matches
    the behavior of skimage.filters.threshold_otsu, including histogram scaling and
    threshold rescaling back to the original intensity range.

    This function enables nearly identical thresholding results to skimage's implementation,
    even on images with floating-point or non-8-bit integer data.

    Args:
        image (np.ndarray): Input image as a 2D array of grayscale values.

    Returns:
        threshold_original (float): Computed Otsu threshold mapped back to the original image range.
    """
    # Compute histogram and bin centers consistent with skimage
    hist, bin_centers = custom_histogram(image, nbins=256)
    hist = hist.astype(np.float64)

    # Normalize histogram to obtain probability distribution
    hist_norm = hist / hist.sum()

    # Compute cumulative sums of class probabilities and means
    weight1 = np.cumsum(hist_norm)
    weight2 = np.cumsum(hist_norm[::-1])[::-1]
    mean1 = np.cumsum(hist_norm * bin_centers)
    mean2 = np.cumsum((hist_norm * bin_centers)[::-1])[::-1]

    # Compute inter-class variance for each threshold
    variance12 = (
        weight1[:-1] * weight2[1:] *
        (mean1[:-1] / weight1[:-1] - mean2[1:] / weight2[1:])**2
    )

    # Find the bin index corresponding to the maximum inter-class variance
    idx = np.argmax(variance12)
    threshold_scaled = bin_centers[idx]

    # Rescale the threshold from [0, 255] back to the original image intensity range
    img_min, img_max = image.min(), image.max()
    final_threshold = threshold_scaled / 255 * (img_max - img_min) + img_min

    return final_threshold


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

# --------------------------------------------------------------
# Berechne die Dice-Scores für die drei Datensätze
dice_gowt1 = berechne_dice_scores(imgs_N2DH_GOWT1, gts_N2DH_GOWT1)
dice_hela = berechne_dice_scores(imgs_N2DL_HeLa, gts_N2DL_HeLa)
dice_nih = berechne_dice_scores(imgs_NIH3T3, gts_NIH3T3)

# Als einfache Floats statt np.float64
dice_gowt1 = [float(score) for score in dice_gowt1]
dice_hela = [float(score) for score in dice_hela]
dice_nih = [float(score) for score in dice_nih]

# Schön formatiert ausgeben
print("GOWT1 Scores =", [f"{score}" for score in dice_gowt1])
print("HeLa Scores =", [f"{score}" for score in dice_hela])
print("NIH3T3 Scores =", [f"{score}" for score in dice_nih])