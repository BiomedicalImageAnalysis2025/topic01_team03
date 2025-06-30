# does not work yet
import os
from skimage.io import imread
from glob import glob
import matplotlib.pyplot as plt
#from skimage.filters import threshold_local
import numpy as np
import sys

# Set the current working directory as the project root
project_root = os.getcwd()
src_dir = os.path.join(project_root, "src")

# Add src directory to the Python path
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Create output directory
output_dir = os.path.join(project_root, "output")
os.makedirs(output_dir, exist_ok=True)

# --------------------------------------------------------------
def load_n2dh_gowt1_images(base_path="data-git/N2DH-GOWT1"):
    """
    Load the N2DH-GOWT1 image dataset and its corresponding ground-truth masks.

    Args:
        base_path (str): Base directory containing the "img" and "gt" subfolders.
                         Default is "data-git/N2DH-GOWT1".

    Returns:
        imgs_N2DH_GOWT1 (list[np.ndarray]): List of loaded grayscale images.
        gts_N2DH_GOWT1 (list[np.ndarray]): List of loaded grayscale ground-truth masks.
        img_paths_N2DH_GOWT1 (list[str]): List of file paths to the images.
        gt_paths_N2DH_GOWT1 (list[str]): List of file paths to the ground-truth masks.
    """
    img_dir_N2DH_GOWT1 = os.path.join(base_path, "img")
    gt_dir_N2DH_GOWT1 = os.path.join(base_path, "gt")

    img_paths_N2DH_GOWT1 = sorted(glob(os.path.join(img_dir_N2DH_GOWT1, "*.tif")))
    gt_paths_N2DH_GOWT1 = sorted(glob(os.path.join(gt_dir_N2DH_GOWT1, "*.tif")))

    imgs_N2DH_GOWT1 = [imread(path, as_gray=True) for path in img_paths_N2DH_GOWT1]
    gts_N2DH_GOWT1 = [imread(path, as_gray=True) for path in gt_paths_N2DH_GOWT1]

    return imgs_N2DH_GOWT1, gts_N2DH_GOWT1, img_paths_N2DH_GOWT1, gt_paths_N2DH_GOWT1

# Load N2DH-GOWT1 data
imgs_N2DH_GOWT1, gts_N2DH_GOWT1, img_paths_N2DH_GOWT1, gt_paths_N2DH_GOWT1 = load_n2dh_gowt1_images()

# --------------------------------------------------------------
def load_n2dl_hela_images(base_path="data-git/N2DL-HeLa"):
    """
    Load the N2DL-HeLa image dataset and its corresponding ground-truth masks.

    Args:
        base_path (str): Base directory containing the "img" and "gt" subfolders.
                         Default is "data-git/N2DL-HeLa".

    Returns:
        imgs_N2DL_HeLa (list[np.ndarray]): List of loaded grayscale images.
        gts_N2DL_HeLa (list[np.ndarray]): List of loaded grayscale ground-truth masks.
        img_paths_N2DL_HeLa (list[str]): List of file paths to the images.
        gt_paths_N2DL_HeLa (list[str]): List of file paths to the ground-truth masks.
    """
    img_dir_N2DL_HeLa = os.path.join(base_path, "img")
    gt_dir_N2DL_HeLa = os.path.join(base_path, "gt")

    img_paths_N2DL_HeLa = sorted(glob(os.path.join(img_dir_N2DL_HeLa, "*.tif")))
    gt_paths_N2DL_HeLa = sorted(glob(os.path.join(gt_dir_N2DL_HeLa, "*.tif")))

    imgs_N2DL_HeLa = [imread(path, as_gray=True) for path in img_paths_N2DL_HeLa]
    gts_N2DL_HeLa = [imread(path, as_gray=True) for path in gt_paths_N2DL_HeLa]

    return imgs_N2DL_HeLa, gts_N2DL_HeLa, img_paths_N2DL_HeLa, gt_paths_N2DL_HeLa

# Load N2DL-HeLa data
imgs_N2DL_HeLa, gts_N2DL_HeLa, img_paths_N2DL_HeLa, gt_paths_N2DL_HeLa = load_n2dl_hela_images()

# --------------------------------------------------------------
def load_nih3t3_images(base_path="data-git/NIH3T3"):
    """
    Load the NIH3T3 image dataset and its corresponding ground-truth masks.

    Args:
        base_path (str): Base directory containing the "img" and "gt" subfolders.
                         Default is "data-git/NIH3T3".

    Returns:
        imgs_NIH3T3 (list[np.ndarray]): List of loaded grayscale images.
        gts_NIH3T3 (list[np.ndarray]): List of loaded grayscale ground-truth masks.
        img_paths_NIH3T3 (list[str]): List of file paths to the images.
        gt_paths_NIH3T3 (list[str]): List of file paths to the ground-truth masks.
    """
    img_dir_NIH3T3 = os.path.join(base_path, "img")
    gt_dir_NIH3T3 = os.path.join(base_path, "gt")

    img_paths_NIH3T3 = sorted(glob(os.path.join(img_dir_NIH3T3, "*.png")))
    gt_paths_NIH3T3 = sorted(glob(os.path.join(gt_dir_NIH3T3, "*.png")))

    imgs_NIH3T3 = [imread(path, as_gray=True) for path in img_paths_NIH3T3]
    gts_NIH3T3 = [imread(path, as_gray=True) for path in gt_paths_NIH3T3]

    return imgs_NIH3T3, gts_NIH3T3, img_paths_NIH3T3, gt_paths_NIH3T3

# Load NIH3T3 data
imgs_NIH3T3, gts_NIH3T3, img_paths_NIH3T3, gt_paths_NIH3T3 = load_nih3t3_images()

# --------------------------------------------------------------
from Dice_Score import dice_score
#from otsu_local import local_otsu
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
from typing import Union, Tuple

def compute_gray_histogram(
    image_source: Union[Path, str, np.ndarray],
    bins: int = 256,
    value_range: Tuple[int, int] = (0, 255)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Liest ein Bild ein (Pfad oder NumPy-Array), wandelt in Graustufen um und
    berechnet das Histogramm.

    Args:
        image_source: Pfad (Path/str) zum Bild ODER ein 2D-NumPy-Array mit Grauwerten.
        bins: Anzahl der Bins für das Histogramm.
        value_range: Wertebereich (min, max).

    Returns:
        hist: Array der Pixelhäufigkeiten pro Bin.
        bin_edges: Randwerte der Bins.
    """
    # 1) Input erkennen und in Grauwert-Array umwandeln
    if isinstance(image_source, (Path, str)):
        img = Image.open(str(image_source)).convert("L")
        arr = np.array(img)
    elif isinstance(image_source, np.ndarray):
        arr = image_source
    else:
        raise TypeError(
            "compute_gray_histogram erwartet einen Pfad (Path/str) oder ein NumPy-Array."
        )

    # 2) Histogramm berechnen
    hist, bin_edges = np.histogram(
        arr.ravel(),
        bins=bins,
        range=value_range
    )
    return hist, bin_edges


def plot_gray_histogram(hist: np.ndarray, bin_edges: np.ndarray):
    """
    Plottet ein Grauwert-Histogramm anhand von hist und bin_edges.
    """
    plt.figure(figsize=(8, 4))
    plt.bar(
        bin_edges[:-1],
        hist,
        width=bin_edges[1] - bin_edges[0],
        align='edge'
    )
    plt.t

def otsu_threshold(p: np.ndarray) -> int:
    """
    Berechnet den globalen Otsu-Schwellenwert aus Wahrscheinlichkeiten p[k].
    """
    P = np.cumsum(p)                    # kumulative Wahrscheinlichkeiten
    bins = np.arange(len(p))            # mögliche Grauwert-Indizes
    mu = np.cumsum(bins * p)            # kumuliertes gewichtetes Mittel
    mu_T = mu[-1]                       # Gesamtmittel
    # Interklassenvarianz mit Epsilon für Stabilität
    sigma_b2 = (mu_T * P - mu)**2 / (P * (1 - P) + 1e-12)
    return int(np.argmax(sigma_b2))

import os
import numpy as np
from skimage import img_as_ubyte

def local_otsu(image: np.ndarray, radius: int = 3) -> (np.ndarray, np.ndarray):
    """
    Führt lokales Otsu Thresholding aus:
      - radius: Halbbreite des Fensters (ergibt (2*radius+1)^2 Pixel).
    Gibt zurück:
      - t_map:  2D‐Array mit lokalem Schwellenwert pro Pixel
      - mask:   2D‐Bool‐Array, True = Objekt (Pixel > t_map)
    """
    # 1. Stelle sicher, dass image uint8 ist
    img_u8 = img_as_ubyte(image)
    H, W = img_u8.shape

    # 2. Lege Map und Maske an
    t_map = np.zeros((H, W), dtype=np.uint8)
    mask  = np.zeros((H, W), dtype=bool)

    # 3. Pad das Bild, damit Randpixel verarbeitet werden können
    pad = radius
    padded = np.pad(img_u8, pad, mode="reflect")

    # 4. Über jeden Pixel im Originalbild iterieren
    w = 2 * radius + 1
    for i in range(H):
        for j in range(W):
            # Fenster im gepaddeten Bild herausschneiden
            block = padded[i : i + w, j : j + w]

            # Histogramm der Graustufen 0..255 im Block
            hist, _ = compute_gray_histogram(block)
            p = hist / hist.sum()

            # Otsu‐Threshold aus dem lokalen Histogramm
            t = otsu_threshold(p)
            t_map[i, j] = t

            # Binärmaske setzen
            mask[i, j] = (img_u8[i, j] > t)

    return t_map, mask

def calculate_dice_scores_local(imgs, gts):
    otsu_imgs = []
    for img in imgs:
        t_map, mask = local_otsu(img)  # beide Rückgabewerte
        otsu_imgs.append(mask)         # nur die binäre Maske nutzen

    gt_binaries = [gt > 0 for gt in gts]

    scores = []
    for i in range(min(len(otsu_imgs), len(gt_binaries))):
        scores.append(dice_score(otsu_imgs[i], gt_binaries[i]))

    return scores


# --------------------------------------------------------------
# Compute Dice-scores for all datasets
dice_gowt1 = calculate_dice_scores_local(imgs_N2DH_GOWT1, gts_N2DH_GOWT1)
dice_hela = calculate_dice_scores_local(imgs_N2DL_HeLa, gts_N2DL_HeLa)
dice_nih = calculate_dice_scores_local(imgs_NIH3T3, gts_NIH3T3)

# Convert scores to regular Python floats
dice_gowt1 = [float(score) for score in dice_gowt1]
dice_hela = [float(score) for score in dice_hela]
dice_nih = [float(score) for score in dice_nih]

# Print all Dice-scores
print("GOWT1 Scores:", [f"{score}" for score in dice_gowt1])
print("HeLa Scores:", [f"{score}" for score in dice_hela])
print("NIH3T3 Scores:", [f"{score}" for score in dice_nih])