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
        hist, _ = compute_gray_histogram(img)
        p = hist / hist.sum()

        # Otsu-Schwelle berechnen
        t = otsu_threshold(p)

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
print("GOWT1 Scores:", [f"{score:.4f}" for score in dice_gowt1])
print("HeLa Scores:", [f"{score:.4f}" for score in dice_hela])
print("NIH3T3 Scores:", [f"{score:.4f}" for score in dice_nih])