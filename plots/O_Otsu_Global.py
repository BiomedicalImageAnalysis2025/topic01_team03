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
# Hilfsfunktion zur Dice-Berechnung
from Dice_Score import dice_score
from src.otsu_global import otsu_threshold
from src.gray_hist import compute_gray_histogram
#from otsu_global import apply_global_otsu
# Hilfsfunktion zur Berechnung der Dice-Werte
#def berechne_dice_scores(imgs, gts, datensatz_name):
    # Otsu-Binarisierung
 #   otsu_imgs = []
  #  for img in imgs:
   #     otsu_thresh = threshold_otsu(img)
    #    otsu_imgs.append(img > otsu_thresh)
    ## GT-Binarisierung
#    gt_binaries = [gt > 0 for gt in gts]
 #   # Dice-Berechnung
  #  print(f"\nDice Scores {datensatz_name}:")
   # for i in range(min(len(otsu_imgs), len(gt_binaries))):
    #    score = dice_score(otsu_imgs[i], gt_binaries[i])
     #   print(f"Bild {i}: {score}")

# Hilfsfunktion zur Berechnung der Dice-Werte und Rückgabe der Scores
def berechne_dice_scores(imgs, gts):
    hist, _ = compute_gray_histogram(imgs)
    p = hist / hist.sum()
    otsu_imgs = [img > otsu_threshold(p) for img in imgs]
    gt_binaries = [gt > 0 for gt in gts]
    scores = []
    for i in range(min(len(otsu_imgs), len(gt_binaries))):
        score = dice_score(otsu_imgs[i], gt_binaries[i])
        scores.append(score)
    return scores

# --------------------------------------------------------------------------
# N2DH-GOWT1
#berechne_dice_scores(imgs_N2DH_GOWT1, gts_N2DH_GOWT1, "N2DH-GOWT1")

# N2DL-HeLa
#berechne_dice_scores(imgs_N2DL_HeLa, gts_N2DL_HeLa, "N2DL-HeLa")

# NIH3T3
#berechne_dice_scores(imgs_NIH3T3, gts_NIH3T3, "NIH3T3")

# Berechne die Dice-Scores
dice_gowt1 = berechne_dice_scores(imgs_N2DH_GOWT1, gts_N2DH_GOWT1)
dice_hela = berechne_dice_scores(imgs_N2DL_HeLa, gts_N2DL_HeLa)
dice_nih = berechne_dice_scores(imgs_NIH3T3, gts_NIH3T3)

# Als einfache Floats statt np.float64
dice_gowt1 = [float(score) for score in dice_gowt1]
dice_hela = [float(score) for score in dice_hela]
dice_nih = [float(score) for score in dice_nih]

# Schön formatiert mit 3 Nachkommastellen ausgeben
print("GOWT1 Scores:", [f"{score}" for score in dice_gowt1])
print("HeLa Scores:", [f"{score}" for score in dice_hela])
print("NIH3T3 Scores:", [f"{score}" for score in dice_nih])
