#!/usr/bin/env python3
"""
otsu_threshold.py

Modul für globalen Otsu-Schwellenwert:
  - compute_gray_histogram aus src/gray_hist verwenden
  - Funktionen:
      * otsu_threshold(p): berechnet den optimalen Schwellenwert
      * binarize(arr, t): wendet den Schwellenwert an
      * apply_global_otsu(image): volle Pipeline (Histogramm → Threshold → Binarisierung)

Im __main__-Block:
  - Beispielbild laden
  - Binarisierung anwenden
  - Original- und Binärbild anzeigen und abspeichern
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.gray_hist import compute_gray_histogram


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

def binarize(arr: np.ndarray, t: int) -> np.ndarray:
    """
    Wendet den Schwellenwert t an und gibt ein binäres 0/1-Array zurück.
    """
    return (arr > t).astype(np.uint8)

def threshold_global(image: np.ndarray) -> np.ndarray:
    """
    Vollständige Pipeline:
    - Histogramm berechnen
    - Wahrscheinlichkeiten p[k] bilden
    - Otsu-Schwellenwert berechnen

    Returns ein 2D-Binär-Array (0/1).
    """
    hist, _ = compute_gray_histogram(image)
    p = hist / hist.sum()
    return otsu_threshold(p)

def apply_global_otsu(image: np.ndarray) -> np.ndarray:
    """
    Vollständige Pipeline:
    - Histogramm berechnen
    - Wahrscheinlichkeiten p[k] bilden
    - Otsu-Schwellenwert berechnen
    - Binarisierung durchführen

    Returns ein 2D-Binär-Array (0/1).
    """
    hist, _ = compute_gray_histogram(image)
    p = hist / hist.sum()
    t = otsu_threshold(p)
    return binarize(image, t)


if __name__ == "__main__":
    # Ausgabe-Ordner vorbereiten
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Beispielbild laden und in Grauwert-Array umwandeln
    img_path = os.path.join(project_root, "Data", "N2DH-GOWT1", "img", "t01.tif")
    pil_img = Image.open(img_path).convert("L")
    img_arr = np.array(pil_img)

    # Globalen Otsu anwenden
    binary = apply_global_otsu(img_arr)

    # Schwellenwert ermitteln (nur zur Anzeige)
    hist, _ = compute_gray_histogram(img_arr)
    p = hist / hist.sum()
    t = otsu_threshold(p)
    print(f"Berechneter Otsu-Schwellenwert: {t}")

    # Binärbild (0/255) speichern
    binary_uint8 = binary * 255
    out_path = os.path.join(output_dir, "t01_binary.png")
    Image.fromarray(binary_uint8).save(out_path)
    print(f"Binärbild gespeichert unter: {out_path}")

    # Original- und Ergebnisbild anzeigen
    plt.figure(figsize=(5,5))
    plt.imshow(img_arr, cmap="gray")
    plt.title("Original Grauwertbild")
    plt.axis("off")

    plt.figure(figsize=(5,5))
    plt.imshow(binary, cmap="gray")
    plt.title(f"Global Otsu Binärbild (t={t})")
    plt.axis("off")

    plt.show()
