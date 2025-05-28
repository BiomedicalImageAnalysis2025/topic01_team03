"""
otsu_threshold.py
Verwendet compute_gray_histogram aus src.gray_hist und behält
die mathematischen Kommentare zur Otsu‐Implementierung bei.
"""

import math
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path

# Histogramm‐Utilities aus src/gray_hist.py
from gray_hist import compute_gray_histogram, plot_gray_histogram

def otsu_threshold(p: np.ndarray) -> int:
    """
    Berechnet Otsus globalen Schwellenwert basierend auf
    Wahrscheinlichkeiten p[k] für k=0..255.
    """
    # kumulative Summen
    P = np.cumsum(p)                     # P[i] = sum_{k=0}^i p[k]
    bins = np.arange(256)
    μ = np.cumsum(bins * p)              # μ[i] = sum_{k=0}^i k * p[k]
    μ_T = μ[-1]                          # Gesamtmittel

    # Interklassen-Varianz σ_B²(i) = [μ_T * P[i] - μ[i]]² / [P[i]*(1-P[i])]
    # Wir vermeiden Division durch 0, indem wir kleinen Epsilon‐Term hinzufügen.
    σ_B2 = (μ_T * P - μ)**2 / (P * (1 - P) + 1e-12)

    # besten Schwellenwert wählen
    t_opt = np.argmax(σ_B2)
    return t_opt

def binarize(arr: np.ndarray, t: int) -> np.ndarray:
    """Wendet den Schwellenwert t an und liefert ein Boolean-Array."""
    return arr > t

if __name__ == "__main__":
    # Bild laden und in Graustufen konvertieren
    img_path = Path("data-git/N2DH-GOWT1/gt/man_seg01.tif")
    img = Image.open(str(img_path)).convert("L")
    arr = np.array(img)

    # Histogramm + Bin-Ränder aus src.gray_hist
    hist, bin_edges = compute_gray_histogram(img_path)

    # Wahrscheinlichkeiten p[k] = hist[k] / N
    p = hist / hist.sum()

    # Otsu-Schwellenwert bestimmen
    t = otsu_threshold(p)
    print(f"Optimaler Otsu-Schwellenwert: {t}")

    # Binarisieren und anzeigen
    binary = binarize(arr, t)
    plt.figure(figsize=(6,6))
    plt.imshow(binary, cmap='gray')
    plt.title("Binärbild mit globalem Otsu-Threshold")
    plt.axis('off')
    plt.show()

    # Histogramm plotten (aus src.gray_hist) und Markierung t
    plot_gray_histogram(hist, bin_edges)
    plt.axvline(bin_edges[t], color='r', linestyle='--', label=f"t={bin_edges[t]:.0f}")
    plt.legend()
    plt.show()
