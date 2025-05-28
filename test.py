"""
otsu_threshold_with_utils.py
"""

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path

from src.gray_hist import compute_gray_histogram, plot_gray_histogram

def otsu_threshold(p):
    """
    Berechnet Otsus globalen Schwellenwert basierend auf
    Wahrscheinlichkeiten p[k] für k=0..255.
    """
    P = np.cumsum(p)
    bins = np.arange(256)
    mu = np.cumsum(bins * p)
    mu_T = mu[-1]
    sigma2 = (mu_T * P - mu)**2 / (P * (1 - P) + 1e-12)
    return np.argmax(sigma2)

def binarize(arr, t):
    """Wendet den Schwellenwert t an und liefert ein Boolean-Array."""
    return arr > t

if __name__ == "__main__":
    # 1. Bildpfad und Laden
    img_path = Path("data-git/N2DH-GOWT1/gt/man_seg01.tif")
    img = Image.open(str(img_path)).convert("L")
    arr = np.array(img)

    # 2. Histogramm und Bin-Grenzen aus hist_utils
    hist, bin_edges = compute_gray_histogram(img_path)

    # 3. Wahrscheinlichkeiten
    p = hist / hist.sum()

    # 4. Otsu-Schwellenwert bestimmen
    t = otsu_threshold(p)
    print(f"Optimaler Otsu-Schwellenwert: {t}")

    # 5. Binarisieren
    binary = binarize(arr, t)

    # 6. Visualisierung
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(arr, cmap="gray")
    axes[0].set_title("Original (Graustufen)")
    axes[0].axis("off")

    axes[1].imshow(binary, cmap="gray")
    axes[1].set_title(f"Binär (t={t})")
    axes[1].axis("off")

    # 7. Histogramm plotten und Threshold markieren
    axes[2].bar(bin_edges[:-1], hist, width=bin_edges[1] - bin_edges[0], alpha=0.6)
    axes[2].axvline(bin_edges[t], color='r', linestyle='--', label=f"t={bin_edges[t]:.0f}")
    axes[2].set_title("Histogramm mit Otsu-Threshold")
    axes[2].set_xlabel("Grauwert")
    axes[2].set_ylabel("Anzahl Pixel")
    axes[2].legend()

    plt.tight_layout()
    plt.show()
