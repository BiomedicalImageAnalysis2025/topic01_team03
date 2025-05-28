# gray_hist.py

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path

def compute_gray_histogram(image_path: Path, bins: int = 256, value_range=(0, 255)):
    """
    Liest ein Bild ein, wandelt es in Graustufen um und berechnet das Histogramm.
    """
    img = Image.open(str(image_path)).convert("L")
    arr = np.array(img)
    hist, bin_edges = np.histogram(arr.ravel(), bins=bins, range=value_range)
    return hist, bin_edges

def plot_gray_histogram(hist: np.ndarray, bin_edges: np.ndarray):
    """
    Plottet ein Grauwert-Histogramm anhand von hist und bin_edges.
    """
    plt.figure(figsize=(8, 4))
    plt.bar(bin_edges[:-1], hist, width=bin_edges[1] - bin_edges[0])
    plt.title("Grauwert-Histogramm")
    plt.xlabel(f"Grauwert ({int(bin_edges[0])}–{int(bin_edges[-1])})")
    plt.ylabel("Anzahl Pixel")
    plt.xlim(bin_edges[0], bin_edges[-1])
    plt.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


"""
if __name__ == "__main__":
    # Pfad zum Bild anpassen
    img_path = Path("data-git/N2DH-GOWT1/gt/man_seg01.tif")
    # Histogramm berechnen
    hist_array, edges = compute_gray_histogram(img_path)
    # Hier hast du das Histogramm-Array zur Weiterverarbeitung:
    print("Histogramm-Array:", hist_array)
    # Optional: plotten
    plot_gray_histogram(hist_array, edges)

    """