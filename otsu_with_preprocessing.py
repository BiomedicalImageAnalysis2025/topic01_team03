"""
otsu_with_preprocessing.py

Verwendet src.gray_hist.compute_gray_histogram für das Original-Histogramm
und src.otsu_global.otsu_threshold sowie binarize für das finale Thresholding.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread
from pathlib import Path

from src.gray_hist import compute_gray_histogram, plot_gray_histogram
from src.otsu_global import otsu_threshold, binarize

# Bild einlesen und skalieren (0–1)
image_path = Path("data-git/N2DH-GOWT1/img/t01.tif")
img = imread(str(image_path)).astype(float)
img /= img.max()

# Gamma–Korrektur
mean_brightness = img.mean()
gamma = 3 if mean_brightness >= 0.5 else 0.5
img_gamma = img ** gamma


# Histogramm des ORIGINALBILDES (optional, zur Vergleich)
orig_hist, orig_edges = compute_gray_histogram(image_path)
plt.figure(figsize=(8,4))
plot_gray_histogram(orig_hist, orig_edges)
plt.title("Histogramm des Originalbildes")

#Histogramm der vorverarbeiteten Version für Otsu
#    Wir erstellen hier das Histogramm manuell, da compute_gray_histogram
#    direkt von einem Dateipfad liest.
pixels = (img_gamma * 255).clip(0,255).astype(np.uint8).ravel()
hist_pre, edges_pre = np.histogram(pixels, bins=256, range=(0,256))
p_pre = hist_pre / hist_pre.sum()

# Otsu-Schwellenwert bestimmen
t = otsu_threshold(p_pre)
print(f"Gefundener Otsu-Threshold: {t} (Grauwert {edges_pre[t]})")

# Binarisierung
binary = binarize((img_gamma * 255).astype(np.uint8), t)

# Visualisierung
fig, axes = plt.subplots(1, 4, figsize=(16,4))

axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original (0–1)")
axes[0].axis("off")

axes[1].imshow(img_gamma, cmap="gray")
axes[1].set_title(f"Gamma (γ={gamma})")
axes[1].axis("off")

axes[2].imshow(binary, cmap="gray")
axes[2].set_title(f"Otsu Binär (t={edges_pre[t]})")
axes[2].axis("off")

axes[3].bar(edges_pre[:-1], hist_pre, width=1.0, alpha=0.6)
axes[3].axvline(edges_pre[t], color="r", linestyle="--",
                label=f"t={edges_pre[t]}")
axes[3].set_title("Preprocessed Histogramm")
axes[3].set_xlabel("Grauwert")
axes[3].set_ylabel("Pixelanzahl")
axes[3].legend()

plt.tight_layout()
plt.show()
