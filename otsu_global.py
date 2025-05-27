"""
otsu_threshold.py

Lädt ein Graustufenbild, berechnet mit Otsu einen globalen Schwellenwert
und speichert das Binärbild. 
"""

import numpy as np
from skimage import io, color
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

# 1. Bild laden
img = io.imread('data-git/N2DH-GOWT1/gt/')
# in Graustufen umwandeln, falls es ein Farbbild ist
if img.ndim == 3:
    img = color.rgb2gray(img)

# 2. Otsu-Schwellenwert berechnen
thresh = threshold_otsu(img)

# 3. Binärbild erstellen
binary = img > thresh

# 4. Ergebnisse anzeigen
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
ax = axes.ravel()
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original')
ax[1].hist(img.ravel(), bins=256)
ax[1].axvline(thresh, color='red', linestyle='--')
ax[1].set_title(f'Histogramm\nOtsu = {thresh:.4f}')
ax[2].imshow(binary, cmap='gray')
ax[2].set_title('Binärbild')
for a in ax:
    a.axis('off')
plt.tight_layout()
plt.show()
