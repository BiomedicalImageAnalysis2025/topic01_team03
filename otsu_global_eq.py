"""
otsu_threshold.py
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread
from PIL import Image
import math

# ————————————————————————————————————————————————————————
# 1. Bild einlesen und skalieren (0–1)
image = "data-git/N2DH-GOWT1/img/t01.tif"
img = imread(image).astype(float)
img /= img.max()  

# 2. Gamma–Korrektur
mean_brightness = img.mean()
gamma = 3 if mean_brightness >= 0.5 else 0.5
img_gamma = img ** gamma


# 3. Histogramm­equalization (optional, verbessert Kontrast weiter)
#    Hier mit CDF–Mapping auf 0–255 und zurück auf 0–1
img8 = (img_gamma * 255).clip(0,255).astype(np.uint8)
hist, bins = np.histogram(img8, bins=256, range=[0,256])
cdf = hist.cumsum()
cdf = (cdf / cdf[-1])          # 0–1 normalisiert
img_eq = cdf[img8]             # Lookup
# jetzt img_eq ist wieder Float–Array in [0,1]

# ————————————————————————————————————————————————————————
# 4. Otsu–Implementation

def otsu_threshold_from_prob(p):
    # p: Vektor der Wahrscheinlichkeiten p[k] für k=0..255
    P = np.cumsum(p)
    bins = np.arange(256)
    mu = np.cumsum(p * bins)
    mu_T = mu[-1]
    # inter-class variance
    sigma2 = (mu_T * P - mu)**2 / (P * (1-P) + 1e-12)
    return np.argmax(sigma2)

# berechne p aus img_eq
img_eq8 = (img_eq * 255).astype(np.uint8)
hist_eq, _ = np.histogram(img_eq8, bins=256, range=(0,255))
p_eq = hist_eq / hist_eq.sum()

# 5. optimalen Threshold finden
t = otsu_threshold_from_prob(p_eq)
print(f"gefundenes Otsu-t: {t}")

# 6. Binarisierung und Visualisierung
binary = img_eq8 > t

fig, ax = plt.subplots(1,3,figsize=(12,4))
ax[0].imshow(img,      cmap='gray'); ax[0].set_title("Original (0–1)")
ax[1].imshow(img_eq,   cmap='gray'); ax[1].set_title("Preprocessed")
ax[2].imshow(binary,   cmap='gray'); ax[2].set_title(f"Binarisiert (t={t})")
for a in ax: a.axis('off')
plt.tight_layout()
plt.show()
