#src/otsu_lokal.py

import os
import numpy as np
from skimage import img_as_ubyte

from src.gray_hist import compute_gray_histogram
from src.Complete_Otsu_Global import otsu_threshold_skimage_like

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
            t = otsu_threshold_skimage_like(p)
            t_map[i, j] = t

            # Binärmaske setzen
            mask[i, j] = (img_u8[i, j] > t)

    return t_map, mask