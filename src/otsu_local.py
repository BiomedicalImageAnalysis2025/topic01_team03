import numpy as np
import os
import sys

# src-Verzeichnis zum Pfad hinzufügen
project_root = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# otsu_threshold und binarize aus dem eigenen Modul importieren
from otsu_global import otsu_threshold

def local_otsu_binarize(img: np.ndarray, tile_size: int = 64) -> np.ndarray:
    """
    Führt lokale Otsu-Binarisierung durch, basierend auf eigener Otsu-Implementierung.
    Teilt das Bild in Tiles auf und berechnet für jedes Tile den Otsu-Schwellenwert.

    Parameters:
        img (np.ndarray): Eingabebild in Graustufen (dtype: uint8).
        tile_size (int): Kachelgröße für die lokale Segmentierung.

    Returns:
        np.ndarray: Binärbild (dtype=bool).
    """
    h, w = img.shape
    binary = np.zeros_like(img, dtype=bool)

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = img[y:y+tile_size, x:x+tile_size]
            if tile.size == 0:
                continue
            hist = np.histogram(tile, bins=256, range=(0, 256))[0]
            p = hist / np.sum(hist)
            t = otsu_threshold(p)
            binary[y:y+tile_size, x:x+tile_size] = tile > t

    return binary
