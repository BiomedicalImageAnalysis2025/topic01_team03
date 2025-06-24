# src/bimodality_enhancement.py

import numpy as np
from skimage import exposure

def enhance_bimodality(img: np.ndarray) -> np.ndarray:
    """
    Verbessert die Bimodalität eines Grauwertbildes durch Gamma-Korrektur
    und Histogramm-Equalisierung.

    Args:
        img (np.ndarray): Eingabebild, erwartet Wertebereich [0, 1] oder [0, 255].

    Returns:
        np.ndarray: Vorverarbeitetes Bild (dtype=np.uint8, Werte 0–255).
    """
    # Skaliere auf [0, 1], falls nötig
    if img.dtype != np.float32 and img.max() > 1.0:
        img = img / 255.0

    # Gamma abhängig von Helligkeit
    mean_intensity = np.mean(img)
    gamma = 3 if mean_intensity >= 0.5 else 0.5
    img_gamma = np.power(img, gamma)

    # Auf 8-Bit skalieren
    img_8bit = (img_gamma * 255).astype(np.uint8)

    # Histogramm-Equalisierung (CDF-based)
    hist, _ = np.histogram(img_8bit.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]  # CDF auf [0, 255] skalieren
    img_eq = cdf_normalized[img_8bit].astype(np.uint8)

    return img_eq
