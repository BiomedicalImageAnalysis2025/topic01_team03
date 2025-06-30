#!/usr/bin/env python3
import numpy as np

def dice_score(otsu_img: np.ndarray, otsu_gt: np.ndarray) -> float:
    """
    Berechnet den Dice-Koeffizienten zwischen zwei binären Bildern.

    Args:
        pred: binäres Vorhersagebild (np.ndarray, dtype=bool)
        target: binäres Ground-Truth-Bild (np.ndarray, dtype=bool)

    Returns:
        Dice Score als float (0.0 bis 1.0)
    """
    # control if the Pictures have the same Size
    if len(otsu_img) != len(otsu_gt):
        raise ValueError("Images don't have the same length!")

    # defining the variables for the Dice Score equation for TRUE FALSE images
    sum_img = np.sum(otsu_img)
    sum_gt = np.sum(otsu_gt)
    positive_overlap = np.sum(np.logical_and(otsu_img, otsu_gt))

    if sum_img + sum_gt == 0:
        return 1.0

    return 2 * positive_overlap / (sum_img + sum_gt)

