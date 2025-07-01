# src/Complete_Otsu_Global.py

import numpy as np
from typing import Tuple

# creating histogram for Otsu threshholding  
def custom_histogram(image: np.ndarray, nbins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the histogram and corresponding bin centers of a grayscale image,
    replicating the behavior of skimage.exposure.histogram, including normalization
    to the [0, 255] range. This ensures consistent behavior with Otsu implementations
    that assume 8-bit images.

    Args:
        image (np.ndarray): Input image as a 2D array of grayscale values.
        nbins (int): Number of bins for the histogram (default: 256).

    Returns:
        hist (np.ndarray): Array of histogram frequencies for each bin.
        bin_centers (np.ndarray): Array of bin center values.
    """
    # Determine the minimum and maximum pixel intensity in the image
    img_min, img_max = image.min(), image.max()

    # Normalize the image intensities to the range [0, 255], as in skimage
    image_scaled = (image - img_min) / (img_max - img_min) * 255

    # Compute the histogram of the scaled image within [0, 255]
    hist, bin_edges = np.histogram(
        image_scaled.ravel(),
        bins=nbins,
        range=(0, 255)
    )

    # Compute bin centers as the average of adjacent bin edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return hist, bin_centers


def otsu_threshold_skimage_like(image: np.ndarray) -> float:
    """
    Computes the global Otsu threshold of an input grayscale image in a way that matches
    the behavior of skimage.filters.threshold_otsu, including histogram scaling and
    threshold rescaling back to the original intensity range.

    This function enables nearly identical thresholding results to skimage's implementation,
    even on images with floating-point or non-8-bit integer data.

    Args:
        image (np.ndarray): Input image as a 2D array of grayscale values.

    Returns:
        threshold_original (float): Computed Otsu threshold mapped back to the original image range.
    """
    # Compute histogram and bin centers consistent with skimage
    hist, bin_centers = custom_histogram(image, nbins=256)
    hist = hist.astype(np.float64)

    # Normalize histogram to obtain probability distribution p(k)
    p = hist / hist.sum()

    # Compute cumulative sums of class probabilities ω0 and ω1
    omega0 = np.cumsum(p)                           # Class probabilities for background
    omega1 = np.cumsum(p[::-1])[::-1]               # Class probabilities for foreground

    # Compute cumulative sums of class means μ0 and μ1
    mu0 = np.cumsum(p * bin_centers)                # Class means for background
    mu1 = np.cumsum((p * bin_centers)[::-1])[::-1]  # Class means for foreground

    # Compute between-class variance σ_b^2 for each possible threshold
    sigma_b_squared = (omega0[:-1] * omega1[1:] * (mu0[:-1] / omega0[:-1] - mu1[1:] / omega1[1:])**2)

    # Find the threshold index t maximizing σ_b^2
    t_idx = np.argmax(sigma_b_squared)
    t_scaled = bin_centers[t_idx]

    # Rescale threshold t back to original image intensity range
    img_min, img_max = image.min(), image.max()
    t_original = t_scaled / 255 * (img_max - img_min) + img_min

    return t_original
