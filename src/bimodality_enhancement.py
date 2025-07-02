# src/bimodality_enhancement.py

import numpy as np

def enhance_bimodality(img: np.ndarray) -> np.ndarray:
    """
    Enhances the bimodality of a grayscale image by applying adaptive gamma correction
    followed by histogram equalization. This process improves the separation between
    foreground and background intensities, making subsequent thresholding algorithms
    (e.g., Otsu) more effective.

    The function accepts input images in the range [0, 1] (floating point) or [0, 255]
    (integer), and returns an 8-bit image scaled to [0, 255].

    The enhancement steps are:
    1) Gamma correction: Increases contrast depending on the mean intensity of the image.
       - If the image is bright (mean intensity >= 0.5), gamma=3 darkens the bright regions.
       - If the image is dark, gamma=0.5 brightens the dark regions.
    2) Histogram equalization: Applies cumulative distribution function (CDF)-based
       equalization to spread the pixel intensities, enhancing the distinction between
       modes in a bimodal histogram.

    Args:
        img (np.ndarray): Input grayscale image. Expected to have pixel values in either
            the range [0, 1] or [0, 255].

    Returns:
        np.ndarray: Preprocessed image with enhanced bimodality, returned as a uint8
            array scaled to [0, 255].
    """
    # Ensure the image is in [0, 1] floating point representation
    if img.dtype != np.float32 and img.max() > 1.0:
        img = img / 255.0

    # Determine gamma value based on mean intensity
    mean_intensity = np.mean(img)
    gamma = 3.0 if mean_intensity >= 0.5 else 0.5

    # Apply gamma correction
    img_gamma = np.power(img, gamma)

    # Scale corrected image to 8-bit range
    img_8bit = (img_gamma * 255).astype(np.uint8)

    # Compute histogram and cumulative distribution function (CDF)
    hist, _ = np.histogram(img_8bit.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]  # Normalize CDF to [0, 255]

    # Map original pixel values through the normalized CDF
    img_eq = cdf_normalized[img_8bit].astype(np.uint8)

    return img_eq
