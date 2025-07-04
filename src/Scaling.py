import numpy as np

def scale_uint8(img: np.ndarray) -> np.ndarray:
    """
    Normalizes a grayscale image to the full 0–255 range and converts it to uint8.

    Parameters:
        img (np.ndarray): Input 2D grayscale image (float or int).

    Returns:
        np.ndarray: Normalized image as uint8 (values in [0, 255]).
    """
    img_min, img_max = np.min(img), np.max(img)
    if img_max == img_min:
        return np.zeros_like(img, dtype=np.uint8)  # avoid division by zero
    img_norm = (img - img_min) / (img_max - img_min)
    img_uint8 = (img_norm * 255).astype(np.uint8)
    return img_uint8