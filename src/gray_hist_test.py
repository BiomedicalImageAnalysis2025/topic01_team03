from typing import Union, Tuple, Optional
import numpy as np
from PIL import Image
from pathlib import Path

def compute_gray_histogram(
    image_source: Union[Path, str, np.ndarray],
    bins: int = 256,
    value_range: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads an image (from file path or NumPy array), converts it to grayscale,
    and computes the histogram with the specified number of bins.

    If value_range is None, the histogram range is set automatically
    to (min, max) of the image, ensuring correct thresholding even for
    images outside the standard 0-255 8-bit range (e.g., float images [0,1] or 16-bit images).

    Args:
        image_source: Path (Path/str) to the image file OR a 2D NumPy array of grayscale values.
        bins: Number of bins for the histogram.
        value_range: Tuple specifying the value range (min, max). If None, the range
                     will be determined dynamically from the image data.

    Returns:
        hist: Array with the frequency of pixels in each bin.
        bin_edges: Edges of the histogram bins.
    """
    # Load image: path or NumPy array
    if isinstance(image_source, (Path, str)):
        img = Image.open(str(image_source)).convert("L")
        arr = np.array(img)
    elif isinstance(image_source, np.ndarray):
        arr = image_source
    else:
        raise TypeError(
            "compute_gray_histogram expects a file path (Path/str) or a NumPy array."
        )

    # Determine value range automatically if not explicitly provided
    if value_range is None:
        min_val, max_val = arr.min(), arr.max()
    else:
        min_val, max_val = value_range

    # Compute histogram
    hist, bin_edges = np.histogram(
        arr.ravel(),
        bins=bins,
        range=(min_val, max_val)
    )
    return hist, bin_edges