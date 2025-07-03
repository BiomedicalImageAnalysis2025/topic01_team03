# src/gray_hist.py

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
from typing import Union, Tuple

def compute_gray_histogram(
    image_source: Union[Path, str, np.ndarray],
    bins: int = 256,
    value_range: Tuple[int, int] = (0, 255)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads an image (from a file path or directly as a NumPy array), converts it to grayscale if necessary,
    and computes its histogram over the specified value range.

    This function allows for direct analysis of image intensity distributions, which can be
    essential for thresholding algorithms and quantitative assessments of grayscale images.

    Args:
        image_source (Path | str | np.ndarray): Either a file path to the image or a 2D NumPy array containing
                                               grayscale pixel values.
        bins (int, optional): Number of bins to use when computing the histogram (default: 256).
        value_range (Tuple[int, int], optional): Value range to cover in the histogram, typically [0, 255] 
                                                 for 8-bit grayscale images (default: (0, 255)).

    Returns:
        hist (np.ndarray): The computed histogram, containing pixel frequencies per bin.
        bin_edges (np.ndarray): The edges of the histogram bins, which can be used for plotting.

    Raises:
        TypeError: If `image_source` is neither a file path nor a NumPy array.
    """
    # Recognize input type and convert to grayscale array if necessary
    if isinstance(image_source, (Path, str)):
        img = Image.open(str(image_source)).convert("L")
        arr = np.array(img)
    elif isinstance(image_source, np.ndarray):
        arr = image_source
    else:
        raise TypeError(
            "compute_gray_histogram expects a file path (Path or str) or a NumPy array as input."
        )

    # Compute histogram over the flattened grayscale array
    hist, bin_edges = np.histogram(
        arr.ravel(),
        bins=bins,
        range=value_range
    )
    return hist, bin_edges


def plot_gray_histogram(hist: np.ndarray, bin_edges: np.ndarray):
    """
    Plots a grayscale histogram using the provided histogram frequencies and bin edges.

    This visualization enables intuitive assessment of pixel intensity distributions,
    which is particularly useful for evaluating image contrast and thresholding behavior.

    Args:
        hist (np.ndarray): Histogram frequencies per bin.
        bin_edges (np.ndarray): The edges of the histogram bins, matching the output from compute_gray_histogram().
    """
    plt.figure(figsize=(8, 4))
    plt.bar(
        bin_edges[:-1],
        hist,
        width=bin_edges[1] - bin_edges[0],
        align='edge'
    )
    plt.xlabel("Pixel intensity")
    plt.ylabel("Frequency")
    plt.title("Grayscale Histogram")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()