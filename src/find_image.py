from pathlib import Path
from skimage import io
import numpy as np

def find_and_load_image(filename: str,
                        data_root: str = "data-git",
                        as_gray: bool = True) -> np.ndarray:
    """
    Searches the given folder (and its subfolders) for a file with the exact name,
    loads it as a grayscale image (if specified), and returns it as a NumPy array.
    
    Args:
      filename: name of the file to look for (e.g., "t01.tif" or "dna-44.png")
      data_root: base folder where the search starts (default is "data-git")
      as_gray: whether to load the image in grayscale (2D) or in color (3D)
  
    Returns:
      A 2D NumPy array containing the image data.
    """
    root = Path(data_root)
    # Search recursively for the file in all subfolders
    for candidate in root.rglob(filename):
        # If we reach this line, the file was found
        return io.imread(str(candidate), as_gray=as_gray)
    # If no file was found, raise an error
    raise FileNotFoundError(f"{filename!r} not found in {data_root!r}")