import os
from skimage import io

def find_image_path(root_folder: str, filename: str) -> str:

    """
    searches through root_folder (incl. subfolders) for filename.
    If succesful, returns full path of the file, otherwise raises FileNotFoundError.
    """

    for dirpath, dirnames, filenames in os.walk(root_folder):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    raise FileNotFoundError(f"file '{filename}' was not found under '{root_folder}'")


def load_image_gray(filename: str,
                    root_folder: str = "Data",
                    as_gray: bool = True):
    
    """
    Finds Image file and loads it as a 2D NumPy array with grayscale values. 
    
    Params:
        filename: Name of image-file, e.g. "dna-44.png"
        root_folder: Folder where image should be found (default "Data")
        as_gray: option if image should be loaded as grayscale (default True)
    
    Returns:
        image_array: 2D-numpy array with grayscale values

    Usage:
    image = load_image_gray("dna-44.png")
    """

    path = find_image_path(root_folder, filename)
    return io.imread(path, as_gray=as_gray)