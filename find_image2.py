import os
from pathlib import Path
from skimage import io
import numpy as np

print("Working Directory:", os.getcwd())


def find_image_in_data_git(filename: str, 
                           data_root: str = "data-git") -> Path:
    """
    Durchsucht data-git und genau dessen drei Unterordner (N2DH-GOWT1, N2DL-HeLa, NIH3T3)
    nach einer Datei `filename`. Gibt den vollständigen Path zurück oder wirft
    FileNotFoundError, falls das Bild nicht gefunden wird.
    """
    root = Path(data_root)
    # Liste der drei erwarteten Unterordner
    subdirs = ["N2DH-GOWT1", "N2DL-HeLa", "NIH3T3"]
    
    for sd in subdirs:
        img_dir = root / sd / "img"
        if not img_dir.is_dir():
            continue
        # Suche nur in img/ des jeweiligen Unterordners
        candidate = img_dir / filename
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"{filename!r} nicht gefunden in '{data_root}'/…/img/")

def load_image_gray(filename: str,
                            data_root: str = "data-git",
                            as_gray: bool = True) -> np.ndarray:
    """
    Findet das Bild über find_image_in_data_git und lädt es als Graustufen-Array.
    """
    img_path = find_image_in_data_git(filename, data_root)
    return io.imread(str(img_path), as_gray=as_gray)

