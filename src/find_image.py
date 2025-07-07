from pathlib import Path
from skimage import io
import numpy as np

def find_and_load_image(filename: str,
                        data_root: str = "data-git",
                        as_gray: bool = True) -> np.ndarray:
    """
    Durchsucht data_root rekursiv nach genau filename,
    lädt es als Graustufen‐Array und gibt es zurück.
    
    Args:
      filename: z.B. "t01.tif" oder "dna-44.png"
      data_root: Basis‐Ordner, default = "data-git"
      as_gray:   ob als 2D Gray‐Array geladen wird
  
    Returns:
      2D numpy‐Array mit Werten 0..1 (float) oder 0..255 (uint8, je nach io.imread).
    """
    root = Path(data_root)
    # rglob sucht rekursiv in allen Unterordnern:
    for candidate in root.rglob(filename):
        # wenn wir hier ankommen, existiert die Datei:
        return io.imread(str(candidate), as_gray=as_gray)
    # kein Treffer:
    raise FileNotFoundError(f"{filename!r} nicht gefunden unter {data_root!r}")

# Beispielnutzung:
#if __name__ == "__main__":
#    img = find_and_load_image("t01.tif")
#    print("Shape:", img.shape, "Min/Max:", img.min(), img.max())