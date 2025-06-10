import os
from skimage import io

def find_image_path(root_folder: str, filename: str) -> str:
    """
    Durchsucht root_folder (inkl. Unterordner) nach filename.
    Gibt bei Erfolg den absoluten Pfad zurück, sonst wird ein FileNotFoundError geworfen.
    """
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    raise FileNotFoundError(f"Datei '{filename}' wurde nicht unter '{root_folder}' gefunden.")

def load_image_gray(filename: str,
                    root_folder: str = "Data",
                    as_gray: bool = True):
    """
    Findet das Bild-Datei und lädt es als Graustufen-NumPy-Array.
    
    Params:
        filename: Name der Bilddatei, z.B. "dna-44.png"
        root_folder: Verzeichnis, in dem die Suche startet (default "Data")
        as_gray: ob das Bild als Graustufen geladen werden soll
    
    Returns:
        image_array: 2D-NumPy-Array mit den Grauwerten
    """
    path = find_image_path(root_folder, filename)
    return io.imread(path, as_gray=as_gray)