import os
import numpy as np
import pandas as pd
import skimage as io  # oder skimage.io, wenn installiert

# Pfad zum Ordner, der die drei Datensätze enthält
# z.B. wenn intensity_test.py im gleichen Ordner wie data-git liegt:
base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(base_dir, "data-git")

results = []

for dataset in os.listdir(root_dir):
    ds_path = os.path.join(root_dir, dataset)
    img_dir = os.path.join(ds_path, "img")  # Nur hier drin liegen die Bilder
    if os.path.isdir(img_dir):
        intensities = []
        for fname in os.listdir(img_dir):
            if fname.lower().endswith((".png", ".jpg", ".tif", ".tiff")):
                img_path = os.path.join(img_dir, fname)
                img = imageio.v2.imread(img_path)
                # Falls farbig, in Graustufen umwandeln
                if img.ndim == 3:
                    img = img.mean(axis=-1)
                intensities.append(img.mean())
        if intensities:
            results.append({
                "Dataset": dataset,
                "Anzahl Bilder": len(intensities),
                "Mittelwert Intensität": np.mean(intensities),
                "Median Intensität": np.median(intensities),
                "StdDev Intensität": np.std(intensities),
            })

df = pd.DataFrame(results)
print(df.to_string(index=False))
