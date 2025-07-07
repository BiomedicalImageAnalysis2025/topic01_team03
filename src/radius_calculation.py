import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import sys

# add project root
script_dir = os.getcwd()
project_root = os.path.abspath(script_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.Dice_Score import dice_score
from src.Otsu_Local import local_otsu_package

def find_best_radius_for_datasets_package(
    datasets=None,
    radii=None,
    save_csv_path=None
):
    """
    Berechnet für alle Bilder der angegebenen Datensätze den besten lokalen Otsu-Radius.

    Args:
        datasets (list of tuples, optional): Liste von (dataset_name, loader_funktion).
            Wenn None, werden NIH3T3, N2DH-GOWT1, N2DL-HeLa geladen.
        radii (list of int, optional): Liste der zu testenden Radien. Defaults to [1,3,...,49].
        save_csv_path (str, optional): Falls angegeben, wird eine CSV mit den Ergebnissen gespeichert.

    Returns:
        list of dict: Liste mit Ergebnissen für alle Bilder (Dataset, Image, BestRadius, BestDice).
    """
    if radii is None:
        radii = list(range(1, 3, 2))

    if datasets is None:
        from src.imread_all import load_n2dh_gowt1_images, load_n2dl_hela_images, load_nih3t3_images
        datasets = [
            ("NIH3T3", load_nih3t3_images),
            ("N2DH-GOWT1", load_n2dh_gowt1_images),
            ("N2DL-HeLa", load_n2dl_hela_images),
        ]

    all_results = []

    for dataset_name, loader in datasets:
        print(f"\n--- Processing dataset: {dataset_name} ---")
        imgs, gts, img_paths, gt_paths = loader()

        for img, gt, img_path in tqdm(zip(imgs, gts, img_paths), total=len(imgs), desc=f"Images in {dataset_name}"):
            best_dice = -1
            best_radius = None

            for r in radii:
                try:
                    local_thresh = local_otsu_package(img, radius=r)
                    mask = img > local_thresh
                    dice = dice_score(mask, gt)
                except Exception as e:
                    print(f"[WARN] Fehler bei Bild {img_path} mit Radius {r}: {e}")
                    continue

                if dice > best_dice:
                    best_dice = dice
                    best_radius = r

            all_results.append({
                "Dataset": dataset_name,
                "Image": os.path.basename(img_path),
                "BestRadius": best_radius,
                "BestDice": best_dice
            })

    # Ergebnisse ausgeben
    for result in all_results:
        print(f"{result['Dataset']}/{result['Image']}: bester Radius = {result['BestRadius']}, Dice={result['BestDice']:.4f}")

    # Optional speichern
    if save_csv_path:
        df = pd.DataFrame(all_results)
        df.to_csv(save_csv_path, index=False)
        print(f"\nErgebnisse gespeichert in {save_csv_path}")

    return all_results

import numpy as np
from tqdm import tqdm

def calculate_best_radii_and_dice1(imgs, gts, radii):
    """
    Für jedes Bild: teste alle angegebenen Radien und bestimme den besten Dice-Score samt Radius.

    Args:
        imgs (list of np.ndarray): Liste von Input-Bildern.
        gts (list of np.ndarray): Liste der Ground-Truth-Bilder.
        radii (list of int): Liste der zu testenden Radien.

    Returns:
        list of dict: Pro Bild ein Dict mit 'BestRadius' und 'BestDice'.
    """
    from src.Dice_Score import dice_score
    from src.Otsu_Local import local_otsu_package

    results_dice = []
    results_radius = []
    for img, gt in tqdm(zip(imgs, gts), total=len(imgs), desc="Processing images"):
        best_dice = -1
        best_radius = None

        for r in radii:
            try:
                local_thresh = local_otsu_package(img, radius=r)
                mask = img > local_thresh
                dice = dice_score(mask, gt)
            except Exception as e:
                print(f"[WARN] Fehler bei Radius {r}: {e}")
                continue

            if dice > best_dice:
                best_dice = dice
                best_radius = r

        results_dice.append(best_dice)

        results_radius.append(best_radius)

    return results_radius, results_dice

def print_clean_results(all_results, all_image_paths): 

    from src.imread_all import load_n2dh_gowt1_images, load_n2dl_hela_images, load_nih3t3_images

    dataset_name = ["N2DH-GOWT1", "N2DL-HeLa", "NIH3T3"]

    cleanresult = {}

    for dataset, (rad, di), img in zip(dataset_name, all_results, all_image_paths):
        for name, radius, dice in zip(img, rad, di):
            filename = os.path.basename(name)
            cleanresult.setdefault(dataset, []).append((filename, radius, dice))

    return cleanresult 