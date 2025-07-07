import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import sys

# Add the current working directory to the Python path to enable local imports
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
    Determines the optimal radius for local Otsu thresholding across multiple datasets.

    For each image in the selected datasets, all specified radii are evaluated, and the one 
    yielding the highest Dice score is recorded. Optionally, results can be saved to a CSV file.

    Parameters:
        datasets (list of tuples): Optional list of (dataset_name, loader_function).
                                   If None, predefined datasets will be used.
        radii (list of int): List of radius values to evaluate. Defaults to odd values up to 49.
        save_csv_path (str): Optional path to save the results as a CSV file.

    Returns:
        List of dictionaries containing the best radius and corresponding Dice score for each image.
    """
    if radii is None:
        radii = list(range(1, 3, 2))  # Default: [1]

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
                    print(f"[WARN] Error processing image {img_path} with radius {r}: {e}")
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

    # Display results in console
    for result in all_results:
        print(f"{result['Dataset']}/{result['Image']}: best radius = {result['BestRadius']}, Dice = {result['BestDice']:.4f}")

    # Optionally export results to CSV
    if save_csv_path:
        df = pd.DataFrame(all_results)
        df.to_csv(save_csv_path, index=False)
        print(f"\nResults saved to {save_csv_path}")

    return all_results


def calculate_best_radii_and_dice1(imgs, gts, radii):
    """
    Identifies the optimal radius per image by computing Dice scores 
    over all specified radii and selecting the best-performing one.

    Parameters:
        imgs (list of np.ndarray): Input images.
        gts (list of np.ndarray): Corresponding ground truth masks.
        radii (list of int): Radius values to evaluate.

    Returns:
        Tuple containing two lists:
            - Best radius per image.
            - Corresponding best Dice score.
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
                print(f"[WARN] Error with radius {r}: {e}")
                continue

            if dice > best_dice:
                best_dice = dice
                best_radius = r

        results_dice.append(best_dice)
        results_radius.append(best_radius)

    return results_radius, results_dice


def print_clean_results(all_results, all_image_paths): 
    """
    Organizes radius and Dice score results into a structured dictionary per dataset.

    Parameters:
        all_results (list of tuples): Each tuple contains best radii and scores per dataset.
        all_image_paths (list): Corresponding file paths for all images.

    Returns:
        Dictionary grouping results by dataset, with tuples of (filename, best radius, Dice score).
    """
    from src.imread_all import load_n2dh_gowt1_images, load_n2dl_hela_images, load_nih3t3_images

    dataset_name = ["N2DH-GOWT1", "N2DL-HeLa", "NIH3T3"]
    cleanresult = {}

    for dataset, (rad, di), img in zip(dataset_name, all_results, all_image_paths):
        for name, radius, dice in zip(img, rad, di):
            filename = os.path.basename(name)
            cleanresult.setdefault(dataset, []).append((filename, radius, dice))

    return cleanresult