import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Current folder as project_root (3 levels up)
# add project root
script_dir = os.getcwd()
project_root = os.path.abspath(script_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import importlib
import src.Dice_Score
importlib.reload(src.Dice_Score)

from src.pre_processing import gammacorrection
from src.Complete_Otsu_Global import otsu_threshold_skimage_like
from src.Dice_Score import dice_score

# Provided loader functions (unchanged)
from src.imread_all import (
    load_nih3t3_images,
    load_n2dl_hela_images,
    load_n2dh_gowt1_images,
)

def calculate_dice_scores_gamma_global(imgs, gts, gamma):
    """
    Process all images and corresponding ground truths to compute
    Dice scores at the specified gamma value.
    """
    dice_scores = []
    for img, gt in zip(imgs, gts):
        # Binarize groundtruth
        gt_bin = (gt > 0).astype(np.uint8)

        # Gamma correction
        img_gamma = gammacorrection(img, gamma=gamma)

        # Global Otsu thresholding
        t = otsu_threshold_skimage_like(img_gamma)
        binary1 = (img_gamma > t).astype(np.uint8)

        # Calculate Dice score
        score = dice_score(binary1.flatten(), gt_bin.flatten())
        dice_scores.append(score)

    return dice_scores

def evaluate_datasets_gamma(datasets, gamma_values, path):
    """
    Runs evaluation loop over given datasets and gamma sweep,
    combines all mean Dice scores into one 2D array
    (columns: datasets, rows: gamma values) and returns it.
    """
    all_means_list = []

    for name, loader in datasets:
        # Load images & ground truths
        imgs, gts, _, _ = loader(
            base_path=os.path.join(path, "data-git", name)
        )

        dice_means = []
        print(f"=== Processing {name} ===")
        for gamma in gamma_values:
            # Pass current gamma into the calculator
            scores = calculate_dice_scores_gamma_global(imgs, gts, gamma)
            mean_score = np.mean(scores)
            print(f"Gamma {gamma:.2f} → Mean Dice: {mean_score:.4f}")
            dice_means.append(mean_score)

        all_means_list.append(dice_means)

    # Stack per-dataset lists and transpose → shape (num_gammas, num_datasets)
    all_means = np.vstack(all_means_list).T
    return all_means

def find_best_gamma(all_means, gamma_values):
    """
    Finds the gamma that maximizes the average Dice score across datasets.
    """
    mean_across = np.mean(all_means, axis=1)
    best_idx    = np.argmax(mean_across)
    return gamma_values[best_idx], mean_across[best_idx]

def plot_mean_dice_vs_gamma(gamma_values, all_means, dataset_names, y):
    """
    Plot mean Dice score vs. gamma for each dataset.
    """
    plt.figure(figsize=(10, 6))
    for idx, name in enumerate(dataset_names):
        plt.plot(
            gamma_values,
            all_means[:, idx],
            marker='o',
            linestyle='-',
            label=name
        )
    plt.xlabel('Gamma value')
    plt.ylabel('Mean Dice score')
    plt.title(f'Mean Dice Score vs. Gamma, optimal gamma is {y}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()