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

from src.Dice_Score_comparison import calculate_dice_scores_gamma_global


def calculate_dice_score_means_with_different_gamma(datasets, gamma_values):
    """
    Runs evaluation loop over given datasets and gamma sweep,
    combines all mean Dice scores into one 2D array
    (columns: datasets, rows: gamma values) and returns it.
    """
    all_means_list = []

    for name, loader in datasets:
        # Load images & ground truths
        imgs, gts, _, _ = loader(
            base_path=os.path.join(project_root, "data-git", name)
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

def plot_mean_dice_vs_gamma(gamma_values, all_means, dataset_names):
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
    plt.title('Mean Dice Score vs. Gamma')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()