import os
import sys
import numpy as np
from pathlib import Path
from skimage.io import imread
import matplotlib.pyplot as plt
from typing import Sequence, Tuple

# Current folder as project_root
# add project root
script_dir = os.getcwd()
project_root = os.path.abspath(script_dir)

import importlib
import src.Dice_Score
importlib.reload(src.Dice_Score)


from src.Complete_Otsu_Global import otsu_threshold_skimage_like
from src.Dice_Score import dice_score
from src.pre_processing import mean_filter  # ensure mean_filter is imported
from src.imread_all import (
    load_nih3t3_images,
    load_n2dl_hela_images,
    load_n2dh_gowt1_images,
)

def calculate_dice_scores_meanfilter_global(imgs, gts, s):
    """
    Process all images and corresponding ground truths to compute a list of Dice scores.

    Args:
        imgs (list of np.ndarray): Grayscale input images.
        gts (list of np.ndarray): Corresponding ground-truth masks.

    Returns:
        dice_scores (list of float): Dice scores for each image-groundtruth pair.
    """
    dice_scores = []

    for img, gt in zip(imgs, gts):
        
        # binarize groundtruth
        gt_bin = 1 - ((gt / gt.max()) == 0)

        # mean filter
        img_filtered = mean_filter(img, kernel_size=s)

        # global otsu thresholding
        t = otsu_threshold_skimage_like(img_filtered)
        binary1 = (img_filtered > t)
        # calculate dice score
        score = dice_score(binary1.flatten(), gt_bin.flatten())
        dice_scores.append(score)

    return dice_scores

# Generic evaluation across datasets and window sizes
def evaluate_datasets_mean_filter(
    datasets, window_sizes: np.ndarray, output_dir: Path = Path('Dice_scores'), project_root=None
) -> dict:
    """
    Runs evaluation for multiple datasets and window sizes,
    returns dictionary of mean Dice arrays keyed by dataset name.
    """
    output_dir.mkdir(exist_ok=True)
    results = {}

    for name, loader in datasets:
        imgs, gts, img_paths, gt_paths = loader(base_path=os.path.join(project_root, 'data-git', name))
        means = []
        print(f"=== Processing {name} ===")
        for s in window_sizes:

         scores = calculate_dice_scores_meanfilter_global(imgs, gts, s)
         mean_score = float(np.mean(scores))
         print(f"window {s:2d} → Mean Dice: {mean_score:.4f}")
         means.append(mean_score)
        results[name] = np.array(means)
        np.save(output_dir / f"{name}_dice_means_meanfilter.npy", results[name])
        print(f"Saved {name} results to {output_dir / f'{name}_dice_means_meanfilter.npy'}\n")

    # Combine into table: rows=window_sizes, cols=datasets order
    table = np.vstack([results[name] for name, _ in datasets]).T
    np.save(output_dir / "all_datasets_dice_means_meanfilter.npy", table)
    print(f"Saved combined table to {output_dir / 'all_datasets_dice_means_meanfilter.npy'}")
    return results


def find_best_window(all_means: np.ndarray,
                     window_sizes: Sequence[int]
                    ) -> Tuple[int, float]:
    """
    Finds the window size that maximizes the average Dice score across datasets.

    Args:
        all_means: 2D array of shape (num_window_sizes, num_datasets),
                   as returned by evaluate_datasets_mean_filter (rows = window sizes).
        window_sizes: 1D sequence of window sizes corresponding to the rows of all_means.

    Returns:
        best_window: the window size with the highest mean Dice score.
        best_score: the mean Dice score at best_window.
    """
    # compute mean across datasets for each window size (mean of each row)
    mean_across = np.mean(all_means, axis=1)
    # find the index of the maximum mean
    best_idx = int(np.argmax(mean_across))
    # map back to the window size
    best_window = window_sizes[best_idx]
    best_score = float(mean_across[best_idx])
    return best_window, best_score

import matplotlib.pyplot as plt
from typing import Sequence

def plot_all_datasets_means(
    all_means: np.ndarray,
    param_values: Sequence[float],
    dataset_names: Sequence[str],
    xlabel: str = "Parameter",
    ylabel: str = "Mean Dice Score",
    title: str = "Mean Dice vs. Parameter",
    figsize: tuple[int, int] = (10, 6)
) -> None:
    """
    Plot each column of a 2D mean-Dice array as a separate curve.

    Args:
        all_means: 2D array of shape (len(param_values), len(dataset_names))
        param_values: 1D sequence of parameter values (rows of all_means)
        dataset_names: list of names, one per column in all_means
        xlabel: label for the x-axis
        ylabel: label for the y-axis
        title: plot title
        figsize: figure size in inches
    """
    plt.figure(figsize=figsize)
    n_datasets = all_means.shape[1]
    for idx in range(n_datasets):
        plt.plot(
            param_values,
            all_means[:, idx],
            marker='o',
            linestyle='-',
            label=dataset_names[idx]
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()