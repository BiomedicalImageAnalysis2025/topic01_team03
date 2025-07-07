import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Tuple
import seaborn as sns
import pandas as pd

def scatterplot_without_regression(
    our_scores, 
    package_scores, 
    xlabel="Score 1 (ours)", 
    ylabel="Score 2 (reference)", 
    title="Title",
    label_red="Package > Ours",
    label_green="Ours > Package",
    label_blue="Scores ~ equal"
):
    """
    Displays a scatterplot to visually compare two equally sized score vectors.

    Each point is color-coded based on whether it favors the first, the second,
    or shows similar values. A diagonal reference line is added, along with a custom
    legend using embedded markers and labels.
    """
    our_scores = np.array(our_scores)
    package_scores = np.array(package_scores)
    
    if len(our_scores) != len(package_scores):
        raise ValueError("Both score vectors must be of equal length.")
    
    slope, intercept = np.polyfit(our_scores, package_scores, 1)
    
    colors = []
    for x, y in zip(our_scores, package_scores):
        if np.isclose(y, x):
            colors.append("blue")
        elif y > x:
            colors.append("red")
        else:
            colors.append("green")
    
    df = pd.DataFrame({
        "Score 1": our_scores,
        "Score 2": package_scores,
        "Color": colors
    })
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="Score 1",
        y="Score 2",
        hue="Color",
        palette={"red": "red", "green": "green", "blue": "blue"},
        legend=False,
        s=50
    )
    
    min_val, max_val = min(np.min(our_scores), np.min(package_scores)), max(np.max(our_scores), np.max(package_scores))
    sns.lineplot(x=[min_val, max_val], y=[min_val, max_val], color="black", linestyle="--", label="y = x")
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    # Add explanatory dummy points for legend
    x_legend = min_val + 0.05 * (max_val - min_val)
    y_base = min_val - 0.1 * (max_val - min_val)

    plt.scatter([x_legend], [y_base], color="red", s=50)
    plt.text(x_legend + 0.02, y_base, label_red, va="center", fontsize=10)

    plt.scatter([x_legend + 0.2], [y_base], color="green", s=50)
    plt.text(x_legend + 0.22, y_base, label_green, va="center", fontsize=10)

    plt.scatter([x_legend + 0.4], [y_base], color="blue", s=50)
    plt.text(x_legend + 0.42, y_base, label_blue, va="center", fontsize=10)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_pairwise_scores(
    x_positions: Sequence,
    scores1: Sequence,
    scores2: Sequence,
    labels: Sequence[str],
    title: str = "Comparison of Methods",
    xlabel: str = "Files",
    ylabel: str = "Dice Score",
    legend_labels: tuple[str, str] = ("Method 1", "Method 2"),
    figsize: tuple[int, int] = (10, 6)
):
    """
    Generates a paired comparison plot for two score vectors.

    Each data point pair is shown with a connecting line, highlighting
    differences on a per-sample basis.
    """
    plt.figure(figsize=figsize)
    plt.scatter(x_positions, scores1, color='C0', label=legend_labels[0])
    plt.scatter(x_positions, scores2, color='C1', label=legend_labels[1])

    for xi, y1, y2 in zip(x_positions, scores1, scores2):
        plt.plot([xi, xi], [y1, y2], color='gray', alpha=0.5)

    plt.xticks(x_positions, labels, rotation=45, ha='right')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_triplet_scores(
    x_positions: Sequence,
    scores1: Sequence,
    scores2: Sequence,
    scores3: Sequence,
    labels: Sequence[str],
    title: str = "Comparison of 3 Methods",
    xlabel: str = "Files",
    ylabel: str = "Dice Score",
    legend_labels: Tuple[str, str, str] = ("Method 1", "Method 2", "Method 3"),
    markers: Tuple[str, str, str] = ("o", "s", "^"),
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Visualizes three sets of scores using distinct markers and connecting lines per item.

    Useful for direct visual comparison across three methods or configurations.
    """
    plt.figure(figsize=figsize)
    plt.scatter(x_positions, scores1, marker=markers[0], color='C0', label=legend_labels[0])
    plt.scatter(x_positions, scores2, marker=markers[1], color='C1', label=legend_labels[1])
    plt.scatter(x_positions, scores3, marker=markers[2], color='C2', label=legend_labels[2])

    for xi, y1, y2, y3 in zip(x_positions, scores1, scores2, scores3):
        plt.plot([xi, xi, xi], [y1, y2, y3], color='gray', alpha=0.5)

    plt.xticks(x_positions, labels, rotation=45, ha='right')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_quadruplet_scores(
    x_positions: Sequence,
    scores1: Sequence,
    scores2: Sequence,
    scores3: Sequence,
    scores4: Sequence,
    labels: Sequence[str],
    title: str = "Comparison of 4 Methods",
    xlabel: str = "Files",
    ylabel: str = "Dice Score",
    legend_labels: Tuple[str, str, str, str] = ("Method 1", "Method 2", "Method 3", "Method 4"),
    markers: Tuple[str, str, str, str] = ("o", "s", "^", "D"),
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Displays four sets of scores with distinct markers and connecting lines.

    Facilitates direct sample-wise comparison across four different methods.
    """
    plt.figure(figsize=figsize)
    plt.scatter(x_positions, scores1, marker=markers[0], color='C0', label=legend_labels[0])
    plt.scatter(x_positions, scores2, marker=markers[2], color='C0', label=legend_labels[1])
    plt.scatter(x_positions, scores3, marker=markers[0], color='C1', label=legend_labels[2])
    plt.scatter(x_positions, scores4, marker=markers[2], color='C1', label=legend_labels[3])

    for xi, y1, y2, y3, y4 in zip(x_positions, scores1, scores2, scores3, scores4):
        plt.plot([xi]*4, [y1, y2, y3, y4], color='gray', alpha=0.5)

    plt.xticks(x_positions, labels, rotation=45, ha='right')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_gray_histogram(image: np.ndarray, bins: int = 256, title: str = "Gray Value Histogram") -> None:
    """
    Displays a grayscale histogram for a 2D image.

    The histogram shows the distribution of pixel intensities across the entire image.
    """
    if image.ndim != 2:
        raise ValueError("Input must be a 2D grayscale image.")

    plt.figure(figsize=(8, 5))
    plt.hist(image.ravel(), bins=bins, range=[0, 255], color='gray', edgecolor='black')
    plt.title(title)
    plt.xlabel("Gray Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_single_grouped_boxplot(df, title, ylabel, palette="pastel", figsize=(8,6), save_path=None):
    """
    Creates a grouped boxplot for multiple datasets, showing individual points overlaid.

    Useful for visualizing score distributions across different conditions or datasets.
    """
    plt.figure(figsize=figsize)

    sns.boxplot(
        x="Dataset",
        y="Value",
        hue="Dataset",
        data=df,
        palette=palette,
        showfliers=True,
        legend=False
    )
    sns.stripplot(
        x="Dataset",
        y="Value",
        data=df,
        color="gray",
        alpha=0.6,
        jitter=True,
        size=4
    )

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()


def combine_results_to_dataframe(names, lists):
    """
    Combines multiple result lists into a single DataFrame for grouped analysis or plotting.

    Each value is assigned a dataset label, resulting in a long-form table with two columns:
    'Dataset' and 'Value'.
    """
    if len(names) != len(lists):
        raise ValueError("Lengths of 'names' and 'lists' must match.")

    dfs = []
    for name, values in zip(names, lists):
        df = pd.DataFrame({
            "Dataset": [name] * len(values),
            "Value": values
        })
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)
