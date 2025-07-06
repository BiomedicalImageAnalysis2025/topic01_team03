import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence
import seaborn as sns
import pandas as pd

def scatterplot_with_regression(
    our_scores, 
    package_scores, 
    xlabel="Score 1 (ours)", 
    ylabel="Score 2 (reference)", 
    title="Title"
):
    """
    Vergleicht zwei gleichgroße Vektoren mit Dice-Scores durch Scatterplot + Regression.
    
    - our_scores: Liste oder numpy-Array mit unseren Dice-Scores
    - package_scores: Liste oder numpy-Array mit Referenz-Dice-Scores
    - xlabel: Beschriftung für die x-Achse
    - ylabel: Beschriftung für die y-Achse
    - title: Plot-Titel
    """
    our_scores = np.array(our_scores)
    package_scores = np.array(package_scores)
    
    if len(our_scores) != len(package_scores):
        raise ValueError("Beide Vektoren müssen gleich lang sein!")
    
    # Regression
    slope, intercept = np.polyfit(our_scores, package_scores, 1)
    
    # Farben basierend auf Vergleich
    colors = []
    for x, y in zip(our_scores, package_scores):
        if np.isclose(y, x):
            colors.append("blue")
        elif y > x:
            colors.append("red")
        else:
            colors.append("green")
    
    # DataFrame für seaborn
    df = pd.DataFrame({
        "Score 1": our_scores,
        "Score 2": package_scores,
        "Color": colors
    })
    
    # Scatterplot
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
    
    # Regressionslinie
    x_fit = np.linspace(min(our_scores), max(our_scores), 100)
    y_fit = slope * x_fit + intercept
    sns.lineplot(x=x_fit, y=y_fit, color="orange", label="Regression", linestyle="-")
    
    # Identitätslinie y=x
    min_val, max_val = min(np.min(our_scores), np.min(package_scores)), max(np.max(our_scores), np.max(package_scores))
    sns.lineplot(x=[min_val, max_val], y=[min_val, max_val], color="black", linestyle="--", label="y = x")
    
    # Manuelle Achsenbeschriftung und Titel
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    # Sonstiges
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.show()

#------------------------------------------------------------------------------------------------
# example for usage
#a = [0.5705017182130584, 0.32258217915948406, 0.568002229254991, 0.5830196570472606]
#b = [0.28915016099131624, 0.16063829075758015, 0.22105149983872702, 0.2231858373710903]

#slope, intercept = compare_scores_with_regression(
#    a, 
#    b, 
#    xlabel="Dice Score of Our Otsu Local",
#    ylabel="Dice Score of Package Otsu Local",
#    title="Comparison: Package Otsu Local vs. Our Otsu Local (radius=15)"
#)
#------------------------------------------------------------------------------------------------
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
    Vergleicht zwei gleichgroße Vektoren mit Dice-Scores durch Scatterplot + optionale Beschreibung.
    
    Zusätzlich werden Dummy-Punkte mit Erklärungen direkt im Plot angezeigt.
    """
    our_scores = np.array(our_scores)
    package_scores = np.array(package_scores)
    
    if len(our_scores) != len(package_scores):
        raise ValueError("Beide Vektoren müssen gleich lang sein!")
    
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
    
    # Dummy-Punkte und Erklärung unter der Legende:
    x_legend = min_val + 0.05 * (max_val - min_val)
    y_base = min_val - 0.1 * (max_val - min_val)  # Etwas unterhalb des Plots
    
    plt.scatter([x_legend], [y_base], color="red", s=50)
    plt.text(x_legend + 0.02, y_base, label_red, va="center", fontsize=10)
    
    plt.scatter([x_legend + 0.2], [y_base], color="green", s=50)
    plt.text(x_legend + 0.22, y_base, label_green, va="center", fontsize=10)
    
    plt.scatter([x_legend + 0.4], [y_base], color="blue", s=50)
    plt.text(x_legend + 0.42, y_base, label_blue, va="center", fontsize=10)
    
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# SPAGHETTI PLOT WITH 2 VECTORS
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
    Plot paired scores with connecting lines for comparison.

    Parameters:
    - x_positions: positions along x-axis (e.g., [0, 1, 2, ...])
    - scores1: first set of scores (e.g., without mean filter)
    - scores2: second set of scores (e.g., with mean filter)
    - labels: tick labels for x-axis (e.g., file names)
    - file_labels: repeated list of same labels, if needed (optional)
    - title, xlabel, ylabel: plot text labels
    - legend_labels: labels shown in legend for the two datasets
    - figsize: figure size in inches
    """
    
    plt.figure(figsize=figsize)
    plt.scatter(x_positions, scores1, color='C0', label=legend_labels[0])
    plt.scatter(x_positions, scores2, color='C1', label=legend_labels[1])

    # Connect each pair of points
    for xi, y1, y2 in zip(x_positions, scores1, scores2):
        plt.plot([xi, xi], [y1, y2], color='gray', alpha=0.5)

    plt.xticks(x_positions, labels, rotation=45, ha='right')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()


#SPAGHETTI PLOT WITH 3 VECTORS

def plot_triplet_scores(
    x_positions: Sequence,
    scores1: Sequence,
    scores2: Sequence,
    scores3: Sequence,
    labels: Sequence[str],
    title: str = "Comparison of 3 Methods",
    xlabel: str = "Files",
    ylabel: str = "Dice Score",
    legend_labels: tuple[str, str, str] = ("Method 1", "Method 2", "Method 3"),
    figsize: tuple[int, int] = (10, 6)
):
    """
    Plot triplet scores with connecting lines for each file.

    Parameters:
    - x_positions: x-axis positions (e.g. [0, 1, 2, ...])
    - scores1: Dice scores for method 1
    - scores2: Dice scores for method 2
    - scores3: Dice scores for method 3
    - labels: Labels for x-ticks (usually file names)
    - title, xlabel, ylabel: Titles for plot and axes
    - legend_labels: Tuple with labels for the three methods
    - figsize: Size of figure in inches
    """
    plt.figure(figsize=figsize)
    plt.scatter(x_positions, scores1, color='C0', label=legend_labels[0])
    plt.scatter(x_positions, scores2, color='C1', label=legend_labels[1])
    plt.scatter(x_positions, scores3, color='C2', label=legend_labels[2])

    # Draw connecting lines per file
    for xi, y1, y2, y3 in zip(x_positions, scores1, scores2, scores3):
        plt.plot([xi, xi, xi], [y1, y2, y3], color='gray', alpha=0.5)

    plt.xticks(x_positions, labels, rotation=45, ha='right')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()



# GRAY VALUE HISTOGRAM

def plot_gray_histogram(image: np.ndarray, bins: int = 256, title: str = "Gray Value Histogram") -> None:
    """
    Plots the histogram of a grayscale image.

    Parameters:
        image (np.ndarray): 2D array representing a grayscale image.
        bins (int): Number of bins in the histogram (default: 256).
        title (str): Title for the histogram plot.
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
    Erstellt einen einzelnen Boxplot, der die Werte nach 'Dataset' gruppiert.

    Args:
        df (pd.DataFrame): DataFrame mit den Spalten 'Dataset' und 'Value'.
        title (str): Titel des Plots.
        ylabel (str): Beschriftung der y-Achse.
        palette (str, optional): Farbschema für Boxplots. Default: "pastel".
        figsize (tuple, optional): Größe der Figure. Default: (8,6).
        save_path (str, optional): Falls angegeben, wird der Plot als Bilddatei gespeichert.
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
        print(f"Plot gespeichert unter: {save_path}")
    else:
        plt.show()



def combine_results_to_dataframe(names, lists):
    """
    Kombiniert mehrere Ergebnislisten in einen einzigen DataFrame
    mit den Spalten 'Dataset' und 'Value'.

    Args:
        names (list of str): Namen der Datasets, z.B. ["GOWT1", "HeLa", "NIH3T3"].
        lists (list of list of float): Ergebnislisten, z.B. [[...], [...], [...]].

    Returns:
        pd.DataFrame: Gesamter DataFrame mit allen Werten.
    """
    if len(names) != len(lists):
        raise ValueError("Die Länge von names und lists muss übereinstimmen!")

    dfs = []
    for name, values in zip(names, lists):
        df = pd.DataFrame({
            "Dataset": [name] * len(values),
            "Value": values
        })
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)
