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
    
    # Ergebnis der Regression ausgeben
    print(f"Slope: {slope:.2f}, Intercept: {intercept:.2f}")
    return slope, intercept

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
    
    # Ergebnis der Regression ausgeben
    print(f"Slope: {slope:.2f}, Intercept: {intercept:.2f}")
    return slope, intercept


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