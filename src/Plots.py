import numpy as np
import matplotlib.pyplot as plt
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