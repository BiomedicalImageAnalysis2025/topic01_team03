import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

"""
This script provides a quantitative comparison between Dice scores obtained from two methods:
our custom local Otsu thresholding implementation ("Our Otsu Local") and a reference implementation
from skimage.filters ("Package Otsu Local"). The goal is to assess whether our method
systematically achieves better, worse, or comparable segmentation performance.

Dice scores from both methods are visualized in a scatter plot. Points are colored:
- red if the package Dice score is higher (indicating worse performance of our method),
- green if our Dice score is higher (indicating better performance of our method),
- blue if both methods yield similar Dice scores (within numerical precision).

A linear regression line fitted to the paired scores is plotted alongside the identity line (y=x),
which represents ideal agreement between the methods. The slope of the regression is interpreted:
- slope < 1: our method performs better on average,
- slope > 1: our method performs worse on average,
- slope ≈ 1 with intercept ≈ 0: methods perform similarly.

The slope, intercept, and automated performance interpretation are printed as part of the analysis.
"""

# Define the Dice scores obtained from both methods for comparison.
# Here, 'package_dice_scores' comes from skimage-based local Otsu thresholding,
# and 'our_dice_scores' comes from our own implementation.
package_dice_scores = [
    0.17753002655626046, 0.14881122364626792, 0.1581097146912299, 0.16636302540060405,
    0.21184536245369476, 0.24176739061615887, 0.12681392194654348, 0.22058378194395434,
    0.7145416936492868, 0.7209252373673444, 0.25049477381768337, 0.2687685182262189,
    0.3484725943766865, 0.34536836298795426, 0.31031052433757284, 0.27648412483165696,
    0.3682908336736804, 0.4013789216903318, 0.3289368054331053, 0.29559274196337776,
    0.3233109980127425, 0.30894142901699445, 0.27776858136070187, 0.407150525262475,
    0.3158035101131886, 0.3073452021431498, 0.26975665051101894, 0.39521273659531875
]
our_dice_scores = [
    0.28915016099131624, 0.16063829075758015, 0.22105149983872702, 0.2231858373710903,
    0.3088252523054779, 0.30119415014900364, 0.127401800392217, 0.22748756252516544,
    0.8096747709543749, 0.8140793571650516, 0.29039299624199894, 0.33118018193464527,
    0.4176239930846999, 0.4371394982478085, 0.3701623597067227, 0.3284172674518776,
    0.43645741459967713, 0.5000010028581457, 0.4155832945477366, 0.35224958286077346,
    0.38652755251439025, 0.3937666853205875, 0.33780649515453676, 0.499008725175988,
    0.40156051427417766, 0.36125789077430936, 0.3048632763421365, 0.5034307768885828
]

# Check if both lists are identical element-wise for sanity check.
are_identical = package_dice_scores == our_dice_scores
print("Dice scores identical to reference:", are_identical)

# Compute linear regression parameters between our Dice scores and package Dice scores.
slope, intercept = np.polyfit(our_dice_scores, package_dice_scores, 1)

# Assign colors based on relative performance for each point.
colors = []
for x, y in zip(our_dice_scores, package_dice_scores):
    if np.isclose(y, x):
        colors.append("blue")   # scores are essentially identical
    elif y > x:
        colors.append("red")    # package Dice score is higher
    else:
        colors.append("green")  # our Dice score is higher

# Prepare a pandas DataFrame for seaborn plotting.
df = pd.DataFrame({
    "OurDice": our_dice_scores,
    "PackageDice": package_dice_scores,
    "Color": colors
})

# Create the scatter plot.
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="OurDice",
    y="PackageDice",
    hue="Color",
    palette={"red": "red", "green": "green", "blue": "blue"},
    legend=False,
    s=50
)

# Plot regression line derived from linear regression.
x_fit = np.linspace(min(our_dice_scores), max(package_dice_scores), 100)
y_fit = slope * x_fit + intercept
sns.lineplot(x=x_fit, y=y_fit, color="orange", label="Regression", linestyle="-")

# Plot the identity line y=x for ideal agreement reference.
min_val, max_val = min(min(our_dice_scores), min(package_dice_scores)), max(max(our_dice_scores), max(package_dice_scores))
sns.lineplot(x=[min_val, max_val], y=[min_val, max_val], color="black", linestyle="--", label="y = x")

# Annotate regression equation and slope on the plot.
equation = f"y = {slope:.2f} x + {intercept:.2f}"
plt.text(0.05, 0.95, f"Gradient: {slope:.2f}\n{equation}",
         ha="left", va="top", transform=plt.gca().transAxes,
         fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

# Set plot labels and title.
plt.xlabel("Dice Score of Our Otsu Local")
plt.ylabel("Dice Score of Package Otsu Local")
plt.title("Comparison: Package Otsu Local vs. Our Otsu Local\n(radius = 15)")

# Final plot settings.
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()

# Provide interpretation based on regression analysis.
if np.all(np.isclose(our_dice_scores, package_dice_scores)):
    print(f"Slope: {slope:.2f} → Our method yields the same performance as the reference.")
elif slope < 1 or (np.isclose(slope, 1) and intercept < 0):
    print(f"Slope: {slope:.2f} → Our method achieves, on average, better Dice scores than the reference (points lie below y=x).")
elif slope > 1 or (np.isclose(slope, 1) and intercept > 0):
    print(f"Slope: {slope:.2f} → Our method performs worse on average (points lie above y=x).")
else:
    print(f"Slope: {slope:.2f} → Our method yields similar performance to the reference on average.")
