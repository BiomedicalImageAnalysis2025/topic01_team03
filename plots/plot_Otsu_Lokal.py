# our Otsu
#GOWT1 Scores: ['0.28915016099131624', '0.16063829075758015', '0.22105149983872702', '0.2231858373710903', '0.3088252523054779', '0.30119415014900364']
#HeLa Scores: ['0.127401800392217', '0.22748756252516544', '0.8096747709543749', '0.8140793571650516']
#NIH3T3 Scores: ['0.29039299624199894', '0.33118018193464527', '0.4176239930846999', '0.4371394982478085', '0.3701623597067227', '0.3284172674518776', '0.43645741459967713', '0.5000010028581457', '0.4155832945477366', '0.35224958286077346', '0.38652755251439025', '0.3937666853205875', '0.33780649515453676', '0.499008725175988', '0.40156051427417766', '0.36125789077430936', '0.3048632763421365', '0.5034307768885828']

# Pakage otsu
#GOWT1 Scores: ['0.17753002655626046', '0.14881122364626792', '0.1581097146912299', '0.16636302540060405', '0.21184536245369476', '0.24176739061615887']
#HeLa Scores: ['0.12681392194654348', '0.22058378194395434', '0.7145416936492868', '0.7209252373673444']
#NIH3T3 Scores: ['0.25049477381768337', '0.2687685182262189', '0.3484725943766865', '0.34536836298795426', '0.31031052433757284', '0.27648412483165696', '0.3682908336736804', '0.4013789216903318', '0.3289368054331053', '0.29559274196337776', '0.3233109980127425', '0.30894142901699445', '0.27776858136070187', '0.407150525262475', '0.3158035101131886', '0.3073452021431498', '0.26975665051101894', '0.39521273659531875']

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

"""
This script provides a quantitative comparison between Dice scores obtained from two different methods:
our custom local Otsu thresholding implementation ("our Otsu Lokal") and a reference implementation
("Package Otsu Lokal"). The goal is to evaluate whether our method consistently produces better, worse,
or equivalent segmentation performance.

First, Dice scores from both methods are plotted against each other in a scatter plot. Points are colored
red if our Dice score for that image exceeds the package Dice score (indicating superior performance),
and blue otherwise. Additionally, the regression line fitted to the paired scores is plotted alongside
the identity line (y=x), which represents ideal agreement.

The linear regression slope is then used to interpret the relative performance:

- A slope < 1 indicates that our method generally achieves higher Dice scores than the reference, as
  the regression line lies below y=x (most points above the identity line).
- A slope > 1 suggests our method performs worse on average, as the regression line is above y=x
  (most points below the identity line).
- A slope approximately equal to 1 with an intercept ≈ 0 indicates similar performance across both methods.

The computed slope and intercept, as well as the regression equation, are displayed on the plot for
visual inspection. An automated interpretation is printed, summarizing whether our method is
superior, inferior, or comparable to the reference based on the regression analysis.
"""


# Define the Dice scores obtained from two methods for comparison.
# Here, 'dice_scores' represents our computed Dice coefficients, and 'our_indices' represents the reference or package Dice scores.
package_dice_scores = [0.17753002655626046, 0.14881122364626792, 0.1581097146912299, 0.16636302540060405, 0.21184536245369476, 0.24176739061615887, 0.12681392194654348, 0.22058378194395434, 0.7145416936492868, 0.7209252373673444, 0.25049477381768337, 0.2687685182262189, 0.3484725943766865, 0.34536836298795426, 0.31031052433757284, 0.27648412483165696, 0.3682908336736804, 0.4013789216903318, 0.3289368054331053, 0.29559274196337776, 0.3233109980127425, 0.30894142901699445, 0.27776858136070187, 0.407150525262475, 0.3158035101131886, 0.3073452021431498, 0.26975665051101894, 0.39521273659531875]
our_dice_scores = [0.28915016099131624, 0.16063829075758015, 0.22105149983872702, 0.2231858373710903, 0.3088252523054779, 0.30119415014900364, 0.127401800392217, 0.22748756252516544, 0.8096747709543749, 0.8140793571650516, 0.29039299624199894, 0.33118018193464527, 0.4176239930846999, 0.4371394982478085, 0.3701623597067227, 0.3284172674518776, 0.43645741459967713, 0.5000010028581457, 0.4155832945477366, 0.35224958286077346, 0.38652755251439025, 0.3937666853205875, 0.33780649515453676, 0.499008725175988, 0.40156051427417766, 0.36125789077430936, 0.3048632763421365, 0.5034307768885828]

# Verify whether both lists are identical element-wise; expected True if identical.
a = package_dice_scores == our_dice_scores
print("Dice scores identical to reference:", a)

# Compute linear regression parameters
slope, intercept = np.polyfit(our_dice_scores, package_dice_scores, 1)

# Create figure
plt.figure(figsize=(8, 6))

for x, y in zip(our_dice_scores, package_dice_scores):
    if np.isclose(y, x):         # zuerst prüfen, ob y gleich x (innerhalb Toleranz)
        color = "blue"
    elif y > x:
        color = "red"
    else:
        color = "green"
    plt.scatter(x, y, color=color)


# Regression line
x_fit = np.linspace(min(our_dice_scores), max(package_dice_scores), 100)
y_fit = slope * x_fit + intercept
plt.plot(x_fit, y_fit, color="orange", linestyle="-", label="Regression")

# Identity line (y = x)
min_val, max_val = min(min(our_dice_scores), min(package_dice_scores)), max(max(our_dice_scores), max(package_dice_scores))
plt.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--", label="y = x")

# Regression equation annotation
equation = f"y = {slope:.2f} x + {intercept:.2f}"
plt.text(0.05, 0.95, f"Gradient: {slope:.2f}\n{equation}",
         ha="left", va="top", transform=plt.gca().transAxes,
         fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

# Axes labels and title
plt.xlabel("Dice Score of Our Otsu Global")
plt.ylabel("Dice Score of Package Otsu Global")
plt.title("Package Otsu Global vs. Our Otsu Global")

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

# Show plot
plt.show()

# Interpretation based on regression line relative to y=x
if np.all(np.isclose(our_dice_scores, package_dice_scores)):
    print(f"Slope: {slope:.2f} → Our method yields the same performance as the reference.")
elif slope < 1 or (np.isclose(slope, 1) and intercept < 0):
    print(f"Slope: {slope:.2f} → Our method achieves, on average, better Dice scores than the reference (points lie below y=x).")
elif slope > 1 or (np.isclose(slope, 1) and intercept > 0):
    print(f"Slope: {slope:.2f} → Our method performs worse on average (points lie above y=x).")
else:
    print(f"Slope: {slope:.2f} → Our method yields similar performance to the reference on average.")
