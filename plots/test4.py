import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

"""
This script provides a quantitative comparison between Dice scores obtained from two different methods:
our custom global Otsu thresholding implementation ("Our Otsu Global") and a reference implementation from skimage.filters
("Package Otsu Global"). The objective is to determine whether our method consistently produces better, worse,
or equivalent segmentation results compared to the established implementation.

A scatter plot visualizes Dice scores from both methods, where each point represents one image. Points are colored:
- Red if the package's Dice score is higher (indicating inferior performance of our method),
- Green if our Dice score is higher (indicating superior performance),
- Blue if both scores are nearly identical (within floating-point precision).

The script also fits and plots a linear regression line for the paired Dice scores,
alongside the ideal identity line y=x. The slope and intercept of the regression are analyzed:
- A slope < 1 suggests our method performs better overall (points below y=x),
- A slope > 1 suggests worse performance (points above y=x),
- A slope ≈ 1 with intercept ≈ 0 indicates similar performance on average.

The regression equation and slope are shown on the plot for visual inspection,
and an automated interpretation summarizes whether our method is superior, inferior, or comparable
based on the regression analysis.
"""

# Define Dice scores obtained from both methods for comparison.
# package_dice_scores: Dice scores from the reference implementation.
# our_dice_scores: Dice scores from our custom implementation.
package_dice_scores = [
    0.9128436675562167, 0.8845252721173281, 0.8225479821936802, 0.758336987687637, 0.7237186625334818, 0.6762501531852865, 0.5757554586315079, 0.6165448260228947, 0.7925039681767514
]
our_dice_scores = [
    0.7920301161671823, 0.7266143352296371, 0.2789833540967043, 0.7576211555278375, 0.7094022368473593, 0.17775525398161526, 0.463560814118282, 0.30095600629458924, 0.15430873807203174
]

# Verify whether both lists are identical element-wise.
are_identical = package_dice_scores == our_dice_scores
print("Dice scores identical to reference:", are_identical)

# Compute linear regression between our Dice scores (x) and package Dice scores (y).
slope, intercept = np.polyfit(our_dice_scores, package_dice_scores, 1)

# Determine point colors for scatter plot based on relative performance.
colors = []
for x, y in zip(our_dice_scores, package_dice_scores):
    if np.isclose(y, x):
        colors.append("blue")   # scores are essentially identical
    elif y > x:
        colors.append("red")    # package score higher → our method worse
    else:
        colors.append("green")  # our score higher → our method better

# Prepare DataFrame for seaborn plotting.
df = pd.DataFrame({
    "OurDice": our_dice_scores,
    "PackageDice": package_dice_scores,
    "Color": colors
})

# Create scatter plot with seaborn.
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

# Plot regression line based on fitted linear model.
x_fit = np.linspace(min(our_dice_scores), max(package_dice_scores), 100)
y_fit = slope * x_fit + intercept
sns.lineplot(x=x_fit, y=y_fit, color="orange", label="Regression", linestyle="-")

# Plot identity line y=x for reference of perfect agreement.
min_val, max_val = min(min(our_dice_scores), min(package_dice_scores)), max(max(our_dice_scores), max(package_dice_scores))
sns.lineplot(x=[min_val, max_val], y=[min_val, max_val], color="black", linestyle="--", label="y = x")

# Set axis labels and title.
plt.xlabel("Dice Score of Our Otsu Global")
plt.ylabel("Dice Score of Package Otsu Global")
plt.title("Comparison: Package Otsu Global vs. Our Otsu Global")

# Finalize plot with grid and legend.
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

# Show plot window.
plt.show()

# Automated interpretation based on regression slope relative to y=x.
if np.all(np.isclose(our_dice_scores, package_dice_scores)):
    print(f"Slope: {slope:.2f} → Our method yields identical performance to the reference.")
elif slope < 1 or (np.isclose(slope, 1) and intercept < 0):
    print(f"Slope: {slope:.2f} → Our method achieves, on average, better Dice scores than the reference (points lie below y=x).")
elif slope > 1 or (np.isclose(slope, 1) and intercept > 0):
    print(f"Slope: {slope:.2f} → Our method performs worse on average (points lie above y=x).")
else:
    print(f"Slope: {slope:.2f} → Our method yields similar performance to the reference on average.")
