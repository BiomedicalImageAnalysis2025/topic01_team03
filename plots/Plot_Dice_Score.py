import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

"""
This script performs a quantitative evaluation of segmentation performance by comparing Dice scores 
obtained from our custom implementation against a reference Dice calculation. Specifically, the goal 
is to determine whether our method achieves better, worse, or equivalent performance compared to the 
reference across a set of images.

The Dice scores from both methods are plotted in a scatter plot:
- Points are colored **green** if our method yields a higher Dice score than the reference (superior performance).
- Points are colored **red** if the reference score is higher than ours (inferior performance).
- Points are colored **blue** if both methods produce approximately identical scores (numerical agreement within tolerance).

A linear regression is fitted to the paired scores, and the regression line is plotted alongside the 
identity line (y=x), representing perfect agreement. The regression's slope and intercept are annotated 
in the figure for transparency.

Interpretation criteria:
- Slope < 1 → our method generally outperforms the reference (regression line below y=x).
- Slope > 1 → our method underperforms relative to the reference (regression line above y=x).
- Slope ≈ 1 and intercept ≈ 0 → our method achieves similar performance.

An automatic interpretation based on the regression results is printed.
"""

# Define the Dice scores obtained from two methods for comparison.
# 'package_dice_scores' (from P_Dice_Score.py) are the reference Dice coefficients,
# and 'our_dice_scores' (from O_Otsu_Global.py) are our computed Dice scores.
package_dice_scores = [
    0.5705017182130584, 0.32258217915948406, 0.568002229254991, 0.6271474725294504, 0.6502180828858916,
    0.6615248976783192, 0.6923060104510571, 0.649295517879001, 0.7760944676315427, 0.7761119912979735,
    0.9128436675562167, 0.8845252721173281, 0.8225479821936802, 0.758336987687637, 0.7528567225654604,
    0.647632667167185, 0.6466958730507323, 0.7237186625334818, 0.03500481623642597, 0.46518566600901357,
    0.0, 0.6762501531852865, 0.00026339009389856846, 0.5757554586315079, 0.6165448260228947,
    0.07605520913993832, 0.07186834004262373, 0.7925039681767514
]
our_dice_scores = [
    0.5705017182130584, 0.32258217915948406, 0.568002229254991, 0.6271474725294504, 0.6502180828858916,
    0.6615248976783192, 0.6923060104510571, 0.649295517879001, 0.7760944676315427, 0.7761119912979735,
    0.9128436675562167, 0.8845252721173281, 0.8225479821936802, 0.758336987687637, 0.7528567225654604,
    0.647632667167185, 0.6466958730507323, 0.7237186625334818, 0.03500481623642597, 0.46518566600901357,
    0.0, 0.6762501531852865, 0.00026339009389856846, 0.5757554586315079, 0.6165448260228947,
    0.07605520913993832, 0.07186834004262373, 0.7925039681767514
]

# Check for exact equality
identical = package_dice_scores == our_dice_scores
print("Dice scores identical to reference:", identical)

# Compute linear regression
slope, intercept = np.polyfit(our_dice_scores, package_dice_scores, 1)

# Assign colors for scatter points
colors = []
for x, y in zip(our_dice_scores, package_dice_scores):
    if np.isclose(y, x):
        colors.append("blue")
    elif y > x:
        colors.append("red")
    else:
        colors.append("green")

# Create DataFrame for plotting
df = pd.DataFrame({
    "OurDice": our_dice_scores,
    "PackageDice": package_dice_scores,
    "Color": colors
})

# Plot
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df, x="OurDice", y="PackageDice", hue="Color",
    palette={"red": "red", "green": "green", "blue": "blue"},
    legend=False, s=50
)

# Regression line
x_fit = np.linspace(min(our_dice_scores), max(package_dice_scores), 100)
y_fit = slope * x_fit + intercept
sns.lineplot(x=x_fit, y=y_fit, color="orange", label="Regression", linestyle="-")

# Identity line y=x
min_val = min(min(our_dice_scores), min(package_dice_scores))
max_val = max(max(our_dice_scores), max(package_dice_scores))
sns.lineplot(x=[min_val, max_val], y=[min_val, max_val], color="black", linestyle="--", label="y = x")

# Annotate regression equation
equation = f"y = {slope:.2f} x + {intercept:.2f}"
plt.text(0.05, 0.95, f"Gradient: {slope:.2f}\n{equation}",
         ha="left", va="top", transform=plt.gca().transAxes,
         fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

# Labels, title, grid, legend
plt.xlabel("Our Dice Score")
plt.ylabel("Package Dice Score")
plt.title("Package Dice Score vs. Our Dice Score\n(using our Otsu Global)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

plt.show()

# Interpretation based on regression results
if np.all(np.isclose(our_dice_scores, package_dice_scores)):
    print(f"Slope: {slope:.2f} → Our method yields the same performance as the reference.")
elif slope < 1 or (np.isclose(slope, 1) and intercept < 0):
    print(f"Slope: {slope:.2f} → Our method achieves, on average, better Dice scores than the reference (points lie below y=x).")
elif slope > 1 or (np.isclose(slope, 1) and intercept > 0):
    print(f"Slope: {slope:.2f} → Our method performs worse on average (points lie above y=x).")
else:
    print(f"Slope: {slope:.2f} → Our method yields similar performance to the reference on average.")
