import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

"""
This script provides a quantitative comparison between Dice scores obtained from two different methods:
our custom Dice Score implementation ("our Dice Score") and a reference implementation from medpy.metric
("Package Dice Score"). The goal is to evaluate whether our method consistently produces better, worse,
or equivalent segmentation performance.

First, Dice scores from both methods are plotted against each other in a scatter plot. Points are colored
red if the package Dice Score for that image exceeds our Dice score (indicating an inferior performance of our Method),
and green otherwise. Blue on the other hand indicates that the Dice Scores of both methods are similar.
Additionally, the regression line fitted to the paired scores is plotted alongside
the identity line (y=x), which represents ideal agreement.

The linear regression slope is then used to interpret the relative performance:

- A slope < 1 indicates that our method generally achieves higher Dice scores than the reference, as
  the regression line lies below y=x (most points below the identity line).
- A slope > 1 suggests our method performs worse on average, as the regression line is above y=x
  (most points above the identity line).
- A slope approximately equal to 1 with an intercept ≈ 0 indicates similar performance across both methods.

The computed slope and intercept, as well as the regression equation, are displayed on the plot for
visual inspection. An automated interpretation is printed, summarizing whether our method is
superior, inferior, or comparable to the reference based on the regression analysis.
"""

# Define the Dice scores obtained from two methods for comparison.
# Here, 'package_dice_scores' (origin: output of P_Dice_Score.py) represents the package computed Dice coefficients, and 'our_dice_scores' (origin: output of O_Otsu_Global.py) represents the reference our Dice scores. 
package_dice_scores =   [0.5705017182130584, 0.32258217915948406, 0.568002229254991, 0.6271474725294504, 0.6502180828858916, 0.6615248976783192, 0.6923060104510571, 0.649295517879001, 0.7760944676315427, 0.7761119912979735, 0.9128436675562167, 0.8845252721173281, 0.8225479821936802, 0.758336987687637, 0.7528567225654604, 0.647632667167185, 0.6466958730507323, 0.7237186625334818, 0.03500481623642597, 0.46518566600901357, 0.0, 0.6762501531852865, 0.00026339009389856846, 0.5757554586315079, 0.6165448260228947, 0.07605520913993832, 0.07186834004262373, 0.7925039681767514]
our_dice_scores =       [0.5705017182130584, 0.32258217915948406, 0.568002229254991, 0.6271474725294504, 0.6502180828858916, 0.6615248976783192, 0.6923060104510571, 0.649295517879001, 0.7760944676315427, 0.7761119912979735, 0.9128436675562167, 0.8845252721173281, 0.8225479821936802, 0.758336987687637, 0.7528567225654604, 0.647632667167185, 0.6466958730507323, 0.7237186625334818, 0.03500481623642597, 0.46518566600901357, 0.0, 0.6762501531852865, 0.00026339009389856846, 0.5757554586315079, 0.6165448260228947, 0.07605520913993832, 0.07186834004262373, 0.7925039681767514]

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
#equation = f"y = {slope:.2f} x + {intercept:.2f}"
#plt.text(0.05, 0.95, f"Gradient: {slope:.2f}\n{equation}",
 #        ha="left", va="top", transform=plt.gca().transAxes,
  #       fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

# Axes labels and title
plt.xlabel("Our Dice Score")
plt.ylabel("Package Dice Score")
plt.title("Package Dice Score vs. Our Dice Score\n(using our Otsu Global)")

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
