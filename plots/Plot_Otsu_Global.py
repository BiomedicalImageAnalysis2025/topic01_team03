import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the Dice scores obtained from two methods for comparison.
# Here, 'dice_scores' represents our computed Dice coefficients, and 'indices' represents the reference or package Dice scores.
dice_scores =   [0.5705017182130584, 0.32258217915948406, 0.568002229254991, 0.5830196570472606, 0.6502180828858916, 0.588161094224924, 0.6855886556007476, 0.6478693311433525, 0.76845804612436, 0.7660179019659928, 0.8943783223933183, 0.8845252721173281, 0.8133777288472382, 0.7026585714883163, 0.7528567225654604, 0.647632667167185, 0.6095864948254034, 0.7237186625334818, 0.028995403964229303, 0.42998864378556173, 0.0, 0.680536637820677, 0.00021090001581750118, 0.5789127665023891, 0.6286483663940449, 0.07605520913993832, 0.06919951480604966, 0.7965485556767932]
indices =       [0.5705017182130584, 0.32258217915948406, 0.568002229254991, 0.6271474725294504, 0.6502180828858916, 0.6615248976783192, 0.6923060104510571, 0.649295517879001, 0.7760944676315427, 0.7761119912979735, 0.9128436675562167, 0.8845252721173281, 0.8225479821936802, 0.758336987687637, 0.7528567225654604, 0.647632667167185, 0.6466958730507323, 0.7237186625334818, 0.03500481623642597, 0.46518566600901357, 0.0, 0.6762501531852865, 0.00026339009389856846, 0.5757554586315079, 0.6165448260228947, 0.07605520913993832, 0.07186834004262373, 0.7925039681767514]

# Verify whether both lists are identical element-wise; expected True if identical.
a = dice_scores == indices
print(a)

# Compute the linear regression between the two sets of Dice scores. 'slope' represents the gradient of the regression line, and 'intercept' is the point where the line crosses the y-axis.
slope, intercept = np.polyfit(indices, dice_scores, 1)


# Initialize the figure for the plot with a defined size.
plt.figure(figsize=(8, 6))

# Plot the scatter plot of Dice scores with a fitted regression line using seaborn.
sns.regplot(
    x=indices,
    y=dice_scores,
    scatter=True,
    ci=None,
    scatter_kws={"color": "blue"},       # Set point color to blue
    line_kws={"color": "red"},           # Set regression line color to red
)

# Generate a string representation of the regression equation with cases handling special situations where slope or intercept is zero
if slope == 0:
    equation = f"y = {intercept:.1f}"
    
elif intercept == 0:
    
    equation = f"y = {slope:.1f} x"

elif intercept == 0 and slope == 0:
    equation = f"y = 0"

else:
    equation = f"y = {slope:.1f} x + {intercept:.1f}"

# Display the regression gradient and the regression equation in the plot using a text box in the upper-left corner of the plot area.
plt.text(
    0.05, 0.95,                             # Position relative to the axes (0-1 scale)
    f"Gradient: {slope:.1f}\n{equation}",   # Text content: gradient + equation
    ha="left", va="top",                    # Align text box to the top-left
    transform=plt.gca().transAxes,          # Use axes-relative coordinates
    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)  # Style text box
)

# Add labels and a descriptive title to the plot.
plt.xlabel("Dice Score of our Otsu Global")
plt.ylabel("Dice Score of Package Otsu Global")
plt.title("Package Otsu Global vs. Our Otsu Global")

# Add a grid to improve plot readability.
plt.grid(True, linestyle='--', alpha=0.5)

# Render the plot.
plt.show()