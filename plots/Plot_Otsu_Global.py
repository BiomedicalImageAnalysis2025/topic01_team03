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

slope, intercept = np.polyfit(indices, dice_scores, 1)

# Figure:
plt.figure(figsize=(8, 6))

# Punkte manuell plotten
for x, y in zip(indices, dice_scores):
    color = "red" if y > x else "blue"
    plt.scatter(x, y, color=color)

# Regressionslinie
x_fit = np.linspace(min(indices), max(indices), 100)
y_fit = slope * x_fit + intercept
plt.plot(x_fit, y_fit, color="black", linestyle="-", label="Regression")

# Linie y = x
plt.plot([0, 1], [0, 1], color="green", linestyle="--", label="y = x")

# Regressionsgleichung
equation = f"y = {slope:.2f} x + {intercept:.2f}"
plt.text(
    0.05, 0.95,
    f"Gradient: {slope:.2f}\n{equation}",
    ha="left", va="top",
    transform=plt.gca().transAxes,
    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
)

# Labels und Titel
plt.xlabel("Dice Score of our Otsu Global")
plt.ylabel("Dice Score of Package Otsu Global")
plt.title("Package Otsu Global vs. Our Otsu Global")

# Legende & Grid
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

# Plot anzeigen
plt.show()

# Interpretation basierend auf Regressionslinie vs. y=x:
if slope < 1 or (np.isclose(slope, 1) and intercept < 0):
    print(f"Steigung: {slope:.2f} → Deine Methode erzielt im Durchschnitt bessere Dice-Scores als die Vergleichsmethode (Punkte liegen unter y=x).")
elif slope > 1 or (np.isclose(slope, 1) and intercept > 0):
    print(f"Steigung: {slope:.2f} → Deine Methode erzielt im Durchschnitt schlechtere Dice-Scores als die Vergleichsmethode (Punkte liegen über y=x).")
else:
    print(f"Steigung: {slope:.2f} → Deine Methode liefert im Durchschnitt ähnliche Dice-Scores wie die Vergleichsmethode.")
