import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# All-Daten: Dice-Scores oder beliebige Zahlen
dice_scores =   [0.5705017182130584, 0.32258217915948406, 0.568002229254991, 0.5830196570472606, 0.6502180828858916, 0.588161094224924, 0.07436694813419911, 0.12357364181457739, 0.4032134810257176, 0.4112803223656655, 0.8943783223933183, 0.8845252721173281, 0.8133777288472382, 0.7026585714883163, 0.3158152367118249, 0.28370685279662594, 0.6095864948254034, 0.7237186625334818, 0.028995403964229303, 0.42998864378556173, 0.311728665771575, 0.680536637820677, 0.00021090001581750118, 0.5789127665023891, 0.6286483663940449, 0.07605520913993832, 0.06919951480604966, 0.7965485556767932]
indices =       [0.5705017182130584, 0.32258217915948406, 0.568002229254991, 0.5830196570472606, 0.6502180828858916, 0.588161094224924, 0.07436694813419911, 0.12357364181457739, 0.4032134810257176, 0.4112803223656655, 0.8943783223933183, 0.8845252721173281, 0.8133777288472382, 0.7026585714883163, 0.3158152367118249, 0.28370685279662594, 0.6095864948254034, 0.7237186625334818, 0.028995403964229303, 0.42998864378556173, 0.311728665771575, 0.680536637820677, 0.00021090001581750118, 0.5789127665023891, 0.6286483663940449, 0.07605520913993832, 0.06919951480604966, 0.7965485556767932]
a = dice_scores == indices
print(a)

# Lineare Regression berechnen (für Steigung & Achsenabschnitt)
slope, intercept = np.polyfit(indices, dice_scores, 1)


#plot
plt.figure(figsize=(8, 6))
sns.regplot(
    x=indices,
    y=dice_scores,
    scatter=True,
    ci=None,
    scatter_kws={"color": "blue"},       # Farbe der Punkte
    line_kws={"color": "red"},           # Farbe der Regressionslinie
)
if slope == 0:
    equation = f"y = {intercept:.1f}"
    
elif intercept == 0:
    
    equation = f"y = {slope:.1f} x"

elif intercept == 0 and slope == 0:
    equation = f"y = 0"

else:
    equation = f"y = {slope:.1f} x + {intercept:.1f}"

plt.text(
    0.05, 0.95,                         # Position (x,y) in Achsen-Koordinaten
    f"Gradient: {slope:.1f}\n{equation}",
    ha="left", va="top",
    transform=plt.gca().transAxes,      # in Achsen-Koordinaten (0-1)
    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
)

plt.xlabel("Our Dice Score")
plt.ylabel("Package Dice Score")
plt.title("Package Dice Score vs. Our Dice Score\n(Otsu Global)")
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()
