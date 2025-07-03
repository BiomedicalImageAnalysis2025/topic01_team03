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
package_dice_scores = [0.9128436675562167, 0.8845252721173281, 0.8225479821936802, 0.758336987687637, 0.7528567225654604, 0.647632667167185, 0.6466958730507323, 0.7237186625334818, 0.03500481623642597, 0.46518566600901357, 0.0, 0.6762501531852865, 0.00026339009389856846, 0.5757554586315079, 0.6165448260228947, 0.07605520913993832, 0.07186834004262373, 0.7925039681767514]
our_dice_scores = [0.7920301161671823, 0.7266143352296371, 0.2789833540967043, 0.7576211555278375, 0.7559234810310762, 0.7455316470373867, 0.68634751044449, 0.7094022368473593, 0.4132205323655431, 0.6166252169764397, 0.6572326594245911, 0.17775525398161526, 0.5257368984631994, 0.463560814118282, 0.30095600629458924, 0.6467204621586329, 0.7079234718155428, 0.15430873807203174]

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
#x_fit = np.linspace(min(our_dice_scores), max(package_dice_scores), 100)
#y_fit = slope * x_fit + intercept
#sns.lineplot(x=x_fit, y=y_fit, color="orange", label="Regression", linestyle="-")

# Plot identity line y=x for reference of perfect agreement.
min_val, max_val = min(min(our_dice_scores), min(package_dice_scores)), max(max(our_dice_scores), max(package_dice_scores))
sns.lineplot(x=[min_val, max_val], y=[min_val, max_val], color="black", linestyle="--", label="y = x")

# Set axis labels and title.
plt.xlabel("Dice Score of Multi Otsu")
plt.ylabel("Dice Score of Otsu Global")
plt.title("Comparison: Multi Otsu vs. Otsu Global")

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


import sys
import os

# Hole das Verzeichnis dieses Skripts
script_dir = os.path.dirname(os.path.realpath(__file__))

# Projekt-Root: zwei Level über /plots/
project_root = os.path.abspath(os.path.join(script_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)


from skimage.filters import threshold_multiotsu
from skimage import img_as_ubyte

# Determine point colors for scatter plot based on relative performance.
colors = []
red_indices = []  # hier speichern wir die Positionen der roten Punkte
for idx, (x, y) in enumerate(zip(our_dice_scores, package_dice_scores)):
    if np.isclose(y, x):
        colors.append("blue")   # scores are essentially identical
    elif y > x:
        colors.append("red")    # package score higher → our method worse
        red_indices.append(idx)  # Index merken, wenn Paket besser ist (rot)
    else:
        colors.append("green")  # our score higher → our method better

print("Indices of red points (package Dice > our Dice):", red_indices)


from src.Complete_Otsu_Global import otsu_threshold_skimage_like
from src.imread_all import load_nih3t3_images

imgs_NIH3T3, gts_NIH3T3, img_paths_NIH3T3, gt_paths_NIH3T3 = load_nih3t3_images()

def apply_multiotsu_mask_class1_foreground(img):
    """
    Nutzt Multi-Otsu und weist nur Klasse 1 (zwischen den ersten beiden Schwellen)
    als Vordergrund zu. Klasse 0 (dunkel) und Klassen >=2 (heller) werden Hintergrund.
    """
    img_u8 = img_as_ubyte(img)
    thresholds = threshold_multiotsu(img_u8, classes=3)
    regions = np.digitize(img_u8, bins=thresholds)
    
    # Nur Klasse 1 wird Vordergrund
    mask = (regions == 1)
    return mask


# Beispiel: imgs_NIH3T3, gts_NIH3T3 kommen aus deinem Dataset-Loader
# multi_masks: binäre Masken aus apply_multiotsu_mask_class1_foreground
# otsu_masks: binäre Masken aus Otsu-Global-Binarisierung
# red_indices: Liste der Indizes, die du aus deinem Vergleichscode erhältst

# Berechne alle Masken einmal, damit du sie hast
#multi_masks = [apply_multiotsu_mask_class1_foreground(img) for img in imgs_NIH3T3]
#otsu_masks = [img > otsu_threshold_skimage_like(img) for img in imgs_NIH3T3]

# Für alle roten Punkte plots erzeugen
#for idx in red_indices:
    #img_multi = multi_masks[idx]
    #img_otsu = otsu_masks[idx]
    #img_gt = gts_NIH3T3[idx] > 0  # Ground truth binär

    #fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    #axes[0].imshow(img_multi, cmap="gray")
    #axes[0].set_title(f"Multi-Otsu (Idx {idx})")
    #axes[0].axis("off")
    
    #axes[1].imshow(img_otsu, cmap="gray")
    #axes[1].set_title("Otsu Global")
    #axes[1].axis("off")
    
    #axes[2].imshow(img_gt, cmap="gray")
    #axes[2].set_title("Ground Truth")
    #axes[2].axis("off")
    
    #plt.tight_layout()
    #plt.show()



#anderer versuch--------------------------------------------------

from skimage.filters import threshold_multiotsu
from skimage import img_as_ubyte, exposure
from skimage.morphology import remove_small_objects, remove_small_holes

# Methode 1: nur hellste Klasse (basic)
def apply_multiotsu_hellste_klasse(img, n_classes=3):
    img_u8 = img_as_ubyte(img)
    thresholds = threshold_multiotsu(img_u8, classes=n_classes)
    regions = np.digitize(img_u8, bins=thresholds)
    means = [np.mean(img_u8[regions == k]) for k in range(n_classes)]
    brightest_class = np.argmax(means)
    mask = (regions == brightest_class)
    return mask

# Methode 2: CLAHE + hellste Klasse
def apply_multiotsu_clahe_hellste(img, n_classes=3, clip_limit=0.03):
    img_u8 = img_as_ubyte(img)
    img_clahe = exposure.equalize_adapthist(img_u8, clip_limit=clip_limit)
    img_clahe_u8 = img_as_ubyte(img_clahe)
    thresholds = threshold_multiotsu(img_clahe_u8, classes=n_classes)
    regions = np.digitize(img_clahe_u8, bins=thresholds)
    means = [np.mean(img_clahe_u8[regions == k]) for k in range(n_classes)]
    brightest_class = np.argmax(means)
    mask = (regions == brightest_class)
    return mask

# Methode 3: CLAHE + hellste Klasse + Rauschreduktion
def apply_multiotsu_clahe_clean(img, n_classes=3, clip_limit=0.03, min_size=200):
    img_u8 = img_as_ubyte(img)
    img_clahe = exposure.equalize_adapthist(img_u8, clip_limit=clip_limit)
    img_clahe_u8 = img_as_ubyte(img_clahe)
    thresholds = threshold_multiotsu(img_clahe_u8, classes=n_classes)
    regions = np.digitize(img_clahe_u8, bins=thresholds)
    means = [np.mean(img_clahe_u8[regions == k]) for k in range(n_classes)]
    brightest_class = np.argmax(means)
    mask = (regions == brightest_class)
    mask = remove_small_objects(mask, min_size=min_size)
    mask = remove_small_holes(mask, area_threshold=min_size)
    return mask

# -----------------------------------------
# Beispiel: Berechne alle Masken mit der bevorzugten Methode:
#multi_masks = [apply_multiotsu_hellste_klasse(img) for img in imgs_NIH3T3]
multi_masks = [apply_multiotsu_clahe_hellste(img) for img in imgs_NIH3T3]
#multi_masks = [apply_multiotsu_clahe_clean(img) for img in imgs_NIH3T3]  # <<== am robustesten

# Otsu-Global-Masken bleiben gleich
otsu_masks = [img > otsu_threshold_skimage_like(img) for img in imgs_NIH3T3]

# Für alle roten Punkte plots erzeugen
for idx in red_indices:
    img_multi = multi_masks[idx]
    img_otsu = otsu_masks[idx]
    img_gt = gts_NIH3T3[idx] > 0  # Ground truth binär

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(img_multi, cmap="gray")
    axes[0].set_title(f"Multi-Otsu (Idx {idx})")
    axes[0].axis("off")
    
    axes[1].imshow(img_otsu, cmap="gray")
    axes[1].set_title("Otsu Global")
    axes[1].axis("off")
    
    axes[2].imshow(img_gt, cmap="gray")
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()

from src.Dice_Score import dice_score

def calculate_multiotsu_dice_scores1(imgs, gts):
    scores = []
    for img, gt in zip(imgs, gts):
        mask = apply_multiotsu_hellste_klasse(img)
        gt_binary = gt > 0
        scores.append(dice_score(mask, gt_binary))
    return scores

def calculate_multiotsu_dice_scores2(imgs, gts):
    scores = []
    for img, gt in zip(imgs, gts):
        mask = apply_multiotsu_clahe_hellste(img)
        gt_binary = gt > 0
        scores.append(dice_score(mask, gt_binary))
    return scores

def calculate_multiotsu_dice_scores3(imgs, gts):
    scores = []
    for img, gt in zip(imgs, gts):
        mask = apply_multiotsu_clahe_clean(img)
        gt_binary = gt > 0
        scores.append(dice_score(mask, gt_binary))
    return scores


def calculate_multiotsu_dice_scores4(imgs, gts):
    scores = []
    for img, gt in zip(imgs, gts):
        t = otsu_threshold_skimage_like(img)
        mask = img > t  # Schwelle auf das Bild anwenden
        gt_binary = gt > 0
        scores.append(dice_score(mask, gt_binary))
    return scores


def calculate_multiotsu_dice_scores5(imgs, gts):
    scores = []
    for img, gt in zip(imgs, gts):
        mask = apply_multiotsu_mask_class1_foreground(img)
        gt_binary = gt > 0
        scores.append(dice_score(mask, gt_binary))
    return scores



#dice_nih1   = calculate_multiotsu_dice_scores1(imgs_NIH3T3, gts_NIH3T3)
#dice_nih2   = calculate_multiotsu_dice_scores2(imgs_NIH3T3, gts_NIH3T3)
#dice_nih3   = calculate_multiotsu_dice_scores3(imgs_NIH3T3, gts_NIH3T3)
#dice_otsu = calculate_multiotsu_dice_scores4(imgs_NIH3T3, gts_NIH3T3)
#multi = calculate_multiotsu_dice_scores5(imgs_NIH3T3, gts_NIH3T3)
# Als normale Floats ausgeben
# Nur die roten Bilder ausgeben
#dice_nih1_red = [dice_nih1[idx] for idx in red_indices]
#dice_nih2_red = [dice_nih2[idx] for idx in red_indices]
#dice_nih3_red = [dice_nih3[idx] for idx in red_indices]
#dice_otsu_red = [dice_otsu[idx] for idx in red_indices]
#multi_red = [multi[idx] for idx in red_indices]

#print("\n--- Nur rote Bilder (package Dice > our Dice) ---")
#print("NIH3T3_Scores1 =", ", ".join(f"{score}" for score in dice_nih1_red))
#print("NIH3T3_Scores2 =", ", ".join(f"{score}" for score in dice_nih2_red))
#print("NIH3T3_Scores3 =", ", ".join(f"{score}" for score in dice_nih3_red))
#print("Otsu =", ", ".join(f"{score}" for score in dice_otsu_red))
#print("NIH3T3_multi =", ", ".join(f"{score}" for score in multi_red))

