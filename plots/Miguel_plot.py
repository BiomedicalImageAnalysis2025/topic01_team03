import pandas as pd

# Beispiel-Daten
methoden = ["U-Net", "U-Net", "U-Net", "DeepLabV3", "DeepLabV3", "DeepLabV3"]
bilder = ["image_001", "image_002", "image_003",
          "image_001", "image_002", "image_003"]
dice_scores = [0.92, 0.89, 0.87, 0.85, 0.83, 0.84]

df = pd.DataFrame({"Methode": methoden,
                   "Bild-ID": bilder,
                   "Dice-Score": dice_scores})

print(df)
