# topic01_team03
# Segmentation of Cell Nuclei Using Otsu Thresholding

This repository contains all code, documentation, and resources for Topic 01 of the 2025 Data Analysis Project: Biomedical Image Analysis (Team 03). Below you will find an overview of the project, how this repository is organized, and instructions for setup, data downloads, and recommended workflows.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Distributin Of Tasks And Use Of AI](#distribution-of-tasks-and-use-of-ai)
3. [Repository Structure](#repository-structure)  
4. [Prerequisites & Dependencies](#prerequisites--dependencies)  


---

## Project Overview

**Topic 01: Biomedical Image Analysis**  
Team 03 focuses on designing and implementing a robust pipeline for segmenting biomedical microscopy images (e.g., cell nuclei, tissue samples), with a particular emphasis on implementing our own Otsu thresholding algorithm. Our primary goals are:

- Develop reproducible image-processing workflows (preprocessing, segmentation).  
- Evaluate and compare multiple segmentation algorithms on hand-annotated benchmark datasets.  
- Build scripts/notebooks for downstream statistical analysis and visualization.  

For a detailed description of the overall course and context, refer to:
- [2025 Data Analysis Topic 01: Biomedical Image Analysis (GitHub)](https://github.com/maiwen-ch/2025_Data_Analysis_Topic_01_Biomedical_Image_Analysis)  
- [2025 Data Analysis Project (General Info)](https://github.com/maiwen-ch/2025_Data_Analysis_Project)  

---

## Repository Structure
```
topic01_team03/
├── data-git/
│   ├── N2DH-GOWT1/
│   │   ├── gt/
│   │   └── img/
│   ├── N2DL-HeLa/
│   │   ├── gt/
│   │   └── img/
│   └── NIH3T3/
│       ├── gt/
│       └── img/
├── output/
│   ├── all_dice_scores_wienerfilter.npy
│   ├── best_radius_our_package.csv
│   ├── dice_all_gowt1_local.npy
│   ├── dice_all_hela_local.npy
│   ├── dice_all_local.npy
│   ├── dice_all_nih_local.npy
│   └── img_gowt1_otsu_local.npy
├── src/
│   ├── __init__.py
│   ├── Complete_Otsu_Global.py
│   ├── Dice_Score_comparison.py
│   ├── Dice_Score.py
│   ├── find_image.py
│   ├── imread_all.py
│   ├── multi.py
│   ├── optimization_gamma.py
│   ├── optimization_meanfilter.py
│   ├── optimization_wienerfilter.py
│   ├── Otsu_Local.py
│   ├── Plots.py
│   ├── pre_processing.py
│   ├── radius_calculation.py
│   ├── Scaling.py
│   └── show_imgs.py
├── .gitignore   
├── README.md
├── main.ipynb
```
---

## Prerequisites & Dependencies

Before running any of the scripts or notebooks in this repository, please ensure that the following software and Python packages are installed.

1. **Python Version**
   - Python 3.8 or newer

---

2. **Core Python Libraries (Built-in)**

These libraries are included with Python and do not require installation:

   - `os` – for file and directory operations  
   - `sys` – for system path manipulation  
   - `pathlib` – for object-oriented filesystem paths (`Path`)  
   - `typing` – for static type annotations (`List`, `Tuple`, etc.)

---

## 3. **Third-Party Python Packages**

Below is a list of required external libraries, including usage and installation instructions:

**NumPy (`numpy`)**  
- Provides support for numerical arrays and mathematical operations.  
- Used throughout for image data, masks, and statistical analysis.  
```bash
pip install numpy
```

**Pandas (`pandas`)**  
  - Used for storing and processing result tables, CSVs, and structured image scores.  
```bash
pip install pandas
```

**Matplotlib (`matplotlib`)**  
- Required for plotting, visualization of segmentation, and histograms.  
- Especially `matplotlib.pyplot` is used for image display and figure generation.  
```bash
pip install matplotlib
```

**Seaborn (`seaborn`)**  
- Enhances Matplotlib with aesthetically pleasing statistical visualizations (e.g., boxplots, stripplots).  
```bash
pip install seaborn
```

**TQDM (`tqdm`)**  
- Provides progress bars for loops and image processing steps.  
```bash
pip install tqdm
```

**Pillow (`pillow`)**  
- Used for image I/O (reading and saving PNG, JPEG, etc.).  
```bash
pip install pillow
```

**scikit-image (indirectly required via `MedPy` or `SimpleITK`)**  
- Functions like `threshold_otsu`, `img_as_ubyte`, and local/multi-Otsu methods.  
- Already bundled into higher-level packages, but can be explicitly installed if needed:  
```bash
pip install scikit-image
```

**OpenCV (`opencv-python`)**  
- Used for additional image processing operations (e.g., smoothing, filtering).  
```bash
pip install opencv-python
```

**MedPy (`medpy`)**  
- Medical image processing package used for some image preprocessing or Dice score comparisons.  
```bash
pip install medpy
```

**SimpleITK (`SimpleITK`)**  
- Optional, may be used for reading 3D medical formats or additional visualization routines.  
```bash
pip install SimpleITK
```

**imagecodecs**  
- Provides low-level codecs for TIFF and other image types. Sometimes needed for I/O performance.  
```bash
pip install imagecodecs
```

---

## Distribution Of Tasks And Use Of AI

Throughout the project, each team member focused on specific components of the image segmentation pipeline:

	•	Marius Mander 
  worked on pre-processing methods, including gamma correction, histogram equalization, mean filtering, and Wiener filter–based background removal.

	•	Victor De Souza Enning 
  implemented the global Otsu thresholding algorithm. He also created the src folder structure for organizing all functions and wrote the import logic so that the modules could be loaded as packages into other parts of the project.

	•	Leo Müller-de Ahne 
  focused on the implementation of the local Otsu thresholding method.

	•	Miguel Gonzalez Ries 
  developed the Dice Score evaluation module and implemented multi-Otsu thresholding.

Final Phase Contributions

	•	Miguel Gonzalez Ries
Took the lead on structuring the final Jupyter Notebook, integrating most components into a clear and coherent workflow.
He also created several plots for the final poster.

	•	Marius Mander
Focused on implementing the Jupyter Notebook section covering pre-processing methods.
Additionally contributed to plotting a few visualizations.

	•	Leo Müller-de Ahne
Took the lead on poster layout and design, as well as crafting the accompanying texts.

	•	Victor De Souza Enning 
Was unable to contribute during the final phase of the project due to health reasons.

Use of AI Tools

AI tools were used throughout the entire project for code generation, optimization, and background research.  Specifically, OpenAI’s models ChatGPT-4o, ChatGPT-o3, ChatGPT-o4-mini, and ChatGPT-o4-mini-high were utilized to support development. All AI-assisted contributions were critically reviewed: we read, checked, and understood the generated code and explanations, and validated the approaches.

---

## Sources

- Otsu, N. (1979). A Threshold Selection Method from Gray-Level Histograms. IEEE Transactions on Systems, Man, and Cybernetics, SMC-9(1), 62–66.  
- Saddami, K., Munadi, K., Away, Y., & Arnia, F. (2019). Improvement of Binarization Performance using Local Otsu Thresholding. International Journal of Electrical and Computer Engineering (IJECE), 9(1), 264–272.
- Win, K. Y., Choomchuay, S., Hamamoto, K., & Raveesunthornkiat, M. (2018). Comparative Study on Automated Cell Nuclei Segmentation Methods for Cytology Pleural Effusion Images. Journal of Healthcare Engineering, 2018, Article ID 9240389.
- Erwin. (2020). Improving Retinal Image Quality Using the Contrast Stretching, Histogram Equalization, and CLAHE Methods with Median Filters. International Journal of Image, Graphics and Signal Processing, 12(2), 30–41.