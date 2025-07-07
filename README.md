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

## Repository Structure

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
├── .gitignore│   
├── README.md
├── main.ipynb
---

## Prerequisites & Dependencies

Before running any of the scripts or notebooks that rely on the imports shown above, please ensure you have the following software and Python packages installed:

1. **Python Version**  
   - Python 3.8 or newer  

2. **Core Python Libraries (built-in)**  
   - `os` (standard library)  
   - `sys` (standard library)
   - `pathlib` – for modern, object-oriented filesystem paths (`Path` objects)  
   - `typing` – for type hints (`Union`, `Tuple`, etc.)  

3. **Third‐Party Python Packages**  
   - **NumPy** (`numpy`)  
     - Used for numerical arrays and array‐based computations.  
     - Install via pip:  
       ```bash
       pip install numpy
       ```  
   - **Pillow** (`PIL`)  
     - Provides the `Image` class for loading and manipulating image files (e.g., TIFF, PNG, JPEG).  
     - Install via pip:  
       ```bash
       pip install pillow
       ```  
   - **Matplotlib** (`matplotlib`)  
     - Specifically, `matplotlib.pyplot` is used for plotting and visualizing images or data.  
     - Install via pip:  
       ```bash
       pip install matplotlib
       ```  
    - **scikit-image** (`scikit-image`)  
        - A collection of algorithms for image processing.  
        - In particular, our code imports:  
        - `skimage.io` – for generalized image I/O (reading/writing TIFF, PNG, JPEG, etc.)  
        - `skimage.filters.threshold_otsu` – computes the Otsu threshold for automatic image segmentation  
        - Install via pip:  
        ```bash
        pip install scikit-image
        ```
