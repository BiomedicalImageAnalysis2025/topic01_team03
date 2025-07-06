import numpy as np
import os
import sys
from medpy.metric import binary
from skimage.filters import threshold_otsu


# add project root
script_dir = os.getcwd()
project_root = os.path.abspath(script_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Import the global Otsu implementation from the project source
import importlib
import src.Dice_Score
importlib.reload(src.Dice_Score)
from src.Complete_Otsu_Global import otsu_threshold_skimage_like
from src.Otsu_Local import local_otsu
from src.Dice_Score import dice_score
from src.Otsu_Local import local_otsu_package
from src.pre_processing import gammacorrection, histogramequalization, mean_filter, local_wiener_filter



def calculate_dice_scores_global(imgs, gts):
    """
    Calculates Dice scores between Otsu-binarized images and ground-truth masks.

    For each image, computes the global Otsu threshold, binarizes the image,
    and evaluates the Dice coefficient against the corresponding ground-truth.

    Args:
        imgs (list[np.ndarray]): List of grayscale input images.
        gts (list[np.ndarray]): Corresponding ground-truth binary masks.

    Returns:
        list[float]: Computed Dice scores.
    """
    scores = []
    for img, gt in zip(imgs, gts):
        # Compute global Otsu threshold on the input image
        t = otsu_threshold_skimage_like(img)

        # Apply threshold to binarize the input image
        otsu_img = img > t

        # Binarize the ground-truth (any positive value is foreground)
        gt_binary = gt > 0

        # Compute Dice score
        score = dice_score(otsu_img, gt_binary)
        scores.append(score)
    return scores

def calculate_dice_scores_local(imgs, gts):
    """
    Computes Dice scores for a list of images using local Otsu thresholding.

    Args:
        imgs (list[np.ndarray]): List of grayscale input images.
        gts (list[np.ndarray]): Corresponding ground-truth binary masks.

    Returns:
        list[float]: Computed Dice scores.
    """
    scores = []
    for img, gt in zip(imgs, gts):
        # Compute local Otsu threshold map
        t_map = local_otsu(img, radius=15)
        # Apply local threshold
        mask = img > t_map
        # Binarize ground-truth mask
        gt_binary = gt > 0
        # Calculate Dice score
        score = dice_score(mask, gt_binary)
        scores.append(score)
    return scores

def calculate_dice_scores_local_package(imgs, gts):
    """
    Computes Dice scores by binarizing each image with skimage's local threshold method
    (threshold_local) and comparing the result to its corresponding ground-truth mask.

    Args:
        imgs (list[np.ndarray]): List of grayscale input images.
        gts (list[np.ndarray]): Corresponding ground-truth masks.

    Returns:
        list[float]: Dice scores for each image/ground-truth pair.
    """
    # Apply local Otsu thresholding
    otsu_imgs = [img > local_otsu_package(img, radius=15) for img in imgs]
    
    # Convert ground-truth masks to binary
    gt_binaries = [gt > 0 for gt in gts]

    # Compute Dice scores
    scores = []
    for otsu_img, gt_binary in zip(otsu_imgs, gt_binaries):
        score = dice_score(otsu_img, gt_binary)
        scores.append(score)

    return scores

def calculate_package_dice_scores(imgs, gts):
    """
    Calculates Dice scores between Otsu-thresholded images and ground-truth masks using
    the medpy.metric Dice implementation (package Dice Score).

    Args:
        imgs (list[np.ndarray]): List of input grayscale images.
        gts (list[np.ndarray]): Corresponding ground-truth binary masks.

    Returns:
        list[float]: Calculated Dice scores.
    """
    scores = []
    for img, gt in zip(imgs, gts):
        # Compute Otsu threshold
        t = otsu_threshold_skimage_like(img)

        # Binarize the input image with the computed threshold
        otsu_img = img > t

        # Binarize the ground-truth mask
        gt_binary = gt > 0

        # Compute Dice score using medpy's implementation
        score = binary.dc(otsu_img, gt_binary)
        scores.append(score)
    return scores

def calculate_dice_scores_otsu_package(imgs, gts):
    """
    Binarizes each image using skimage's global Otsu threshold and computes the Dice score
    against the corresponding ground-truth mask.

    Args:
        imgs (list[np.ndarray]): List of grayscale input images.
        gts (list[np.ndarray]): Corresponding ground-truth masks.

    Returns:
        list[float]: Dice scores for each image/ground-truth pair.
    """
    scores = []
    for img, gt in zip(imgs, gts):
        # Binarize image using global Otsu threshold
        otsu_img = img > threshold_otsu(img)

        # Ensure ground-truth is binary
        gt_binary = gt > 0

        # Compute Dice score
        score = dice_score(otsu_img, gt_binary)
        scores.append(score)
    return scores

def calculate_dice_scores_gamma_global(imgs, gts):
    """
    Process all images and corresponding ground truths to compute a list of Dice scores.

    Args:
        imgs (list of np.ndarray): Grayscale input images.
        gts (list of np.ndarray): Corresponding ground-truth masks.

    Returns:
        dice_scores (list of float): Dice scores for each image-groundtruth pair.
    """
    dice_scores = []

    for img, gt in zip(imgs, gts):
        
        # binarize groundtruth
        gt_bin = 1 - ((gt / gt.max()) == 0)

        # gamma correction
        img_gamma = gammacorrection(img, gamma=0.6)

        # global otsu thresholding
        t = otsu_threshold_skimage_like(img_gamma)
        binary1 = (img_gamma > t)
        # calculate dice score
        score = dice_score(binary1.flatten(), gt_bin.flatten())
        dice_scores.append(score)

    return dice_scores

def calculate_dice_scores_histeq_global(imgs, gts):
    """
    Process all images and corresponding ground truths to compute a list of Dice scores.

    Args:
        imgs (list of np.ndarray): Grayscale input images.
        gts (list of np.ndarray): Corresponding ground-truth masks.

    Returns:
        dice_scores (list of float): Dice scores for each image-groundtruth pair.
    """
    dice_scores = []

    for img, gt in zip(imgs, gts):
        
        # binarize groundtruth
        gt_bin = 1 - ((gt / gt.max()) == 0)

        # histogram equalization
        img_eq = histogramequalization(img)

        # global otsu thresholding
        t = otsu_threshold_skimage_like(img_eq)
        binary1 = (img_eq > t)
        # calculate dice score
        score = dice_score(binary1.flatten(), gt_bin.flatten())
        dice_scores.append(score)

    return dice_scores

def calculate_dice_scores_meanfilter_global(imgs, gts):
    """
    Process all images and corresponding ground truths to compute a list of Dice scores.

    Args:
        imgs (list of np.ndarray): Grayscale input images.
        gts (list of np.ndarray): Corresponding ground-truth masks.

    Returns:
        dice_scores (list of float): Dice scores for each image-groundtruth pair.
    """
    dice_scores = []

    for img, gt in zip(imgs, gts):
        
        # binarize groundtruth
        gt_bin = 1 - ((gt / gt.max()) == 0)

        # mean filter
        img_filtered = mean_filter(img)

        # global otsu thresholding
        t = otsu_threshold_skimage_like(img_filtered)
        binary1 = (img_filtered > t)
        # calculate dice score
        score = dice_score(binary1.flatten(), gt_bin.flatten())
        dice_scores.append(score)

    return dice_scores

def calculate_dice_scores_wienerfilter_global(imgs, gts):
    """
    Process all images and corresponding ground truths to compute a list of Dice scores.

    Args:
        imgs (list of np.ndarray): Grayscale input images.
        gts (list of np.ndarray): Corresponding ground-truth masks.

    Returns:
        dice_scores (list of float): Dice scores for each image-groundtruth pair.
    """
    dice_scores = []

    for img, gt in zip(imgs, gts):
        
        # binarize groundtruth
        gt_bin = 1 - ((gt / gt.max()) == 0)

        # wienfilter-based background estimation and removal
        background = local_wiener_filter(img)
        img_wiener = img - background


        # global otsu thresholding
        t = otsu_threshold_skimage_like(img_wiener)
        binary1 = (img_wiener > t)

        # Invert if necessary
        if np.mean(binary1) > 0.5:
           binary1 = ~binary1
           
        # calculate dice score
        score = dice_score(binary1.flatten(), gt_bin.flatten())
        dice_scores.append(score)

    return dice_scores

def calculate_dice_scores_gamma_meanfilter_global(imgs, gts):
    """
    Process all images and corresponding ground truths to compute a list of Dice scores.

    Args:
        imgs (list of np.ndarray): Grayscale input images.
        gts (list of np.ndarray): Corresponding ground-truth masks.

    Returns:
        dice_scores (list of float): Dice scores for each image-groundtruth pair.
    """
    dice_scores = []

    for img, gt in zip(imgs, gts):
        
        # binarize groundtruth
        gt_bin = 1 - ((gt / gt.max()) == 0)

        # gamma correction
        img_gamma = gammacorrection(img)

        # meanfilter
        img_filtered = mean_filter(img_gamma, radius=5)

        # global otsu thresholding
        t = otsu_threshold_skimage_like(img_filtered)
        binary1 = (img_filtered > t)
        # calculate dice score
        score = dice_score(binary1.flatten(), gt_bin.flatten())
        dice_scores.append(score)

    return dice_scores