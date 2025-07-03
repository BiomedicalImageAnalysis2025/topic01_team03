# src/Dice_Score.py

import numpy as np

def dice_score(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Computes the Dice coefficient (Dice Similarity Coefficient, DSC) between two binary images.

    The Dice score quantifies the overlap between a predicted binary segmentation mask and the 
    corresponding ground-truth mask. It is defined as:
        DSC = 2 * |A ∩ B| / (|A| + |B|)
    where A and B represent the sets of foreground pixels in the predicted and ground-truth masks,
    respectively.

    A Dice score of 1 indicates perfect overlap (i.e., identical masks), while 0 indicates no overlap.

    Args:
        prediction (np.ndarray): Binary predicted segmentation mask (dtype=bool).
        ground_truth (np.ndarray): Binary ground-truth segmentation mask (dtype=bool).

    Returns:
        float: Computed Dice score in the range [0.0, 1.0].

    Raises:
        ValueError: If the input arrays do not have the same shape.
    """
    # 1. Validate that the two images have identical dimensions
    if prediction.shape != ground_truth.shape:
        raise ValueError("Prediction and ground truth images must have the same shape.")

    # 2. Calculate the number of positive (True) pixels in both images
    sum_prediction = np.sum(prediction)
    sum_ground_truth = np.sum(ground_truth)

    # 3. Calculate the number of overlapping positive pixels (intersection)
    positive_overlap = np.sum(np.logical_and(prediction, ground_truth))

    # 4. Handle the edge case where both masks are empty (i.e., no positives in either image)
    if sum_prediction + sum_ground_truth == 0:
        return 1.0

    # 5. Compute and return the Dice coefficient
    return 2 * positive_overlap / (sum_prediction + sum_ground_truth)
