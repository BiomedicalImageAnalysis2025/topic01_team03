# other approach for calculating the Dice Score!!!
from skimage.io import imread
from skimage.filters import threshold_otsu  # otsu-global pakage


# skimage Otsu-Global
image = imread("data-git/N2DH-GOWT1/img/t01.tif", as_gray=True)
otsu_threshold = threshold_otsu(image)
otsu = (image > otsu_threshold).astype(int).flatten()  # binary & 1D

# start of Dice Score code
from skimage.io import imread


otsu_img = otsu   # output of otsu

ground_truth = imread("data-git/N2DH-GOWT1/gt/man_seg01.tif", as_gray=True)
otsu_gt = (ground_truth > 0).astype(int).flatten()  # gt binary & 1D

print("sum_gt:", len(otsu_gt))     # test how sum_gt looks like
print("sum_img:", len(otsu_img))   # test how sum_img looks like

def dice_score(otsu_img, otsu_gt):

    # control if the Pictures have the same Size
    if len(otsu_img) != len(otsu_gt):
        raise ValueError("Images don't have the same length!")

    # defining the variables for the Dice Score equation
    positive_overlap = 0
    sum_img = len(otsu_img)
    sum_gt = len(otsu_gt)

    for t, p in zip(otsu_img, otsu_gt):
        if t == p:
            positive_overlap += 1

    print("positive_overlap:", positive_overlap)    # test how positive_overlap looks like
    
    return 2 * positive_overlap / (sum_img + sum_gt) 
    

print("Dice Score:", dice_score(otsu_img, otsu_gt))     # why is the output different to Dice Score
