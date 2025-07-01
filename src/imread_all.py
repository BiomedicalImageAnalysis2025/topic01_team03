import os
from skimage.io import imread
from glob import glob
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))  # eine Ebene über 'plots'

if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_n2dh_gowt1_images(base_path=None):
    """
    Lädt die N2DH-GOWT1-Bilder und ihre Ground-Truth-Masken.
    Falls base_path None ist, wird der Standardpfad verwendet.
    """
    if base_path is None:
        base_path = os.path.join(project_root, "data-git", "N2DH-GOWT1")

    img_dir = os.path.join(base_path, "img")
    gt_dir = os.path.join(base_path, "gt")

    img_paths = sorted(glob(os.path.join(img_dir, "*.tif")))
    gt_paths = sorted(glob(os.path.join(gt_dir, "*.tif")))

    imgs = [imread(path, as_gray=True) for path in img_paths]
    gts = [imread(path, as_gray=True) for path in gt_paths]

    return imgs, gts, img_paths, gt_paths


def load_n2dl_hela_images(base_path=None):
    """
    Lädt die N2DL-HeLa-Bilder und ihre Ground-Truth-Masken.
    Falls base_path None ist, wird der Standardpfad verwendet.
    """
    if base_path is None:
        base_path = os.path.join(project_root, "data-git", "N2DL-HeLa")

    img_dir = os.path.join(base_path, "img")
    gt_dir = os.path.join(base_path, "gt")

    img_paths = sorted(glob(os.path.join(img_dir, "*.tif")))
    gt_paths = sorted(glob(os.path.join(gt_dir, "*.tif")))

    imgs = [imread(path, as_gray=True) for path in img_paths]
    gts = [imread(path, as_gray=True) for path in gt_paths]

    return imgs, gts, img_paths, gt_paths


def load_nih3t3_images(base_path=None):
    """
    Lädt die NIH3T3-Bilder und ihre Ground-Truth-Masken.
    Falls base_path None ist, wird der Standardpfad verwendet.
    """
    if base_path is None:
        base_path = os.path.join(project_root, "data-git", "NIH3T3")

    img_dir = os.path.join(base_path, "img")
    gt_dir = os.path.join(base_path, "gt")

    img_paths = sorted(glob(os.path.join(img_dir, "*.png")))
    gt_paths = sorted(glob(os.path.join(gt_dir, "*.png")))

    imgs = [imread(path, as_gray=True) for path in img_paths]
    gts = [imread(path, as_gray=True) for path in gt_paths]

    return imgs, gts, img_paths, gt_paths
