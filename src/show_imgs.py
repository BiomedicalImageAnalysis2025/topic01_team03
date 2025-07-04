import matplotlib.pyplot as plt
import numpy as np
import cv2  # OpenCV for resizing with interpolation


def plot_images_side_by_side(images, titles=None, figsize_per_image=5, main_title=None):
    """
    Plots a list of images side by side with optional individual titles.

    Args:
        images (list of np.ndarray): The images to display.
        titles (list of str, optional): Titles for each image (should have the same length as images or be None).
        figsize_per_image (float, optional): Width per image in inches (default=5).
        main_title (str, optional): A main title for the entire figure.
    """
    n_images = len(images)
    fig_width = figsize_per_image * n_images
    fig, axes = plt.subplots(1, n_images, figsize=(fig_width, figsize_per_image))
    
    # If there's only one image, axes is not an array → wrap in a list
    if n_images == 1:
        axes = [axes]

    for idx, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img, cmap="gray")
        if titles and idx < len(titles):
            ax.set_title(titles[idx])
        ax.axis("off")
    
    if main_title:
        plt.suptitle(main_title, fontsize=16, y=1.02)
    
    plt.tight_layout()
    plt.show()



def plot_images_fixed_size(images, titles=None, figsize_per_image=5, main_title=None, target_height=512):
    """
    Plots images side by side with identical target height and consistent width,
    ensuring aligned titles by zero-padding images to the same width.

    Args:
        images (list of np.ndarray): Images to display.
        titles (list of str, optional): Titles for each image.
        figsize_per_image (float, optional): Width per image in inches.
        main_title (str, optional): Overall title.
        target_height (int, optional): Uniform height in pixels for all images.
    """
    resized_images = []
    widths = []
    
    # First pass: resize images to target height, store widths
    for img in images:
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img_uint8 = img_norm.astype(np.uint8)
        h, w = img.shape[:2]
        aspect_ratio = w / h
        new_w = int(target_height * aspect_ratio)
        resized_img = cv2.resize(img_uint8, (new_w, target_height), interpolation=cv2.INTER_AREA)
        resized_images.append(resized_img)
        widths.append(new_w)

    # Determine maximum width to pad all images equally
    max_width = max(widths)
    padded_images = []
    for img in resized_images:
        pad_width = max_width - img.shape[1]
        # Add equal padding left & right
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        padded_img = np.pad(img, ((0,0), (pad_left, pad_right)), mode='constant', constant_values=0)
        padded_images.append(padded_img)
    
    fig_width = figsize_per_image * len(padded_images)
    fig, axes = plt.subplots(1, len(padded_images), figsize=(fig_width, figsize_per_image))

    if len(padded_images) == 1:
        axes = [axes]

    for idx, (ax, img) in enumerate(zip(axes, padded_images)):
        ax.imshow(img, cmap="gray")
        if titles and idx < len(titles):
            ax.set_title(titles[idx])
        ax.axis("off")

    if main_title:
        plt.suptitle(main_title, fontsize=16, y=1.02)

    plt.tight_layout()
    plt.show()