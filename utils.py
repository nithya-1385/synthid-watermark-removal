
"""
utils.py

Common helper functions used throughout the project.
"""

import os
import cv2
import numpy as np

from PIL import Image


# ==========================================================
# Create Folder
# ==========================================================

def create_folder(path):
    """
    Create folder if it does not exist.
    """

    os.makedirs(path, exist_ok=True)


# ==========================================================
# Load Image
# ==========================================================

def load_image(path):
    """
    Load image using OpenCV.
    """

    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(
            f"Image not found : {path}"
        )

    return image


# ==========================================================
# Save Image
# ==========================================================

def save_image(path, image):
    """
    Save image.
    """

    folder = os.path.dirname(path)

    if folder != "":
        create_folder(folder)

    cv2.imwrite(path, image)


# ==========================================================
# OpenCV → PIL
# ==========================================================

def opencv_to_pil(image):
    """
    Convert OpenCV (BGR) image
    to PIL (RGB).
    """

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return Image.fromarray(rgb)


# ==========================================================
# PIL → OpenCV
# ==========================================================

def pil_to_opencv(image):
    """
    Convert PIL (RGB)
    to OpenCV (BGR).
    """

    rgb = np.array(image)

    return cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )


# ==========================================================
# Convert to Grayscale
# ==========================================================

def convert_gray(image):
    """
    Convert image to grayscale.
    """

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


# ==========================================================
# Normalize Map
# ==========================================================

def normalize_map(data):
    """
    Normalize map to range [0,1]
    """

    data = data.astype(np.float32)

    return cv2.normalize(
        data,
        None,
        0,
        1,
        cv2.NORM_MINMAX
    )


# ==========================================================
# Resize Image
# ==========================================================

def resize_image(image, size):
    """
    Resize image.
    """

    return cv2.resize(
        image,
        (size, size),
        interpolation=cv2.INTER_AREA
    )


# ==========================================================
# Difference Map
# ==========================================================

def difference_map(original, watermarked):
    """
    Absolute pixel difference
    between original and watermarked image.
    """

    diff = cv2.absdiff(
        original,
        watermarked
    )

    gray = cv2.cvtColor(
        diff,
        cv2.COLOR_BGR2GRAY
    )

    return normalize_map(gray)
