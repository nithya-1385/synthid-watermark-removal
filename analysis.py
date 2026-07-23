
"""
analysis.py

Image Analysis Module

Generates:
1. Texture Map
2. Gradient Map
3. Variance Map
4. Importance Map
"""

import cv2
import numpy as np
from scipy.ndimage import generic_filter

from utils import normalize_map

from config import (
    TEXTURE_WEIGHT,
    GRADIENT_WEIGHT,
    VARIANCE_WEIGHT
)

# ==========================================================
# Texture Map
# ==========================================================

def compute_texture_map(image, window_size=7):
    """
    Local texture using standard deviation.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    texture = generic_filter(
        gray.astype(np.float32),
        np.std,
        size=window_size
    )

    return normalize_map(texture)


# ==========================================================
# Gradient Map
# ==========================================================

def compute_gradient_map(image):
    """
    Sobel gradient magnitude.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    gradient = cv2.magnitude(
        gx,
        gy
    )

    return normalize_map(
        gradient
    )


# ==========================================================
# Variance Map
# ==========================================================

def compute_variance_map(
    image,
    window_size=7
):
    """
    Local variance.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    variance = generic_filter(
        gray.astype(np.float32),
        np.var,
        size=window_size
    )

    return normalize_map(
        variance
    )


# ==========================================================
# Importance Map
# ==========================================================

def compute_importance_map(
    texture,
    gradient,
    variance
):
    """
    Weighted fusion of maps.
    """

    importance = (

        TEXTURE_WEIGHT * texture +

        GRADIENT_WEIGHT * gradient +

        VARIANCE_WEIGHT * variance

    )

    return normalize_map(
        importance
    )


# ==========================================================
# Analyze Image
# ==========================================================

def analyze_image(image):
    """
    Generate all maps.
    """

    texture = compute_texture_map(
        image
    )

    gradient = compute_gradient_map(
        image
    )

    variance = compute_variance_map(
        image
    )

    importance = compute_importance_map(

        texture,

        gradient,

        variance

    )

    return {

        "texture": texture,

        "gradient": gradient,

        "variance": variance,

        "importance": importance

    }
