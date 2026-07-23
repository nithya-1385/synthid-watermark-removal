
"""
adaptive_noise.py

Adaptive Spatial Noise Module

Contains:
1. Adaptive Gaussian Noise
2. Adaptive Speckle Noise
3. Adaptive Salt & Pepper Noise
"""

import numpy as np

from config import (
    MIN_SIGMA,
    MAX_SIGMA,
    RANDOM_SEED
)

np.random.seed(RANDOM_SEED)


# ==========================================================
# Sigma Map
# ==========================================================

def compute_sigma_map(importance_map):
    """
    Convert Importance Map into
    pixel-wise sigma values.
    """

    sigma = (

        MIN_SIGMA +

        importance_map *

        (MAX_SIGMA - MIN_SIGMA)

    )

    return sigma


# ==========================================================
# Adaptive Gaussian Noise
# ==========================================================

def apply_adaptive_gaussian(
    image,
    importance_map
):
    """
    Adaptive Gaussian Noise
    """

    image = image.astype(np.float32)

    sigma = compute_sigma_map(
        importance_map
    )

    sigma = sigma[:, :, np.newaxis]

    noise = np.random.normal(
        0,
        1,
        image.shape
    ).astype(np.float32)

    noisy = image + noise * sigma

    noisy = np.clip(
        noisy,
        0,
        255
    )

    return noisy.astype(np.uint8)


# ==========================================================
# Adaptive Speckle Noise
# ==========================================================

def apply_adaptive_speckle(
    image,
    importance_map
):
    """
    Adaptive Speckle Noise
    """

    image = image.astype(np.float32)

    sigma = compute_sigma_map(
        importance_map
    )

    sigma = sigma[:, :, np.newaxis]

    noise = np.random.normal(
        0,
        1,
        image.shape
    ).astype(np.float32)

    noisy = image + image * noise * sigma / 255.0

    noisy = np.clip(
        noisy,
        0,
        255
    )

    return noisy.astype(np.uint8)


# ==========================================================
# Adaptive Salt & Pepper Noise
# ==========================================================

def apply_adaptive_salt_pepper(
    image,
    importance_map
):
    """
    Adaptive Salt & Pepper Noise
    """

    noisy = image.copy()

    probability = importance_map * 0.04

    random_map = np.random.rand(
        image.shape[0],
        image.shape[1]
    )

    # Salt

    noisy[
        random_map >
        (1 - probability / 2)
    ] = 255

    # Pepper

    noisy[
        random_map <
        (probability / 2)
    ] = 0

    return noisy
