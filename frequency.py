
"""
frequency.py

Adaptive Multi-Level DWT Attack
"""

import cv2
import numpy as np
import pywt

from config import (
    WAVELET,
    FREQUENCY_STRENGTH,
    RANDOM_SEED
)

np.random.seed(RANDOM_SEED)


# ==========================================================
# Attack One DWT Level
# ==========================================================

def attack_level(coeffs, importance):

    LH, HL, HH = coeffs

    # ------------------------------------
    # Resize importance map
    # ------------------------------------

    importance = cv2.resize(
        importance,
        (LH.shape[1], LH.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )

    # ------------------------------------
    # Adaptive attenuation
    # ------------------------------------

    attenuation = 1 - (0.35 * importance)

    LH = LH * attenuation
    HL = HL * attenuation
    HH = HH * attenuation

    # ------------------------------------
    # Adaptive Gaussian perturbation
    # ------------------------------------

    LH += np.random.normal(
        0,
        FREQUENCY_STRENGTH,
        LH.shape
    ) * importance

    HL += np.random.normal(
        0,
        FREQUENCY_STRENGTH,
        HL.shape
    ) * importance

    HH += np.random.normal(
        0,
        FREQUENCY_STRENGTH,
        HH.shape
    ) * importance

    # ------------------------------------
    # Adaptive Thresholding
    # ------------------------------------

    threshold = FREQUENCY_STRENGTH * importance

    LH[np.abs(LH) < threshold] = 0

    HL[np.abs(HL) < threshold] = 0

    HH[np.abs(HH) < threshold] = 0

    return (LH, HL, HH)


# ==========================================================
# Multi-Level DWT Attack
# ==========================================================

def apply_frequency_perturbation(
    image,
    importance_map
):

    ycrcb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2YCrCb
    )

    Y = ycrcb[:, :, 0].astype(np.float32)

    # ------------------------------------
    # Two-Level DWT
    # ------------------------------------

    coeffs = pywt.wavedec2(
        Y,
        wavelet=WAVELET,
        level=2
    )

    LL = coeffs[0]

    level2 = coeffs[1]

    level1 = coeffs[2]

    # ------------------------------------
    # Attack Level 2
    # ------------------------------------

    level2 = attack_level(
        level2,
        importance_map
    )

    # ------------------------------------
    # Attack Level 1
    # ------------------------------------

    level1 = attack_level(
        level1,
        importance_map
    )

    # ------------------------------------
    # Reconstruct
    # ------------------------------------

    reconstructed = pywt.waverec2(
        [
            LL,
            level2,
            level1
        ],
        WAVELET
    )

    reconstructed = np.clip(
        reconstructed,
        0,
        255
    )

    reconstructed = reconstructed.astype(
        np.uint8
    )

    reconstructed = reconstructed[
        :Y.shape[0],
        :Y.shape[1]
    ]

    ycrcb[:, :, 0] = reconstructed

    output = cv2.cvtColor(
        ycrcb,
        cv2.COLOR_YCrCb2BGR
    )

    return output
