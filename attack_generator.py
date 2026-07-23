
"""
attack_generator.py

Generates multiple watermark weakening attacks.

Pipeline

Watermarked Image
        │
        ▼
Importance Map
        │
        ▼
Generate Multiple Attack Variants
        │
        ├── Noise
        ├── Noise + DWT
        ├── Noise + JPEG95
        ├── Noise + DWT + JPEG95
        ├── Noise + DWT + JPEG90
        └── Noise + DWT + JPEG85
"""

from adaptive_noise import apply_adaptive_noise
from frequency import apply_frequency_perturbation
from compression import apply_jpeg_compression

from config import (
    JPEG_QUALITY_HIGH,
    JPEG_QUALITY_MEDIUM,
    JPEG_QUALITY_LOW
)


# ==========================================================
# Attack Generator
# ==========================================================

def generate_attacks(
    watermarked_image,
    importance_map
):
    """
    Returns dictionary containing
    all attack outputs.
    """

    attacks = {}

    # ------------------------------------------
    # Attack 1
    # Adaptive Noise
    # ------------------------------------------

    noise = apply_adaptive_noise(
        watermarked_image,
        importance_map
    )

    attacks["Adaptive Noise"] = noise

    # ------------------------------------------
    # Attack 2
    # Noise + DWT
    # ------------------------------------------

    noise_dwt = apply_frequency_perturbation(
        noise,
        importance_map
    )

    attacks["Noise + DWT"] = noise_dwt

    # ------------------------------------------
    # Attack 3
    # Noise + JPEG95
    # ------------------------------------------

    noise95 = apply_jpeg_compression(
        noise,
        JPEG_QUALITY_HIGH
    )

    attacks["Noise + JPEG95"] = noise95

    # ------------------------------------------
    # Attack 4
    # Noise + DWT + JPEG95
    # ------------------------------------------

    attack95 = apply_jpeg_compression(
        noise_dwt,
        JPEG_QUALITY_HIGH
    )

    attacks["Noise + DWT + JPEG95"] = attack95

    # ------------------------------------------
    # Attack 5
    # Noise + DWT + JPEG90
    # ------------------------------------------

    attack90 = apply_jpeg_compression(
        noise_dwt,
        JPEG_QUALITY_MEDIUM
    )

    attacks["Noise + DWT + JPEG90"] = attack90

    # ------------------------------------------
    # Attack 6
    # Noise + DWT + JPEG85
    # ------------------------------------------

    attack85 = apply_jpeg_compression(
        noise_dwt,
        JPEG_QUALITY_LOW
    )

    attacks["Noise + DWT + JPEG85"] = attack85

    return attacks
