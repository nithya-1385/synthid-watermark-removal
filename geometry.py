
"""
geometry.py

Geometric attacks for watermark weakening.
"""

import cv2


# ==========================================================
# Crop + Resize
# ==========================================================

def crop_resize_attack(image, crop_ratio=0.05):
    """
    Crop border then resize back.
    """

    h, w = image.shape[:2]

    dx = int(w * crop_ratio)
    dy = int(h * crop_ratio)

    cropped = image[
        dy:h-dy,
        dx:w-dx
    ]

    resized = cv2.resize(
        cropped,
        (w, h),
        interpolation=cv2.INTER_CUBIC
    )

    return resized


# ==========================================================
# Scale Cycle
# ==========================================================

def scale_cycle_attack(
    image,
    scale=0.75
):
    """
    Downscale then upscale.
    """

    h, w = image.shape[:2]

    small = cv2.resize(
        image,
        (
            int(w*scale),
            int(h*scale)
        ),
        interpolation=cv2.INTER_AREA
    )

    restored = cv2.resize(
        small,
        (w,h),
        interpolation=cv2.INTER_CUBIC
    )

    return restored


# ==========================================================
# Slight Rotation
# ==========================================================

def rotation_attack(
    image,
    angle=2
):

    h,w = image.shape[:2]

    M = cv2.getRotationMatrix2D(
        (w//2,h//2),
        angle,
        1.0
    )

    rotated = cv2.warpAffine(

        image,

        M,

        (w,h),

        borderMode=cv2.BORDER_REFLECT

    )

    return rotated
