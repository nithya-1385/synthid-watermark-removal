
"""
compression.py

JPEG Compression Module

Used by the Attack Generator to create
multiple JPEG attack variants.

Example

Input Image
      │
      ▼
JPEG Quality = 95
      │
      ▼
Compressed Image
"""

import cv2


# ==========================================================
# JPEG Compression
# ==========================================================

def apply_jpeg_compression(
    image,
    quality
):
    """
    JPEG compression with user-defined quality.
    """

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [
            int(cv2.IMWRITE_JPEG_QUALITY),
            quality
        ]
    )

    if not success:
        raise RuntimeError("JPEG encoding failed.")

    compressed = cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR
    )

    return compressed
