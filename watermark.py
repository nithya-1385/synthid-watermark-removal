
"""
watermark.py

TrustMark Embedding and Decoding
"""

from trustmark import TrustMark

from config import (
    PAYLOAD,
    WM_STRENGTH
)

from utils import (
    opencv_to_pil,
    pil_to_opencv
)


# ==========================================================
# Initialize TrustMark
# ==========================================================

def initialize_trustmark():
    """
    Initialize TrustMark-Q model.
    """

    tm = TrustMark(
        verbose=False,
        model_type="Q",
        use_ECC=True
    )

    return tm


# ==========================================================
# Embed Watermark
# ==========================================================

def embed_watermark(
    tm,
    image,
    payload=PAYLOAD
):
    """
    Embed watermark into image.

    Input:
        OpenCV Image (BGR)

    Output:
        Watermarked OpenCV Image
    """

    pil_image = opencv_to_pil(image)

    watermarked = tm.encode(
        pil_image,
        payload,
        MODE="text",
        WM_STRENGTH=WM_STRENGTH
    )

    return pil_to_opencv(
        watermarked
    )


# ==========================================================
# Decode Watermark
# ==========================================================

def decode_watermark(
    tm,
    image
):
    """
    Decode watermark.

    Returns:
        payload
        detected
        version
    """

    pil_image = opencv_to_pil(image)

    payload, detected, version = tm.decode(
        pil_image,
        MODE="text"
    )

    return {

        "payload": payload,

        "detected": detected,

        "version": version

    }


# ==========================================================
# Verify Embedding
# ==========================================================

def verify_embedding(
    tm,
    image
):
    """
    Decode immediately after embedding.
    """

    result = decode_watermark(
        tm,
        image
    )

    print("\n")
    print("="*45)
    print(" TRUSTMARK VERIFICATION ")
    print("="*45)

    print("Detected :", result["detected"])
    print("Payload  :", result["payload"])
    print("Version  :", result["version"])

    print("="*45)

    return result
