
"""
evaluation.py

Evaluate all generated attacks.

Metrics

1. TrustMark Detection
2. Payload Match
3. PSNR
4. SSIM
"""

import cv2

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from watermark import (
    decode_watermark
)

from config import PAYLOAD


# ==========================================================
# Payload Accuracy
# ==========================================================

def payload_accuracy(
    original,
    decoded
):
    """
    Character-level payload accuracy.
    """

    original = str(original)
    decoded = str(decoded)

    if len(original) == 0:
        return 0

    correct = 0

    for a, b in zip(original, decoded):

        if a == b:
            correct += 1

    return (
        correct /
        len(original)
    ) * 100


# ==========================================================
# Evaluate Single Attack
# ==========================================================

def evaluate_attack(
    tm,
    reference_image,
    attacked_image
):
    """
    Evaluate one attack.
    """

    result = decode_watermark(
        tm,
        attacked_image
    )

    psnr = peak_signal_noise_ratio(

        reference_image,

        attacked_image,

        data_range=255

    )

    ssim = structural_similarity(

        cv2.cvtColor(
            reference_image,
            cv2.COLOR_BGR2GRAY
        ),

        cv2.cvtColor(
            attacked_image,
            cv2.COLOR_BGR2GRAY
        ),

        data_range=255

    )

    payload_match = payload_accuracy(

        PAYLOAD,

        result["payload"]

    )

    return {

        "Detected": result["detected"],

        "Payload": result["payload"],

        "Payload Match (%)": round(
            payload_match,
            2
        ),

        "PSNR": round(
            psnr,
            2
        ),

        "SSIM": round(
            ssim,
            4
        ),

        "Version": result["version"]

    }


# ==========================================================
# Evaluate All Attacks
# ==========================================================

def evaluate_all_attacks(
    tm,
    reference_image,
    attacks
):
    """
    Evaluate every attack.
    """

    report = {}

    for attack_name, image in attacks.items():

        report[attack_name] = evaluate_attack(

            tm,

            reference_image,

            image

        )

    return report


# ==========================================================
# Print Report
# ==========================================================

def print_report(
    report
):

    print()

    print("=" * 70)

    print(" WATERMARK WEAKENING RESULTS ")

    print("=" * 70)

    for attack_name, metrics in report.items():

        print()

        print(attack_name)

        print("-" * 70)

        for key, value in metrics.items():

            print(f"{key:22s}: {value}")

    print("=" * 70)


# ==========================================================
# Best Attack
# ==========================================================

def get_best_attack(
    report
):
    """
    Select best attack.

    Priority

    1. Watermark removed
    2. Highest SSIM
    """

    best_name = None

    best_metrics = None

    best_ssim = -1

    for attack, metrics in report.items():

        if metrics["Detected"] == False:

            if metrics["SSIM"] > best_ssim:

                best_ssim = metrics["SSIM"]

                best_name = attack

                best_metrics = metrics

    if best_name is None:

        for attack, metrics in report.items():

            if metrics["SSIM"] > best_ssim:

                best_ssim = metrics["SSIM"]

                best_name = attack

                best_metrics = metrics

    return best_name, best_metrics
