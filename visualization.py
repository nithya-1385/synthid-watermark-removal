
"""
visualization.py

Visualization Module
"""

import os
import cv2
import matplotlib.pyplot as plt

from config import OUTPUT_DIR


# ==========================================================
# Create Output Folder
# ==========================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# BGR -> RGB
# ==========================================================

def bgr_to_rgb(image):

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


# ==========================================================
# Original vs Watermarked vs Best Output
# ==========================================================

def show_best_attack(
    original,
    watermarked,
    best_image,
    best_name,
    report
):

    original = bgr_to_rgb(original)
    watermarked = bgr_to_rgb(watermarked)
    best_image = bgr_to_rgb(best_image)

    plt.figure(figsize=(18,6))

    # --------------------------------

    plt.subplot(1,3,1)

    plt.imshow(original)

    plt.title("Original Image")

    plt.axis("off")

    # --------------------------------

    plt.subplot(1,3,2)

    plt.imshow(watermarked)

    plt.title("Watermarked Image")

    plt.axis("off")

    # --------------------------------

    plt.subplot(1,3,3)

    plt.imshow(best_image)

    plt.title(

        f"{best_name}\n\n"

        f"Detected : {report['Detected']}\n"

        f"Payload Match : {report['Payload Match (%)']:.2f}%\n"

        f"PSNR : {report['PSNR']:.2f}\n"

        f"SSIM : {report['SSIM']:.4f}"

    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            OUTPUT_DIR,
            "Original_Watermarked_Best.png"
        ),

        dpi=300

    )

    plt.show()


# ==========================================================
# Original vs Final Output
# ==========================================================

def show_original_vs_best(
    original,
    best_image,
    best_name,
    report
):

    original = bgr_to_rgb(original)
    best_image = bgr_to_rgb(best_image)

    plt.figure(figsize=(12,6))

    # --------------------------------

    plt.subplot(1,2,1)

    plt.imshow(original)

    plt.title("Original Image")

    plt.axis("off")

    # --------------------------------

    plt.subplot(1,2,2)

    plt.imshow(best_image)

    plt.title(

        f"Final Output\n\n"

        f"{best_name}\n"

        f"Detected : {report['Detected']}\n"

        f"Payload Match : {report['Payload Match (%)']:.2f}%\n"

        f"PSNR : {report['PSNR']:.2f}\n"

        f"SSIM : {report['SSIM']:.4f}"

    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            OUTPUT_DIR,
            "Original_vs_Final.png"
        ),

        dpi=300

    )

    plt.show()
