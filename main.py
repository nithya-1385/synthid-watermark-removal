
"""
main.py

Adaptive Watermark Weakening Pipeline
"""

from google.colab import files

from utils import load_image

from watermark import (
    initialize_trustmark,
    embed_watermark,
    verify_embedding
)

from analysis import analyze_image

from adaptive_noise import (
    apply_adaptive_gaussian,
    apply_adaptive_speckle,
    apply_adaptive_salt_pepper
)

from frequency import apply_frequency_perturbation

from compression import apply_jpeg_compression

from evaluation import (
    evaluate_all_attacks,
    print_report
)

from visualization import (
    show_best_attack,
    show_original_vs_best
)

from geometry import *

# ==========================================================
# Upload Image
# ==========================================================

print("="*60)
print("UPLOAD IMAGE")
print("="*60)

uploaded = files.upload()

image_path = list(uploaded.keys())[0]

original = load_image(image_path)

# ==========================================================
# Initialize TrustMark
# ==========================================================

tm = initialize_trustmark()

# ==========================================================
# Embed Watermark
# ==========================================================

print("\nEmbedding TrustMark...")

watermarked = embed_watermark(
    tm,
    original
)

# ==========================================================
# Verify
# ==========================================================

verify_embedding(
    tm,
    watermarked
)

# ==========================================================
# Image Analysis
# ==========================================================

print("\nGenerating Importance Map...")

maps = analyze_image(
    watermarked
)

# ==========================================================
# Generate Attacks
# ==========================================================

print("\nGenerating Attacks...")

attacks = {}

# ==========================================================
# Generate Attacks
# ==========================================================

print("\nGenerating Attacks...")

attacks = {}

jpeg_qualities = [
    95,
    90,
    85,
    70,
    60,
    50,
    40,
    30
]

crop_levels = [
    0.02,
    0.04,
    0.06,
    0.08
]

scale_levels = [
    0.90,
    0.80,
    0.75,
    0.70
]

rotation_angles = [
    -3,
    -2,
    2,
    3
]

double_jpeg = [

    (20,95),

    (20,90),

    (20,80),

    (30,95),

    (30,90),

    (30,80),

    (40,95),

    (40,90)

]

# ==========================================================
# Helper Function
# ==========================================================

def generate_pipeline(
    name,
    noisy_image
):

    dwt = apply_frequency_perturbation(
        noisy_image,
        maps["importance"]
    )

    attacks[name] = noisy_image

    attacks[f"{name} + DWT"] = dwt


    # -----------------------------------------
    # Single JPEG
    # -----------------------------------------

    for q in jpeg_qualities:

        attacks[
            f"{name} + DWT + JPEG{q}"
        ] = apply_jpeg_compression(
            dwt,
            quality=q
        )


    # -----------------------------------------
    # Double JPEG
    # -----------------------------------------

    for q1,q2 in double_jpeg:

        first = apply_jpeg_compression(
            dwt,
            quality=q1
        )

        second = apply_jpeg_compression(
            first,
            quality=q2
        )

        attacks[
            f"{name} + DWT + JPEG{q1}->{q2}"
        ] = second


    # =====================================================
    # Adaptive Crop
    # =====================================================
    for crop_ratio in crop_levels:
      crop = crop_resize_attack(
          dwt,
          crop_ratio=crop_ratio
      )
      label = int(crop_ratio * 100)
      attacks[
          f"{name} + Crop{label}%"
      ] = crop

      attacks[
          f"{name} + Crop{label}% + JPEG30"
      ] = apply_jpeg_compression(
          crop,
          quality=30
          )

      attacks[
          f"{name} + Crop{label}% + JPEG30->80"
      ] = apply_jpeg_compression(
          apply_jpeg_compression(
              crop,
              quality=30
          ),
          quality=80
          )

    # =====================================================
    # Adaptive Scale
    # =====================================================
    for scale_factor in scale_levels:
      scale = scale_cycle_attack(
          dwt,
          scale=scale_factor
      )
      label = int(scale_factor * 100)
      attacks[
          f"{name} + Scale{label}"
      ] = scale

      attacks[
          f"{name} + Scale{label} + JPEG30"
      ] = apply_jpeg_compression(
          scale,
          quality=30
          )

      attacks[
          f"{name} + Scale{label} + JPEG30->80"
      ] = apply_jpeg_compression(
          apply_jpeg_compression(
              scale,
              quality=30
          ),
          quality=80
          )

    # =====================================================
    # Adaptive Rotation
    # =====================================================
    for angle in rotation_angles:
      rotate = rotation_attack(
          dwt,
          angle=angle
      )

      attacks[
          f"{name} + Rotate{angle}"
      ] = rotate

      attacks[
          f"{name} + Rotate{angle} + JPEG30"
      ] = apply_jpeg_compression(
          rotate,
          quality=30
          )

      attacks[
          f"{name} + Rotate{angle} + JPEG30->80"
      ] = apply_jpeg_compression(
          apply_jpeg_compression(
              rotate,
              quality=30
          ),
          quality=80
          )

# ==========================================================
# Adaptive Gaussian
# ==========================================================

gaussian = apply_adaptive_gaussian(

    watermarked,

    maps["importance"]

)

generate_pipeline(

    "Gaussian",

    gaussian

)


# ==========================================================
# Adaptive Speckle
# ==========================================================

speckle = apply_adaptive_speckle(

    watermarked,

    maps["importance"]

)

generate_pipeline(

    "Speckle",

    speckle

)


# ==========================================================
# Adaptive Salt & Pepper
# ==========================================================

saltpepper = apply_adaptive_salt_pepper(

    watermarked,

    maps["importance"]

)

generate_pipeline(

    "SaltPepper",

    saltpepper

)

print(f"\nTotal attacks generated : {len(attacks)}")

# ==========================================================
# Evaluate
# ==========================================================

print("\nEvaluating...")

reports = evaluate_all_attacks(
    tm,
    watermarked,
    attacks
)

# ==========================================================
# Print Results
# ==========================================================

print_report(
    reports
)

# ==========================================================
# Select Best Attack
# ==========================================================

best_name = None

removed = []

for name, report in reports.items():

    if report["Detected"] == False:

        removed.append((name, report))

# ----------------------------------------------------------
# If watermark removed
# ----------------------------------------------------------

if len(removed) > 0:

    removed.sort(

        key=lambda x: (

            x[1]["SSIM"],

            x[1]["PSNR"]

        ),

        reverse=True

    )

    best_name = removed[0][0]

# ----------------------------------------------------------
# Otherwise choose lowest payload match
# ----------------------------------------------------------

else:

    weakest = sorted(

        reports.items(),

        key=lambda x: (

            x[1]["Payload Match (%)"],

            -x[1]["SSIM"],

            -x[1]["PSNR"]

        )

    )

    best_name = weakest[0][0]

best_report = reports[best_name]

best_image = attacks[best_name]

print("\n")
print("="*60)
print("BEST ATTACK")
print("="*60)
print(best_name)

# ==========================================================
# Visualization
# ==========================================================

show_best_attack(

    original,

    watermarked,

    best_image,

    best_name,

    best_report

)

show_original_vs_best(

    original,

    best_image,

    best_name,

    best_report

)

# ==========================================================
# Finished
# ==========================================================

print("\n")
print("="*60)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("="*60)
