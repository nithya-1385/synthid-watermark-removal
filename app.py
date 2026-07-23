import spaces
import os
import cv2
import numpy as np
import pywt
import gradio as gr
from PIL import Image
from scipy.ndimage import generic_filter
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from trustmark import TrustMark
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
PAYLOAD            = "HELLO123"
WM_STRENGTH        = 1.0
TEXTURE_WEIGHT     = 0.40
GRADIENT_WEIGHT    = 0.30
VARIANCE_WEIGHT    = 0.30
MIN_SIGMA          = 1.0
MAX_SIGMA          = 6.0
WAVELET            = "haar"
FREQUENCY_STRENGTH = 4.0
np.random.seed(42)

TM = None  # lazy-loaded inside @spaces.GPU

# =============================================================================
# UTILS
# =============================================================================
def opencv_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def pil_to_opencv(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def normalize_map(data):
    data = data.astype(np.float32)
    return cv2.normalize(data, None, 0, 1, cv2.NORM_MINMAX)

# =============================================================================
# WATERMARK
# =============================================================================
def embed_watermark(tm, image):
    watermarked = tm.encode(opencv_to_pil(image), PAYLOAD, MODE="text", WM_STRENGTH=WM_STRENGTH)
    return pil_to_opencv(watermarked)

def decode_watermark(tm, image):
    payload, detected, version = tm.decode(opencv_to_pil(image), MODE="text")
    return {"payload": payload, "detected": bool(detected), "version": version}

# =============================================================================
# ANALYSIS
# =============================================================================
def analyze_image(image):
    gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture  = normalize_map(generic_filter(gray.astype(np.float32), np.std, size=7))
    gx       = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy       = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = normalize_map(cv2.magnitude(gx, gy))
    variance = normalize_map(generic_filter(gray.astype(np.float32), np.var, size=7))
    importance = normalize_map(
        TEXTURE_WEIGHT * texture +
        GRADIENT_WEIGHT * gradient +
        VARIANCE_WEIGHT * variance
    )
    return {"texture": texture, "gradient": gradient,
            "variance": variance, "importance": importance}

# =============================================================================
# ADAPTIVE NOISE
# =============================================================================
def compute_sigma_map(importance_map):
    return MIN_SIGMA + importance_map * (MAX_SIGMA - MIN_SIGMA)

def apply_adaptive_gaussian(image, importance_map):
    image = image.astype(np.float32)
    sigma = compute_sigma_map(importance_map)[:, :, np.newaxis]
    noise = np.random.normal(0, 1, image.shape).astype(np.float32)
    return np.clip(image + noise * sigma, 0, 255).astype(np.uint8)

def apply_adaptive_speckle(image, importance_map):
    image = image.astype(np.float32)
    sigma = compute_sigma_map(importance_map)[:, :, np.newaxis]
    noise = np.random.normal(0, 1, image.shape).astype(np.float32)
    return np.clip(image + image * noise * sigma / 255.0, 0, 255).astype(np.uint8)

def apply_adaptive_salt_pepper(image, importance_map):
    noisy       = image.copy()
    probability = importance_map * 0.04
    random_map  = np.random.rand(image.shape[0], image.shape[1])
    noisy[random_map > (1 - probability / 2)] = 255
    noisy[random_map < (probability / 2)]     = 0
    return noisy

# =============================================================================
# FREQUENCY
# =============================================================================
def attack_level(coeffs, importance):
    LH, HL, HH = coeffs
    imp    = cv2.resize(importance, (LH.shape[1], LH.shape[0]), interpolation=cv2.INTER_LINEAR)
    atten  = 1 - (0.35 * imp)
    LH     = LH * atten + np.random.normal(0, FREQUENCY_STRENGTH, LH.shape) * imp
    HL     = HL * atten + np.random.normal(0, FREQUENCY_STRENGTH, HL.shape) * imp
    HH     = HH * atten + np.random.normal(0, FREQUENCY_STRENGTH, HH.shape) * imp
    thresh = FREQUENCY_STRENGTH * imp
    LH[np.abs(LH) < thresh] = 0
    HL[np.abs(HL) < thresh] = 0
    HH[np.abs(HH) < thresh] = 0
    return (LH, HL, HH)

def apply_frequency_perturbation(image, importance_map):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y     = ycrcb[:, :, 0].astype(np.float32)
    coeffs = pywt.wavedec2(Y, wavelet=WAVELET, level=2)
    rec = pywt.waverec2(
        [coeffs[0],
         attack_level(list(coeffs[1]), importance_map),
         attack_level(list(coeffs[2]), importance_map)],
        WAVELET
    )
    rec = np.clip(rec, 0, 255).astype(np.uint8)[:Y.shape[0], :Y.shape[1]]
    ycrcb[:, :, 0] = rec
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

# =============================================================================
# COMPRESSION
# =============================================================================
def apply_jpeg_compression(image, quality):
    success, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise RuntimeError("JPEG encoding failed.")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

# =============================================================================
# GEOMETRY
# =============================================================================
def crop_resize_attack(image, crop_ratio=0.05):
    h, w   = image.shape[:2]
    dx, dy = int(w * crop_ratio), int(h * crop_ratio)
    return cv2.resize(image[dy:h-dy, dx:w-dx], (w, h), interpolation=cv2.INTER_CUBIC)

def scale_cycle_attack(image, scale=0.75):
    h, w  = image.shape[:2]
    small = cv2.resize(image, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

def rotation_attack(image, angle=2):
    h, w = image.shape[:2]
    M    = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

# =============================================================================
# ATTACK PIPELINE (from main.py — generate_pipeline)
# =============================================================================
jpeg_qualities  = [95, 90, 85, 70, 60, 50, 40, 30]
crop_levels     = [0.02, 0.04, 0.06, 0.08]
scale_levels    = [0.90, 0.80, 0.75, 0.70]
rotation_angles = [-3, -2, 2, 3]
double_jpeg     = [(20,95),(20,90),(20,80),(30,95),(30,90),(30,80),(40,95),(40,90)]

def generate_pipeline(name, noisy_image, importance_map, attacks):
    dwt = apply_frequency_perturbation(noisy_image, importance_map)
    attacks[name]            = noisy_image
    attacks[f"{name} + DWT"] = dwt

    for q in jpeg_qualities:
        attacks[f"{name} + DWT + JPEG{q}"] = apply_jpeg_compression(dwt, quality=q)

    for q1, q2 in double_jpeg:
        attacks[f"{name} + DWT + JPEG{q1}->{q2}"] = apply_jpeg_compression(
            apply_jpeg_compression(dwt, quality=q1), quality=q2)

    for crop_ratio in crop_levels:
        crop  = crop_resize_attack(dwt, crop_ratio=crop_ratio)
        label = int(crop_ratio * 100)
        attacks[f"{name} + Crop{label}%"]          = crop
        attacks[f"{name} + Crop{label}% + JPEG30"] = apply_jpeg_compression(crop, 30)
        attacks[f"{name} + Crop{label}% + JPEG30->80"] = apply_jpeg_compression(
            apply_jpeg_compression(crop, 30), 80)

    for scale_factor in scale_levels:
        scale = scale_cycle_attack(dwt, scale=scale_factor)
        label = int(scale_factor * 100)
        attacks[f"{name} + Scale{label}"]              = scale
        attacks[f"{name} + Scale{label} + JPEG30"]     = apply_jpeg_compression(scale, 30)
        attacks[f"{name} + Scale{label} + JPEG30->80"] = apply_jpeg_compression(
            apply_jpeg_compression(scale, 30), 80)

    for angle in rotation_angles:
        rot = rotation_attack(dwt, angle=angle)
        attacks[f"{name} + Rotate{angle}"]              = rot
        attacks[f"{name} + Rotate{angle} + JPEG30"]     = apply_jpeg_compression(rot, 30)
        attacks[f"{name} + Rotate{angle} + JPEG30->80"] = apply_jpeg_compression(
            apply_jpeg_compression(rot, 30), 80)

# =============================================================================
# EVALUATION (from evaluation.py — get_best_attack logic from main.py)
# =============================================================================
def evaluate_attack(tm, reference_image, attacked_image):
    result = decode_watermark(tm, attacked_image)
    psnr   = peak_signal_noise_ratio(reference_image, attacked_image, data_range=255)
    ssim   = structural_similarity(
        cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(attacked_image,  cv2.COLOR_BGR2GRAY),
        data_range=255
    )
    return {
        "Detected":          result["detected"],
        "Payload":           result["payload"],
        "Payload Match (%)": round(sum(a==b for a,b in zip(PAYLOAD, str(result["payload"])))
                                   / len(PAYLOAD) * 100, 2),
        "PSNR":  round(psnr, 2),
        "SSIM":  round(ssim, 4),
    }

# =============================================================================
# MAIN RUN FUNCTION
# =============================================================================
@spaces.GPU(duration=120)
def run(image):
    global TM
    if TM is None:
        print("Loading TrustMark-Q...")
        TM = TrustMark(verbose=False, model_type="Q", use_ECC=True)
        print("Ready.")

    if image is None:
        return None, None, "⚠️ Please upload an image."

    # Convert PIL → OpenCV BGR (keep original resolution)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    original = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    # Embed watermark
    print("Embedding watermark...")
    watermarked = embed_watermark(TM, original)

    # Verify embedding
    verify = decode_watermark(TM, watermarked)
    print(f"Embedded — Detected: {verify['detected']}  Payload: {verify['payload']}")

    # Analyse image
    print("Analysing image...")
    maps = analyze_image(watermarked)

    # Generate all attacks
    print("Generating attacks...")
    attacks = {}
    generate_pipeline("Gaussian",
        apply_adaptive_gaussian(watermarked, maps["importance"]),
        maps["importance"], attacks)
    generate_pipeline("Speckle",
        apply_adaptive_speckle(watermarked, maps["importance"]),
        maps["importance"], attacks)
    generate_pipeline("SaltPepper",
        apply_adaptive_salt_pepper(watermarked, maps["importance"]),
        maps["importance"], attacks)

    print(f"Total attacks generated: {len(attacks)}")

    # Evaluate all attacks
    print("Evaluating...")
    reports = {}
    for attack_name, att_image in attacks.items():
        try:
            reports[attack_name] = evaluate_attack(TM, watermarked, att_image)
        except:
            pass

    # Select best (from main.py logic)
    removed = [(n, r) for n, r in reports.items() if r["Detected"] == False]
    if removed:
        removed.sort(key=lambda x: (x[1]["SSIM"], x[1]["PSNR"]), reverse=True)
        best_name   = removed[0][0]
        best_report = removed[0][1]
    else:
        weakest   = sorted(reports.items(),
                           key=lambda x: (x[1]["Payload Match (%)"],
                                          -x[1]["SSIM"], -x[1]["PSNR"]))
        best_name   = weakest[0][0]
        best_report = weakest[0][1]

    best_image = attacks[best_name]
    print(f"Best attack: {best_name}")

    # Build result
    wm_pil  = opencv_to_pil(watermarked)
    out_pil = opencv_to_pil(best_image)

    removed_flag = not best_report["Detected"]
    ssim_val     = best_report["SSIM"]
    psnr_val     = best_report["PSNR"]
    payload_m    = best_report["Payload Match (%)"]

    both    = removed_flag and ssim_val >= 0.85
    nl      = removed_flag and ssim_val >= 0.97
    status  = ("✅ BOTH CRITERIA MET"     if both         else
               "✅ Removed (low quality)" if removed_flag else
               "❌ Watermark Not Removed")
    quality = ("Near-lossless ✅" if nl          else
               "OK ✅"            if ssim_val >= 0.85 else
               "Degraded ⚠️")

    md = f"""## {status}

| Metric | Value |
|---|---|
| **Best Attack** | `{best_name}` |
| **Detected** | {"NO ✅" if removed_flag else "YES ❌"} |
| **Payload Match** | `{payload_m:.2f}%` |
| **SSIM** | `{ssim_val:.4f}` |
| **PSNR** | `{psnr_val:.1f} dB` |
| **Quality** | {quality} |

*Proxy: TrustMark-Q as surrogate for SynthID (Gowal et al., 2025)*
"""
    return wm_pil, out_pil, md

# =============================================================================
# GRADIO UI — Dark theme
# =============================================================================
css = """
    body, .gradio-container {
        background: #0f0f0f !important;
        color: #f0f0f0 !important;
        font-family: 'Inter', sans-serif;
    }
    .header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #c9748a;
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 16px;
    }
    .header h1 { color: #e8a0b0 !important; margin: 0 0 8px 0; font-size: 1.8rem; }
    .header p  { color: #aaa !important; margin: 0; font-size: 0.9rem; }
    .card {
        background: #1a1a1a !important;
        border: 1px solid #333 !important;
        border-radius: 14px !important;
        padding: 20px !important;
    }
    .run-btn {
        background: linear-gradient(135deg, #c9748a, #b05070) !important;
        border: none !important;
        border-radius: 10px !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 15px rgba(201,116,138,0.4) !important;
    }
    .stats {
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 16px 24px;
        display: flex;
        gap: 32px;
        flex-wrap: wrap;
        margin-top: 12px;
        align-items: center;
    }
    .sv { font-size: 1.5rem; font-weight: 700; color: #c9748a; }
    .sl { font-size: 0.72rem; color: #666; margin-top: 2px; }
    label, .label-wrap span { color: #ccc !important; }
    .prose p, .prose h1, .prose h2, .prose h3 { color: #f0f0f0 !important; }
    table { color: #f0f0f0 !important; }
    th { background: #2a2a2a !important; color: #e8a0b0 !important; }
    td { border-color: #333 !important; }
    footer { display: none !important; }
"""

with gr.Blocks(
    title="SynthID Watermark Removal",
    theme=gr.themes.Base(
        primary_hue="rose",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=css
) as demo:

    gr.HTML("""
    <div class="header">
        <h1>🔬 SynthID Watermark Removal</h1>
        <p>
            Adaptive Watermark Weakening Pipeline &nbsp;·&nbsp;
            PES University, Bengaluru (RR Campus) &nbsp;·&nbsp;
            No custom neural network &nbsp;·&nbsp; Classical signal processing only
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            with gr.Group(elem_classes="card"):
                img_in  = gr.Image(label="Upload Image", type="pil")
                with gr.Accordion("How it works", open=False):
                    gr.Markdown("""
**Pipeline:**
1. Embed TrustMark watermark
2. Compute importance map (texture + gradient + variance)
3. Apply adaptive noise (Gaussian / Speckle / Salt & Pepper)
4. DWT frequency perturbation on Y channel
5. Geometric attacks (crop, scale, rotation) + JPEG combos
6. Evaluate ~150 attack variants
7. Return best result by SSIM + detection status
                    """)
                run_btn = gr.Button(
                    "▶  Run Pipeline", elem_classes="run-btn", size="lg"
                )

        with gr.Column(scale=2):
            with gr.Group(elem_classes="card"):
                with gr.Row():
                    wm_out  = gr.Image(label="🔒 Watermarked")
                    out_img = gr.Image(label="🔓 Best Attack Output")
                result_md = gr.Markdown("*Upload an image and click Run.*")

    gr.HTML("""
    <div class="stats">
        <div><div class="sv">95.4%</div><div class="sl">Removal Rate</div></div>
        <div><div class="sv">15.0%</div><div class="sl">Quality-Preserving<br><small>(SSIM ≥ 0.85)</small></div></div>
        <div><div class="sv">100%</div><div class="sl">Precision</div></div>
        <div><div class="sv">~150</div><div class="sl">Attack Variants</div></div>
        <div style="margin-left:auto;font-size:0.75rem;color:#555;line-height:1.8">
            Proxy: TrustMark-Q (Adobe Research)<br>
            Target: SynthID (Google DeepMind)<br>
            Gowal et al., arXiv:2510.09263, 2025
        </div>
    </div>
    """)

    run_btn.click(fn=run, inputs=[img_in], outputs=[wm_out, out_img, result_md])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))