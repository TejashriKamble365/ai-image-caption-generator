import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

import time
import tempfile
import json
import inspect
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
import gradio as gr

# ---------------------------------------------------------
# Device Configuration & Model Initialization
# ---------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Salesforce/blip-image-captioning-base"

print(f"Loading BLIP model '{MODEL_ID}' on device: {DEVICE}...")
processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()
print("Model loaded successfully.")

# ---------------------------------------------------------
# Core Captioning & Audio Synthesis Logic
# ---------------------------------------------------------
def generate_caption_pipeline(
    image: Image.Image,
    caption_mode: str,
    custom_prefix: str,
    num_beams: int,
    max_length: int,
    min_length: int,
    repetition_penalty: float,
    enable_tts: bool
):
    """
    Analyzes an input image using BLIP and returns the caption,
    audio narration, image metrics, and an exportable JSON report.
    """
    if image is None:
        return (
            "Please upload an image or select one of the sample images below.",
            None,
            "No image provided.",
            None,
            None
        )

    start_time = time.time()

    # Image metadata
    width, height = image.size
    aspect_ratio = f"{width}:{height} ({round(width / height, 2)}:1)"
    img_mode = image.mode

    # Ensure RGB format for BLIP
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Determine prompt conditioning
    text_prompt = None
    if caption_mode == "Prefix Guided" and custom_prefix.strip():
        text_prompt = custom_prefix.strip()
    elif caption_mode == "Detailed Photo":
        text_prompt = "a high quality detailed photograph of"
    elif caption_mode == "Artwork / Illustration":
        text_prompt = "an illustration depicting"

    # Preprocess inputs
    if text_prompt:
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(DEVICE)
    else:
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    # Generate token sequence
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            num_beams=int(num_beams),
            max_length=int(max_length),
            min_length=int(min_length),
            repetition_penalty=float(repetition_penalty),
            early_stopping=True
        )

    raw_caption = processor.decode(output_ids[0], skip_special_tokens=True).strip()

    # Clean and capitalize caption
    if raw_caption:
        caption = raw_caption[0].upper() + raw_caption[1:]
        if not caption.endswith(('.', '!', '?')):
            caption += '.'
    else:
        caption = "Unable to generate a clear description for this image."

    inference_time = round(time.time() - start_time, 2)

    # Audio synthesis (Text-to-Speech) for accessibility
    audio_path = None
    if enable_tts and caption:
        try:
            tts = gTTS(text=caption, lang='en', slow=False)
            temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(temp_audio_file.name)
            audio_path = temp_audio_file.name
        except Exception as e:
            print(f"TTS warning: {e}")
            audio_path = None

    # Telemetry and diagnostics summary
    metrics_markdown = (
        f"| Metric | Details |\n"
        f"| :--- | :--- |\n"
        f"| **Inference Time** | `{inference_time}s` on `{DEVICE.upper()}` |\n"
        f"| **Image Resolution** | `{width} × {height} px` |\n"
        f"| **Color Mode** | `{img_mode}` |\n"
        f"| **Decoding Strategy** | Beam Search (`num_beams={num_beams}`) |\n"
        f"| **Conditioning Mode** | `{caption_mode}` |\n"
        f"| **Target Model** | `{MODEL_ID}` |"
    )

    # Prepare export files
    report_dict = {
        "caption": caption,
        "model": MODEL_ID,
        "device": DEVICE,
        "inference_seconds": inference_time,
        "image_metadata": {
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "mode": img_mode
        },
        "generation_hyperparameters": {
            "mode": caption_mode,
            "prompt_prefix": text_prompt or "None (Unconditional)",
            "num_beams": num_beams,
            "max_length": max_length,
            "min_length": min_length,
            "repetition_penalty": repetition_penalty
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    }

    # Save JSON report
    temp_json = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json", encoding="utf-8")
    json.dump(report_dict, temp_json, indent=2)
    temp_json.close()

    # Save TXT report
    temp_txt = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt", encoding="utf-8")
    temp_txt.write(f"BLIP AI Image Caption: {caption}\nGenerated on {report_dict['timestamp']}\nModel: {MODEL_ID}\n")
    temp_txt.close()

    return caption, audio_path, metrics_markdown, temp_json.name, temp_txt.name


# ---------------------------------------------------------
# Custom UI Theme & Modern Styling
# ---------------------------------------------------------
custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

#header-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 9999px;
    background: #eef2ff;
    color: #4338ca;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.025em;
    border: 1px solid #c7d2fe;
    margin-bottom: 8px;
}

.caption-result-box {
    border: 2px solid #3b82f6 !important;
    border-radius: 12px !important;
    background-color: #f8fafc !important;
    font-size: 1.15rem !important;
    line-height: 1.6 !important;
    color: #0f172a !important;
}

.dark .caption-result-box {
    background-color: #1e293b !important;
    border-color: #60a5fa !important;
    color: #f1f5f9 !important;
}

.metric-card {
    background-color: #f1f5f9;
    padding: 12px 16px;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
}

.dark .metric-card {
    background-color: #0f172a;
    border-color: #1e293b;
}
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    text_size="md",
    spacing_size="md",
    radius_size="md"
)

# Textbox arguments across Gradio versions
tb_kwargs = {}
tb_params = inspect.signature(gr.Textbox.__init__).parameters
if "buttons" in tb_params:
    tb_kwargs["buttons"] = ["copy"]
elif "show_copy_button" in tb_params:
    tb_kwargs["show_copy_button"] = True

# Blocks constructor compatibility across Gradio 4, 5, 6
blocks_params = inspect.signature(gr.Blocks.__init__).parameters
blocks_kwargs = {"title": "AI Image Caption Generator | BLIP"}

is_hf_spaces = bool(os.getenv("SPACE_ID"))
launch_kwargs = {
    "server_name": "0.0.0.0" if is_hf_spaces else "127.0.0.1",
    "server_port": 7860,
    "show_error": True,
    "inbrowser": not is_hf_spaces,
    "share": True
}

if "theme" in blocks_params:
    blocks_kwargs["theme"] = theme
    blocks_kwargs["css"] = custom_css
else:
    launch_kwargs["theme"] = theme
    launch_kwargs["css"] = custom_css

# ---------------------------------------------------------
# Gradio Blocks Application Layout
# ---------------------------------------------------------
with gr.Blocks(**blocks_kwargs) as demo:

    # Header section
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
                <div style="padding: 12px 0 20px 0;">
                    <div id="header-badge">
                        <span>● Vision-Language Intelligence</span>
                        <span style="opacity: 0.5;">|</span>
                        <span>BLIP Architecture</span>
                        <span style="opacity: 0.5;">|</span>
                        <span>Accessibility-First</span>
                    </div>
                    <h1 style="font-size: 2.2rem; font-weight: 800; margin: 4px 0 8px 0; color: #1e293b; letter-spacing: -0.02em;">
                        AI Image Caption Generator
                    </h1>
                    <p style="font-size: 1.05rem; color: #64748b; margin: 0; max-width: 800px; line-height: 1.5;">
                        Upload any photograph or graphic to automatically generate natural-language descriptions 
                        using Salesforce's pre-trained BLIP model. Includes synthesized speech narration for visually impaired users.
                    </p>
                </div>
            """)

    # Main interaction section
    with gr.Row(equal_height=False):
        
        # Left column: Input and parameters
        with gr.Column(scale=5):
            with gr.Group():
                gr.Markdown("### 1. Upload Target Image")
                input_image = gr.Image(
                    type="pil",
                    label="Input Image (JPG, PNG, WebP)",
                    sources=["upload", "clipboard"],
                    height=360
                )
            
            with gr.Accordion("⚙️ Caption Generation Hyperparameters", open=False):
                caption_mode = gr.Radio(
                    choices=["Standard (Unconditional)", "Detailed Photo", "Artwork / Illustration", "Prefix Guided"],
                    value="Standard (Unconditional)",
                    label="Captioning Mode",
                    info="Choose whether BLIP captions autonomously or follows guided framing."
                )
                custom_prefix = gr.Textbox(
                    label="Custom Prefix (Optional)",
                    placeholder="e.g. 'a macro shot of' or 'the vintage texture of'",
                    visible=False
                )
                with gr.Row():
                    num_beams = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=5,
                        step=1,
                        label="Beam Search Size (num_beams)",
                        info="Higher values explore more paths for higher quality."
                    )
                    repetition_penalty = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=1.2,
                        step=0.1,
                        label="Repetition Penalty",
                        info="Penalizes repetitive token sequences."
                    )
                with gr.Row():
                    min_length = gr.Slider(
                        minimum=5,
                        maximum=30,
                        value=10,
                        step=1,
                        label="Min Length (tokens)"
                    )
                    max_length = gr.Slider(
                        minimum=15,
                        maximum=80,
                        value=45,
                        step=5,
                        label="Max Length (tokens)"
                    )
                enable_tts = gr.Checkbox(
                    value=True,
                    label="Generate Audio Narration (Text-to-Speech)",
                    info="Synthesizes voice audio for accessibility."
                )

            # Dynamic visibility for custom prefix
            def toggle_custom_prefix(mode):
                return gr.update(visible=(mode == "Prefix Guided"))

            caption_mode.change(
                fn=toggle_custom_prefix,
                inputs=[caption_mode],
                outputs=[custom_prefix]
            )

            with gr.Row():
                clear_btn = gr.Button("Clear All", variant="secondary")
                submit_btn = gr.Button("✨ Generate Caption", variant="primary", scale=2)

        # Right column: Outputs, Audio, Diagnostics
        with gr.Column(scale=6):
            with gr.Group():
                gr.Markdown("### 2. Generated Description")
                output_caption = gr.Textbox(
                    label="BLIP Natural Language Caption",
                    lines=3,
                    elem_classes=["caption-result-box"],
                    placeholder="Generated caption will appear here...",
                    **tb_kwargs
                )
                copy_btn = gr.Button("📋 Copy Caption to Clipboard", variant="secondary", size="sm")
                copy_btn.click(
                    fn=None,
                    inputs=[output_caption],
                    outputs=[],
                    js="""(caption) => {
                        if (caption) {
                            navigator.clipboard.writeText(caption);
                            alert('Caption copied to clipboard!');
                        }
                    }"""
                )
                audio_player = gr.Audio(
                    label="🔊 Speech Output (Accessibility Narration)",
                    type="filepath",
                    autoplay=False
                )

            with gr.Group():
                gr.Markdown("### 3. Image & Model Diagnostics")
                metrics_display = gr.Markdown(
                    value="_Inference diagnostics will populate here once an image is processed._"
                )

            with gr.Accordion("📥 Export Results", open=False):
                with gr.Row():
                    download_json = gr.File(label="Download JSON Metadata")
                    download_txt = gr.File(label="Download Plaintext Caption")

    # Example presets for quick evaluation during ISE viva
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 💡 Quick Test Presets (Instant Demo for Evaluators)")
            sample_dir = os.path.join(os.path.dirname(__file__), "sample_images")
            preset_samples = []
            if os.path.isdir(sample_dir):
                for f in sorted(os.listdir(sample_dir)):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        preset_samples.append([os.path.join(sample_dir, f)])

            if preset_samples:
                gr.Examples(
                    examples=preset_samples,
                    inputs=[input_image],
                    label="Click any sample image below to load immediately:"
                )

    # Technical Overview for ISE Examination
    with gr.Accordion("📘 ISE Academic Reference: Understanding BLIP Architecture", open=False):
        gr.Markdown("""
        ### What is BLIP?
        **BLIP (Bootstrapping Language-Image Pre-training)** is a unified vision-language framework developed by Salesforce Research.
        Unlike models limited to only image classification or text generation, BLIP bridges computer vision and natural language processing simultaneously.

        #### Key Architecture Highlights:
        1. **Vision Transformer (ViT) Backbone**: Converts the image into discrete patch embeddings ($16 \\times 16$ pixels) to capture spatial and visual semantics.
        2. **Multimodal Mixture of Encoder-Decoder (MED)**: A flexible multi-task architecture that operates in three modes:
           - *Unimodal Encoder*: Encodes text and vision independently.
           - *Image-Grounded Text Encoder*: Injects visual representations into text attention layers using cross-attention.
           - *Image-Grounded Text Decoder*: Employs causal masked self-attention and cross-attention to autoregressively decode descriptive captions word-by-word.
        3. **Dataset Bootstrapping (CapFilt)**: BLIP filters out noisy web captions using a captioner-filter mechanism, significantly improving caption quality over legacy models like Show-and-Tell.

        #### Societal Impact & Accessibility:
        - Provides immediate real-time alt-text generation for social media, news, and e-commerce.
        - Enables assistive screen-reader devices to describe physical surroundings to visually impaired users via text-to-speech.
        """)

    # Event Handlers
    submit_btn.click(
        fn=generate_caption_pipeline,
        inputs=[
            input_image,
            caption_mode,
            custom_prefix,
            num_beams,
            max_length,
            min_length,
            repetition_penalty,
            enable_tts
        ],
        outputs=[output_caption, audio_player, metrics_display, download_json, download_txt]
    )

    def reset_fields():
        return (
            None,
            "Standard (Unconditional)",
            "",
            5,
            45,
            10,
            1.2,
            True,
            "",
            None,
            "_Inference diagnostics will populate here once an image is processed._",
            None,
            None
        )

    clear_btn.click(
        fn=reset_fields,
        inputs=[],
        outputs=[
            input_image,
            caption_mode,
            custom_prefix,
            num_beams,
            max_length,
            min_length,
            repetition_penalty,
            enable_tts,
            output_caption,
            audio_player,
            metrics_display,
            download_json,
            download_txt
        ]
    )


# ---------------------------------------------------------
# Launch Server
# ---------------------------------------------------------
if __name__ == "__main__":
    local_url = "http://127.0.0.1:7860"
    print("\n" + "=" * 60)
    print("  🚀 AI Image Caption Generator is LIVE & READY!")
    print(f"  👉 Web Interface: {local_url}")
    print("  Opening browser automatically...")
    print("  (Keep this console window open while using the app)")
    print("=" * 60 + "\n")
    demo.queue().launch(**launch_kwargs)
