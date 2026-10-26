---
title: AI Image Caption Generator Using BLIP
emoji: 🖼️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
short_description: Generate natural language captions and audio descriptions using BLIP
---

# AI Image Caption Generator Using BLIP & Generative Vision-Language AI

[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)
[![Transformers](https://img.shields.io/badge/Transformers-Hugging%20Face-yellow.svg)](https://huggingface.co/docs/transformers/index)
[![Model](https://img.shields.io/badge/Model-Salesforce%2Fblip--image--captioning--base-blue.svg)](https://huggingface.co/Salesforce/blip-image-captioning-base)
[![Accessibility](https://img.shields.io/badge/Accessibility-Text--to--Speech-green.svg)](#accessibility--assistive-tech)

An end-to-end vision-language web application built for the **Generative AI In-Semester Evaluation (ISE)**. This system allows users to upload any image (JPG, PNG, WebP) and automatically generates accurate, natural-language captions using Salesforce's pre-trained **BLIP (Bootstrapping Language-Image Pre-training)** model. 

It additionally incorporates **Text-to-Speech (TTS)** narration to serve as an assistive tool for visually impaired users.

---

## 📌 Problem & Motivation

- **Accessibility Barrier**: Over 2.2 billion people globally experience visual impairment. Digital media often lacks descriptive alternative text (`alt-text`), locking visually impaired users out of online content.
- **Manual Overhead**: Content creators, e-commerce retailers, and news outlets spend thousands of hours manually tagging images.
- **Solution**: This system automates visual understanding by analyzing semantic objects, actions, and scenes, producing both high-quality textual descriptions and spoken audio narration in seconds.

---

## 🚀 Key Features

1. **Autonomous & Guided Captioning**:
   - **Standard Mode**: Unconditional vision-language captioning.
   - **Prefix Guidance**: Steers the model's focus (e.g., "a detailed photograph of", "an illustration of").
2. **Integrated Accessibility (Audio Narration)**:
   - Synthesizes the generated caption into spoken MP3 audio on the fly using `gTTS`.
3. **Hyperparameter Tuning for Viva Demonstrations**:
   - Beam search size adjustment (`num_beams` from 1 to 8).
   - Repetition penalty controls (`1.0` to `2.0`).
   - Sequence length limits (`min_length`, `max_length`).
4. **Live Image & Model Telemetry**:
   - Real-time inference time tracker, image dimensions, color mode, and device detection (`CUDA` vs `CPU`).
5. **Report Exports**:
   - One-click copy to clipboard.
   - Download as a structured JSON report or plaintext file.
6. **Built-in Evaluator Presets**:
   - Pre-loaded sample images so evaluators can test the system without downloading external images.

---

## 🛠️ Technology Stack

| Layer | Component | Description |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core application runtime |
| **AI Model** | Salesforce BLIP Base | Pre-trained vision-language transformer for caption generation |
| **Framework** | Hugging Face Transformers & PyTorch | Loads weights, tokenizer, and handles tensor operations |
| **User Interface** | Gradio (Blocks API) | Modern, responsive web interface with custom CSS |
| **Audio Synthesis** | gTTS | Google Text-to-Speech engine for voice narration |
| **Deployment Target** | Hugging Face Spaces | Cloud hosting platform for public access |

---

## 📂 Project Structure

```text
ai-image-captioner/
├── app.py                # Main Gradio application with BLIP inference & TTS
├── requirements.txt      # Python dependencies for local execution & HF Spaces
├── README.md             # HF Space metadata & complete project documentation
├── create_samples.py     # Script to generate sample demo images
├── run.bat               # Windows batch script for one-click local launch
├── run.ps1               # PowerShell launcher
├── .gitignore            # Clean git repository rules
└── sample_images/        # Pre-loaded images for viva demonstration
    ├── living_room.jpg
    ├── park_bench.jpg
    └── sunset_ocean.jpg
```

---

## 💻 Local Setup & Execution

### Prerequisites
- Python 3.10 to 3.13 installed on your system.
- Git (optional, for version control).

### Step 1: Clone or Navigate to the Directory
```bash
cd C:\Users\TEJASHRI\.gemini\antigravity\scratch\ai-image-captioner
```

### Step 2: Set Up Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Generate Demo Sample Images (If not already present)
```bash
python create_samples.py
```

### Step 5: Run the Web Application
```bash
python app.py
```
After starting, open your browser and go to:
👉 **`http://127.0.0.1:7860`**

---

## 🌐 Deploying to Hugging Face Spaces (Public URL for ISE)

Follow these simple steps to deploy your live model publicly so that your examiner can view the working project:

### Option A: Via Hugging Face Web Interface (Easiest - 3 Minutes)
1. Go to [Hugging Face](https://huggingface.co/) and log in (or create a free account).
2. Click on your profile icon in the top right → **New Space**.
3. Name your space (e.g. `blip-image-caption-generator`).
4. Select **Gradio** as the Space SDK.
5. Choose **Public** visibility and the free **CPU basic** hardware tier (2 vCPU, 16 GB RAM is plenty).
6. Click **Create Space**.
7. In the "Files" tab of your newly created Space:
   - Upload `app.py`
   - Upload `requirements.txt`
   - Upload `README.md`
   - Upload the `sample_images/` folder
8. Hugging Face will automatically detect `requirements.txt`, install PyTorch and Transformers, and build the live container!
9. Within 1–2 minutes, your space will say **Running**, and your public URL (e.g. `https://huggingface.co/spaces/<your-username>/blip-image-caption-generator`) is live and accessible worldwide.

### Option B: Via Git CLI
```bash
git init
git remote add origin https://huggingface.co/spaces/<YOUR_USERNAME>/<SPACE_NAME>
git add .
git commit -m "Deploy BLIP Image Caption Generator"
git push -u origin main
```

---

## 🧠 Theory & ISE Viva Questions Guide

### 1. What does BLIP stand for?
**Bootstrapping Language-Image Pre-training**. It is a unified vision-language framework developed by Salesforce Research.

### 2. How does BLIP differ from traditional CNN+RNN models (like Show-and-Tell)?
- Traditional models used a convolutional network (e.g. ResNet) to extract fixed feature vectors and passed them into an LSTM/RNN. This led to information loss and bottlenecking.
- BLIP utilizes a **Vision Transformer (ViT)** to encode image patches directly and a **Multimodal Mixture of Encoder-Decoder (MED)** to attend across modalities using cross-attention mechanisms, generating far more accurate and nuanced language.

### 3. What is the role of Beam Search?
Beam search maintains the top $k$ most probable partial sequences (beams) at each step of text generation rather than greedily picking the single highest-probability token. This prevents the model from choosing a locally optimal word that leads to poor global sentences.

### 4. What is dataset bootstrapping (CapFilt)?
Web-scraped image-text pairs often contain noise (e.g., ads, irrelevant captions). BLIP introduces **Captioner-Filter (CapFilt)**: a captioner synthesizes synthetic captions for web images, and a filter eliminates noisy or mismatched pairs, resulting in a cleaner dataset for training.

---

## 📄 License
This project is licensed under the MIT License - open for educational and research demonstrations.
