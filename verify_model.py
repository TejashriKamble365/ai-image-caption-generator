import sys
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS

print("Python version:", sys.version)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Salesforce/blip-image-captioning-base"

print(f"Loading processor and model {model_id}...")
processor = BlipProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)

img = Image.open("sample_images/park_bench.jpg").convert("RGB")
inputs = processor(images=img, return_tensors="pt").to(device)

print("Generating caption...")
out = model.generate(**inputs, max_new_tokens=30, num_beams=3)
caption = processor.decode(out[0], skip_special_tokens=True)
print("Result Caption:", caption)

print("Testing TTS generation...")
tts = gTTS(text=caption, lang="en")
tts.save("sample_images/test_audio.mp3")
print("TTS Audio generated successfully!")
