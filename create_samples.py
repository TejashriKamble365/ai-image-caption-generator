import os
from PIL import Image, ImageDraw

os.makedirs("sample_images", exist_ok=True)

# 1. Park with trees and bench
img1 = Image.new("RGB", (640, 480), color=(135, 206, 235))
d1 = ImageDraw.Draw(img1)
# Sun
d1.ellipse((500, 30, 580, 110), fill=(255, 223, 0))
# Green grass
d1.rectangle([(0, 300), (640, 480)], fill=(34, 139, 34))
# Tree trunk and foliage
d1.rectangle([(120, 240), (160, 360)], fill=(101, 67, 33))
d1.ellipse([(60, 120), (220, 270)], fill=(0, 100, 0))
# Wooden bench
d1.rectangle([(320, 330), (460, 350)], fill=(139, 69, 19))
d1.rectangle([(340, 350), (350, 390)], fill=(50, 50, 50))
d1.rectangle([(430, 350), (440, 390)], fill=(50, 50, 50))
img1.save("sample_images/park_bench.jpg", quality=95)

# 2. Sunset over ocean
img2 = Image.new("RGB", (640, 480))
d2 = ImageDraw.Draw(img2)
for y in range(480):
    if y < 280:
        # Sky gradient from orange to purple
        r = int(255 - (y / 280) * 40)
        g = int(120 - (y / 280) * 70)
        b = int(40 + (y / 280) * 80)
        d2.line([(0, y), (640, y)], fill=(r, g, b))
    else:
        # Ocean deep blue
        ratio = (y - 280) / 200
        r = int(20 * (1 - ratio))
        g = int(40 + 20 * ratio)
        b = int(100 + 40 * ratio)
        d2.line([(0, y), (640, y)], fill=(r, g, b))
# Setting sun
d2.ellipse([(280, 220), (360, 300)], fill=(255, 200, 50))
# Sun reflection
d2.ellipse([(290, 290), (350, 320)], fill=(255, 180, 80))
img2.save("sample_images/sunset_ocean.jpg", quality=95)

# 3. Modern living room / interior sketch
img3 = Image.new("RGB", (640, 480), color=(240, 240, 242))
d3 = ImageDraw.Draw(img3)
# Wooden floor
d3.rectangle([(0, 320), (640, 480)], fill=(210, 180, 140))
# Sofa
d3.rectangle([(160, 260), (480, 360)], fill=(70, 80, 95))
d3.rectangle([(140, 280), (170, 350)], fill=(60, 70, 85))
d3.rectangle([(470, 280), (500, 350)], fill=(60, 70, 85))
# Cushions
d3.rectangle([(200, 280), (270, 330)], fill=(220, 160, 60))
d3.rectangle([(370, 280), (440, 330)], fill=(220, 160, 60))
# Wall art painting frame
d3.rectangle([(240, 60), (400, 180)], fill=(255, 255, 255), outline=(50, 50, 50), width=4)
d3.ellipse([(270, 80), (330, 140)], fill=(230, 100, 80))
img3.save("sample_images/living_room.jpg", quality=95)

print("Sample images generated successfully in sample_images/")
