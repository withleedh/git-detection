"""Test Korean font rendering"""
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

# Create test image
img = np.ones((200, 600, 3), dtype=np.uint8) * 255
img_pil = Image.fromarray(img)
draw = ImageDraw.Draw(img_pil)

# Try different fonts
fonts_to_test = [
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]

y_pos = 20
for font_path in fonts_to_test:
    try:
        font = ImageFont.truetype(font_path, 28)
        draw.text((10, y_pos), f"스크래치: 77.76%", font=font, fill=(0, 0, 0))
        print(f"✓ {font_path.split('/')[-1]} works!")
        y_pos += 40
    except Exception as e:
        print(f"✗ {font_path}: {e}")

# Try default
try:
    font = ImageFont.load_default()
    draw.text((10, y_pos), f"스크래치 (default): 77.76%", font=font, fill=(0, 0, 0))
    print(f"✓ Default font")
except Exception as e:
    print(f"✗ Default: {e}")

# Save
img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
cv2.imwrite("/Users/dongho/dev/git-detection/font_test.jpg", img_cv)
print("\n✓ Saved to font_test.jpg")
