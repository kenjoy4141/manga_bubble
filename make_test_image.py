"""テスト用サンプル画像を生成する"""
from PIL import Image, ImageDraw, ImageFont
import os

os.makedirs("input", exist_ok=True)

img = Image.new("RGB", (800, 600), color=(200, 220, 255))
draw = ImageDraw.Draw(img)

# 背景：空っぽい色＋簡単なキャラクター風シルエット
# 地面
draw.rectangle([0, 450, 800, 600], fill=(100, 180, 80))

# 太陽
draw.ellipse([650, 40, 750, 140], fill=(255, 220, 50), outline=(255, 180, 0), width=3)

# 雲
for ox, oy in [(100, 60), (300, 30), (500, 80)]:
    draw.ellipse([ox, oy, ox+100, oy+50], fill="white")
    draw.ellipse([ox+25, oy-20, ox+75, oy+40], fill="white")

# キャラA（左）: 丸頭＋胴体
draw.ellipse([100, 280, 180, 360], fill=(255, 220, 185), outline="black", width=2)
draw.rectangle([115, 360, 165, 450], fill=(80, 100, 200), outline="black", width=2)

# キャラB（右）: 丸頭＋胴体
draw.ellipse([580, 280, 660, 360], fill=(255, 220, 185), outline="black", width=2)
draw.rectangle([595, 360, 645, 450], fill=(200, 80, 80), outline="black", width=2)

img.save("input/test.png")
print("input/test.png を作成しました")
