#!/usr/bin/env python3
"""
mosaic.py
input/ 内の画像のセンシティブな箇所を Claude API で自動検出し、
モザイクをかけて output_mosaic/ に保存します。

モザイクサイズは画像ごとに自動計算:
  block = max(4, 長辺 // 100)
  ※長辺 400px 未満は最小の 4px 固定

使い方:
  python mosaic.py              # input/ を処理
  python mosaic.py output       # output/ を処理（吹き出し付き画像に適用する場合）
"""

import os
import sys
import base64
from pathlib import Path
from PIL import Image
from pydantic import BaseModel
import anthropic

DEFAULT_INPUT_DIR  = "input"
OUTPUT_SUFFIX      = "_mosaic"   # 出力フォルダ名: 入力フォルダ名 + _mosaic
MODEL              = "claude-opus-4-6"
MOSAIC_MIN_BLOCK   = 4           # モザイク最小ブロックサイズ（px）

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp"}
MEDIA_TYPES = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
}


# ── 構造化出力スキーマ ─────────────────────────────────────────
class Region(BaseModel):
    x: float  # 左端（画像幅に対する比率 0.0〜1.0）
    y: float  # 上端（画像高さに対する比率 0.0〜1.0）
    w: float  # 幅  （画像幅に対する比率 0.0〜1.0）
    h: float  # 高さ（画像高さに対する比率 0.0〜1.0）

class RegionList(BaseModel):
    regions: list[Region]


SYSTEM_PROMPT = """あなたは画像内のセンシティブな箇所を検出するシステムです。
送られた画像を解析し、モザイク処理が必要な箇所（男性器・女性器）の
位置を画像全体に対する比率（0.0〜1.0）で返してください。

注意:
- 何も見つからなければ regions を空リストで返してください
- 検出対象は男性器・女性器のみです（乳首・胸は対象外）
- 衣服で隠れている場合は検出不要です（露出・描写されている箇所のみ）
- 各領域はその部位ぴったりのサイズで指定してください。余白・余裕は不要です
- 周囲の体の部位（太もも・お腹・手など）を含めないでください
"""


def load_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


def encode_image(path: Path) -> tuple[str, str]:
    media_type = MEDIA_TYPES.get(path.suffix.lower(), "image/jpeg")
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return data, media_type


def detect_regions(client: anthropic.Anthropic, image_path: Path) -> list[Region]:
    """Claude APIで画像のセンシティブ領域を検出する"""
    img_data, media_type = encode_image(image_path)

    response = client.messages.parse(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": img_data,
                    },
                },
                {
                    "type": "text",
                    "text": "この画像のモザイク処理が必要な箇所を検出してください。",
                },
            ],
        }],
        output_format=RegionList,
    )

    return response.parsed_output.regions


def calc_block_size(img_w: int, img_h: int) -> int:
    """
    画像サイズに応じたモザイクブロックサイズを返す。
    - 長辺 400px 未満: 最小 4px
    - 長辺 400px 以上: 長辺 // 100  (例: 800px -> 8px, 1200px -> 12px)
    """
    longest = max(img_w, img_h)
    return max(MOSAIC_MIN_BLOCK, longest // 100)


def apply_mosaic(img: Image.Image, x: int, y: int, w: int, h: int, block: int) -> Image.Image:
    """指定領域にモザイク（ピクセル化）をかける"""
    if w <= 0 or h <= 0:
        return img
    # 縮小→拡大でピクセル化
    region = img.crop((x, y, x + w, y + h))
    small  = region.resize((max(1, w // block), max(1, h // block)), Image.NEAREST)
    mosaic = small.resize((w, h), Image.NEAREST)
    result = img.copy()
    result.paste(mosaic, (x, y))
    return result


def process(input_dir: str):
    api_key = load_api_key()
    if not api_key:
        print("エラー: ANTHROPIC_API_KEY が見つかりません")
        return

    client     = anthropic.Anthropic(api_key=api_key)
    output_dir = input_dir.rstrip("/\\") + OUTPUT_SUFFIX
    os.makedirs(output_dir, exist_ok=True)

    images = sorted(
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in SUPPORTED_EXT
    )

    if not images:
        print(f"[!] {input_dir}/ に対応画像が見つかりません")
        return

    print(f"[設定] 入力: {input_dir}/  出力: {output_dir}/")
    print(f"[対象] {len(images)} ファイル\n")

    for img_path in images:
        print(f"[検出中] {img_path.name} ...", end="", flush=True)
        try:
            regions = detect_regions(client, img_path)

            img     = Image.open(img_path).convert("RGB")
            iw, ih  = img.size

            # 画像サイズに応じたブロックサイズを計算
            block = calc_block_size(iw, ih)

            if not regions:
                print(" 検出なし -> そのままコピー")
            else:
                print(f" {len(regions)}箇所検出 (block={block}px, 長辺={max(iw,ih)}px)")
                for r in regions:
                    # 比率をピクセル座標に変換（画像範囲内にクランプ）
                    px = max(0, int(r.x * iw))
                    py = max(0, int(r.y * ih))
                    pw = min(iw - px, max(1, int(r.w * iw)))
                    ph = min(ih - py, max(1, int(r.h * ih)))
                    print(f"   x={px} y={py} w={pw} h={ph}")
                    img = apply_mosaic(img, px, py, pw, ph, block)

            dst = Path(output_dir) / img_path.stem
            ext = img_path.suffix.lower()
            if ext in (".jpg", ".jpeg"):
                img.save(str(dst) + ".jpg", quality=95)
            else:
                img.save(str(dst) + ".png")
            print(f"   -> {output_dir}/{img_path.name}")

        except Exception as e:
            print(f" エラー: {e}")

    print("\n全て処理しました。")


if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_DIR

    if not os.path.isdir(input_dir):
        print(f"エラー: ディレクトリが見つかりません: {input_dir}")
        sys.exit(1)

    process(input_dir)
