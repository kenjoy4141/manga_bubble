#!/usr/bin/env python3
"""
png_to_jpg.py
指定ディレクトリ内の PNG ファイルを JPG に変換・圧縮します。

使い方:
  python png_to_jpg.py                  # output/ フォルダを対象
  python png_to_jpg.py input            # input/ フォルダを対象
  python png_to_jpg.py output 85        # 品質85で変換
"""

import os
import sys
from pathlib import Path
from PIL import Image

DEFAULT_DIR     = "output"
DEFAULT_QUALITY = 90   # JPG 品質 (1〜95)


def convert(target_dir: str, quality: int):
    files = sorted(Path(target_dir).glob("*.png"))
    if not files:
        print(f"[!] {target_dir}/ に PNG ファイルが見つかりません")
        return

    print(f"[対象] {target_dir}/ - {len(files)} ファイル (quality={quality})")

    for src in files:
        dst = src.with_suffix(".jpg")
        try:
            img = Image.open(src).convert("RGB")
            img.save(dst, "JPEG", quality=quality)
            src_kb = src.stat().st_size / 1024
            dst_kb = dst.stat().st_size / 1024
            ratio  = dst_kb / src_kb * 100
            print(f"  {src.name} -> {dst.name}  ({src_kb:.0f}KB -> {dst_kb:.0f}KB, {ratio:.0f}%)")
        except Exception as e:
            print(f"  エラー: {src.name} — {e}")

    print("\n変換完了")


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    quality    = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_QUALITY

    if not os.path.isdir(target_dir):
        print(f"エラー: ディレクトリが見つかりません: {target_dir}")
        sys.exit(1)

    convert(target_dir, quality)
