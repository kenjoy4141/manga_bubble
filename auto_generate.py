#!/usr/bin/env python3
"""
auto_generate.py
input/ 内の画像を Claude API で解析し、セリフ・位置・色を自動生成して
bubbles.csv を作成後、add_bubbles.py を実行します。

事前準備:
  pip install anthropic
  環境変数 ANTHROPIC_API_KEY を設定、または .env ファイルに記述:
    ANTHROPIC_API_KEY=sk-ant-...
"""

import os
import csv
import base64
import json
import subprocess
from pathlib import Path
from typing import Optional
from pydantic import BaseModel

import anthropic

INPUT_DIR    = "input"
CSV_FILE     = "bubbles.csv"
MODEL        = "claude-haiku-4-5"
CONTEXT_FILE = "context.txt"   # キャラクター設定などの前提条件ファイル

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp"}

MEDIA_TYPES = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
}

# ── 構造化出力用スキーマ ──────────────────────────────────────
class Bubble(BaseModel):
    text:  str
    pos:   str
    color: str

class BubbleList(BaseModel):
    bubbles: list[Bubble]



def load_context() -> str:
    """context.txt が存在すればその内容を返す"""
    path = Path(CONTEXT_FILE)
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def build_system_prompt(context: str) -> str:
    """前提条件を組み込んだシステムプロンプトを生成する"""
    context_section = ""
    if context:
        context_section = f"""
## 前提条件（キャラクター設定など）
{context}

"""
    return f"""あなたは漫画のセリフを考えるプロです。
送られてきた画像を見て、キャラクターのセリフを考えてください。
{context_section}
ルール:
- セリフは日本語で、漫画らしい自然な口語表現にしてください
- 「…」は使わず「・・・」を使ってください
- 「♡」「♥」などのハートマーク記号は絶対に使わないでください
- 吹き出しは最大3つまで（画像の内容に応じて1〜3個）
- 位置は画像内の空きスペースを選んでください
- 使える位置名: top-left, top-center, top-right, center-left, center, center-right, bottom-left, bottom-center, bottom-right
- 使える色名: blue, pink, purple, lavender, green, yellow, red, orange, white
- キャラクターごとに色を1つ決め、同じキャラクターの吹き出しは必ず同じ色を使ってください
  （例：女性キャラ→pink、男性キャラ→blue　のように統一する）
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
    """画像をbase64エンコードし (data, media_type) を返す"""
    media_type = MEDIA_TYPES.get(path.suffix.lower(), "image/jpeg")
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return data, media_type


def generate_bubbles(client: anthropic.Anthropic, image_path: Path, system_prompt: str) -> list[Bubble]:
    """画像を送ってセリフ・位置・色を生成する"""
    img_data, media_type = encode_image(image_path)

    response = client.messages.parse(
        model=MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {
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
                        "text": "この画像のキャラクターに合うセリフを考えてください。",
                    },
                ],
            }
        ],
        output_format=BubbleList,
    )

    return response.parsed_output.bubbles


def main():
    api_key = load_api_key()
    if not api_key:
        print("エラー: ANTHROPIC_API_KEY が見つかりません")
        print("  環境変数または .env ファイルに設定してください:")
        print("  ANTHROPIC_API_KEY=sk-ant-...")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # 前提条件の読み込み
    context = load_context()
    system_prompt = build_system_prompt(context)
    if context:
        print(f"[設定] {CONTEXT_FILE} を読み込みました")
    else:
        print(f"[設定] {CONTEXT_FILE} なし（デフォルト設定で生成）")

    images = sorted(
        p for p in Path(INPUT_DIR).iterdir()
        if p.suffix.lower() in SUPPORTED_EXT
    )

    if not images:
        print(f"[!] {INPUT_DIR}/ に対応画像が見つかりません ({', '.join(SUPPORTED_EXT)})")
        return

    rows = []
    for img_path in images:
        print(f"[解析中] {img_path.name} ...", end="", flush=True)
        try:
            bubbles = generate_bubbles(client, img_path, system_prompt)

            row = {"filename": img_path.name, "day": ""}
            for i in range(1, 4):
                if i <= len(bubbles):
                    b = bubbles[i - 1]
                    row[f"text{i}"]  = b.text
                    row[f"pos{i}"]   = b.pos
                    row[f"color{i}"] = b.color
                else:
                    row[f"text{i}"]  = ""
                    row[f"pos{i}"]   = ""
                    row[f"color{i}"] = ""

            rows.append(row)

            def safe_print(s):
                print(s.encode("utf-8", errors="replace").decode("utf-8", errors="replace"), flush=True)

            safe_print(f" {len(bubbles)}個生成")
            for b in bubbles:
                safe_print(f"   [{b.pos} / {b.color}] {b.text}")

        except Exception as e:
            print(f" エラー: {e}")

    if not rows:
        print("生成できた画像がありませんでした")
        return

    fieldnames = [
        "filename", "day",
        "text1", "pos1", "color1",
        "text2", "pos2", "color2",
        "text3", "pos3", "color3",
    ]
    with open(CSV_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[完了] {CSV_FILE} を生成しました")
    print("[実行] add_bubbles.py ...\n")
    subprocess.run(["python", "add_bubbles.py"])


if __name__ == "__main__":
    main()
