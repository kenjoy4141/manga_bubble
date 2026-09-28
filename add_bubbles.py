#!/usr/bin/env python3
"""
Manga speech bubble adder v2
縦書き・カラー・角丸吹き出し対応

CSV形式:
  filename,text1,pos1,color1,text2,pos2,color2,text3,pos3,color3

色名: blue, pink, purple, lavender, green, yellow, red, orange, white
または16進数: #4488ff

位置名:
  top-left, top-center, top-right
  center-left, center, center-right
  bottom-left, bottom-center, bottom-right
"""

import csv
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# アニメ顔検出カスケード
ANIME_CASCADE_PATH = "lbpcascade_animeface.xml"
_face_cascade = None

def get_face_cascade():
    global _face_cascade
    if _face_cascade is None and os.path.exists(ANIME_CASCADE_PATH):
        _face_cascade = cv2.CascadeClassifier(ANIME_CASCADE_PATH)
    return _face_cascade

def detect_faces(img_path):
    """画像からアニメ顔を検出して矩形リストを返す [(x,y,w,h), ...]"""
    cascade = get_face_cascade()
    if cascade is None:
        return []
    img_cv = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_cv is None:
        return []
    faces = cascade.detectMultiScale(
        img_cv,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
    )
    return faces.tolist() if len(faces) > 0 else []

def overlaps_face(bx1, by1, bx2, by2, faces, margin=20):
    """吹き出し矩形が顔矩形と重なるか判定"""
    for (fx, fy, fw, fh) in faces:
        fx1, fy1 = fx - margin, fy - margin
        fx2, fy2 = fx + fw + margin, fy + fh + margin
        if bx1 < fx2 and bx2 > fx1 and by1 < fy2 and by2 > fy1:
            return True
    return False


def overlaps_bubble(bx1, by1, bx2, by2, placed, margin=5):
    """吹き出し矩形が配置済み吹き出しと重なるか判定"""
    for (px1, py1, px2, py2) in placed:
        if bx1 < px2 + margin and bx2 > px1 - margin \
                and by1 < py2 + margin and by2 > py1 - margin:
            return True
    return False

INPUT_DIR  = "input"
OUTPUT_DIR = "output"
CSV_FILE   = "bubbles.csv"
BUBBLE_ALPHA = 170  # 吹き出し透明度 0(透明)〜255(不透明)

# カスタムフォントパス（fonts/ フォルダに置いた .ttf/.otf を指定）
# 例: CUSTOM_FONT = "fonts/logotype-gothic.otf"
#     CUSTOM_FONT = "fonts/anzumoji.ttf"
CUSTOM_FONT = "font/あんずもじ等幅/APJapanesefontT.ttf"  # 空文字のままだとシステムフォントを使用

# 位置名 -> (中心X比, 中心Y比, しっぽ方向)
POSITION_MAP = {
    "top-left":      (0.13, 0.13, "down-right"),
    "top-center":    (0.50, 0.09, "down"),
    "top-right":     (0.87, 0.13, "down-left"),
    "center-left":   (0.08, 0.50, "right"),
    "center":        (0.50, 0.50, "down"),
    "center-right":  (0.92, 0.50, "left"),
    "bottom-left":   (0.13, 0.87, "up-right"),
    "bottom-center": (0.50, 0.91, "up"),
    "bottom-right":  (0.87, 0.87, "up-left"),
}

# 色名 -> RGB
COLOR_MAP = {
    "blue":     (80,  170, 255),
    "pink":     (255, 150, 200),
    "purple":   (185, 130, 255),
    "lavender": (205, 175, 255),
    "green":    (120, 215, 130),
    "yellow":   (255, 228,  90),
    "red":      (255, 110, 110),
    "orange":   (255, 175,  70),
    "white":    (255, 255, 255),
    "gray":     (195, 195, 195),
}


def parse_color(s, alpha=BUBBLE_ALPHA):
    s = s.strip().lower() if s else "white"
    if s.startswith("#") and len(s) == 7:
        r = int(s[1:3], 16)
        g = int(s[3:5], 16)
        b = int(s[5:7], 16)
        return (r, g, b, alpha)
    rgb = COLOR_MAP.get(s, (255, 255, 255))
    return (*rgb, alpha)


def find_font(size):
    # カスタムフォントが指定されていれば最優先
    if CUSTOM_FONT and os.path.exists(CUSTOM_FONT):
        try:
            return ImageFont.truetype(CUSTOM_FONT, size)
        except Exception as e:
            print(f"警告: カスタムフォント読み込み失敗 ({e})、システムフォントで代替")

    # fonts/ フォルダ内を自動検索
    if os.path.isdir("fonts"):
        for fname in os.listdir("fonts"):
            if fname.lower().endswith((".ttf", ".otf", ".ttc")):
                path = os.path.join("fonts", fname)
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue

    # システムフォント候補
    candidates = [
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/YuGothM.ttc",
        "C:/Windows/Fonts/yugothm.ttf",
        "C:/Windows/Fonts/msgothic.ttc",
        "C:/Windows/Fonts/YuGoth.ttc",
        "C:/Windows/Fonts/meiryob.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    print("警告: 日本語フォント未検出、デフォルト使用")
    return ImageFont.load_default()


def char_size(font):
    """全角文字の代表サイズ (w, h)"""
    bbox = font.getbbox("あ")
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


# 行頭禁則文字（これらで列を始めてはいけない）
KINSOKU_START = set(
    "。、！？…‥・）｝〕〉》】〙〗〟\u2019\u201dヽヾゞ々〻"
    "ーっッゃャゅュょョぁァぃィぅゥぇェぉォ"
)
# 行末禁則文字（これらで列を終えてはいけない）
KINSOKU_END = set("（｛〔〈《【〘〖〝\u2018\u201c")

# 改行優先度：最高（スペースの位置で区切る・スペース自体は除去）
BREAK_SPACE = {" ", "\u3000"}  # 半角・全角スペース
# 改行優先度：高（句点・感嘆符の直後）
BREAK_HIGH = set("。！？")
# 改行優先度：中（読点・中点の直後）
BREAK_MID  = set("、・")
# 改行優先度：低（助詞の直後）
BREAK_LOW  = set("はがをにでともやのへ")

COL_MIN = 13
COL_TARGET = 16
COL_MAX = 20


def layout_vertical(text, font, max_col_height):
    """
    縦書きレイアウト（禁則処理 + 自然な改行位置）
    1列 COL_MIN〜COL_MAX 字、スペース > 句読点 > 助詞の順で改行する
    スペースは改行点として使い、出力には含めない
    Returns: columns (list of list of str), char_w, char_h
    """
    cw, ch = char_size(font)
    height_limit = max(COL_MIN, max_col_height // ch)
    col_max = min(COL_MAX, height_limit)
    col_min = min(COL_MIN, col_max)

    chars = list(text)
    n = len(chars)
    columns = []
    i = 0

    while i < n:
        remaining = n - i
        if remaining <= col_max:
            col = [c for c in chars[i:] if c not in BREAK_SPACE]
            if col:
                columns.append(col)
            break

        # 最優先：スペースの位置（MIN〜MAX の範囲で左から探す）
        found = False
        for k in range(col_min - 1, min(col_max, remaining)):
            if chars[i + k] in BREAK_SPACE:
                col = [c for c in chars[i:i + k] if c not in BREAK_SPACE]
                if col:
                    columns.append(col)
                i += k + 1  # スペース自体を読み飛ばす
                found = True
                break

        if found:
            continue

        # 優先度順に改行位置を探す
        break_at = min(COL_TARGET, remaining)

        # 高優先：句点・感嘆符の直後
        for k in range(col_min - 1, min(col_max, remaining)):
            if chars[i + k] in BREAK_HIGH:
                break_at = k + 1
                found = True
                break

        if not found:
            # 中優先：読点の直後（右端から探す）
            for k in range(min(col_max, remaining) - 1, col_min - 2, -1):
                if chars[i + k] in BREAK_MID:
                    break_at = k + 1
                    found = True
                    break

        if not found:
            # 低優先：助詞の直後（右端から探す）
            for k in range(min(col_max, remaining) - 1, col_min - 2, -1):
                if chars[i + k] in BREAK_LOW:
                    break_at = k + 1
                    found = True
                    break

        # 禁則：次の文字が行頭禁則なら1文字ぶら下げ
        end = i + break_at
        if end < n and chars[end] in KINSOKU_START:
            break_at += 1
        elif break_at > 1 and chars[i + break_at - 1] in KINSOKU_END:
            break_at -= 1

        columns.append(chars[i:i + break_at])
        i += break_at

    return columns, cw, ch


# 句読点：右上にオフセット
PUNCT_SHIFT = {"、", "。", "，", "．"}
# 縦書きで90度回転が必要な文字
ROTATE_CHARS = {"ー", "〜", "～", "―", "─", "（", "）", "(", ")"}
# 鉤括弧：位置調整（「は左、」は右）
BRACKET_LEFT  = {"「", "『"}   # 左にずらす
BRACKET_RIGHT = {"」", "』"}   # 右にずらす


def draw_char_vertical(overlay, draw, char, x, y, font, cw, ch):
    """1文字を縦書き用に描画（句読点オフセット・ー回転に対応）"""
    stroke_kw = dict(stroke_width=2, stroke_fill="white")

    if char in PUNCT_SHIFT:
        # 右上にシフト
        draw.text((x + cw * 0.45, y - ch * 0.25), char,
                  fill="black", font=font, **stroke_kw)

    elif char in BRACKET_LEFT:
        # 左にずらす
        draw.text((x - cw * 0.5, y), char, fill="black", font=font, **stroke_kw)

    elif char in BRACKET_RIGHT:
        # 右にずらす
        draw.text((x + cw * 0.5, y), char, fill="black", font=font, **stroke_kw)

    elif char in ROTATE_CHARS:
        # 正方形の一時画像に中央揃えで描いて-90度回転、セル中央にペースト
        size = max(cw, ch) + 8
        tmp = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        tx = (size - cw) // 2
        ty = (size - ch) // 2
        ImageDraw.Draw(tmp).text((tx, ty), char,
                                  fill="black", font=font, **stroke_kw)
        tmp = tmp.rotate(-90, expand=False)
        paste_x = x - (size - cw) // 2 + cw // 8
        paste_y = y - (size - ch) // 2 + ch // 8
        overlay.paste(tmp, (paste_x, paste_y), tmp)

    else:
        draw.text((x, y), char, fill="black", font=font, **stroke_kw)


def draw_bubble(overlay, draw, img_w, img_h, text, pos_name, color_str, font,
                day_label_bottom=0, placed_bubbles=None):
    # 「…」「...」を縦書き向けの「・・・」に自動変換
    text = text.replace("…", "・・・").replace("...", "・・・")

    if placed_bubbles is None:
        placed_bubbles = []

    pos_name = pos_name.strip().lower()
    if pos_name not in POSITION_MAP:
        print(f"  警告: 不明な位置 '{pos_name}' — スキップ")
        return None

    cx_r, cy_r, tail_dir = POSITION_MAP[pos_name]
    cx = int(cx_r * img_w)
    cy = int(cy_r * img_h)

    fill = parse_color(color_str)

    padding  = max(22, img_w // 26)
    col_gap  = 5
    max_col_h = int(img_h * 0.42)

    columns, cw, ch = layout_vertical(text, font, max_col_h)
    n_cols     = len(columns)
    max_chars  = max(len(c) for c in columns)

    text_h = max_chars * ch
    text_w = n_cols * cw + max(0, n_cols - 1) * col_gap

    bw = text_w + padding * 2
    bh = text_h + padding * 2

    # 画像内クランプ用ヘルパー
    margin = 5
    def clamp(nx1, ny1):
        nx2, ny2 = nx1 + bw, ny1 + bh
        if nx1 < margin:            nx1 = margin;            nx2 = nx1 + bw
        if nx2 > img_w - margin:    nx2 = img_w - margin;   nx1 = nx2 - bw
        if ny1 < margin:            ny1 = margin;            ny2 = ny1 + bh
        if ny2 > img_h - margin:    ny2 = img_h - margin;   ny1 = ny2 - bh
        return nx1, ny1, nx2, ny2

    bx1, by1, bx2, by2 = clamp(cx - bw // 2, cy - bh // 2)

    # top-left の吹き出しが N日目ラベルと被らないよう上端を押し下げる
    if pos_name == "top-left" and day_label_bottom > 0:
        if by1 < day_label_bottom + 10:
            by1 = day_label_bottom + 10
            by2 = by1 + bh

    # 吹き出し同士の重複回避
    if placed_bubbles and overlaps_bubble(bx1, by1, bx2, by2, placed_bubbles):
        gap = 10
        candidates = []
        # 配置済み各吹き出しの隣にぴったり移動する候補（x/y それぞれ維持）
        for (px1, py1, px2, py2) in placed_bubbles:
            candidates += [
                clamp(bx1, py1 - bh - gap),      # 配置済みの上（x維持）
                clamp(bx1, py2 + gap),            # 配置済みの下（x維持）
                clamp(px1 - bw - gap, by1),       # 配置済みの左（y維持）
                clamp(px2 + gap, by1),            # 配置済みの右（y維持）
            ]
        # 上記で解決しない場合の補助候補（元位置から小→大シフト）
        for step in [bh // 2, bh, bh + bw // 2]:
            candidates += [
                clamp(bx1, by1 - step),
                clamp(bx1, by1 + step),
                clamp(bx1 - step, by1),
                clamp(bx1 + step, by1),
            ]
        for c in candidates:
            if not overlaps_bubble(*c, placed_bubbles):
                bx1, by1, bx2, by2 = c
                break
    cx = (bx1 + bx2) // 2
    cy = (by1 + by2) // 2

    # しっぽパラメータ（細長い三角形）
    tail_len  = max(28, min(55, img_h // 14))
    half_base = max(6, img_w // 80)

    tips = {
        "down":       (cx,                  by2 + tail_len),
        "up":         (cx,                  by1 - tail_len),
        "right":      (bx2 + tail_len,      cy),
        "left":       (bx1 - tail_len,      cy),
        "down-right": (bx2 + tail_len // 2, by2 + tail_len),
        "down-left":  (bx1 - tail_len // 2, by2 + tail_len),
        "up-right":   (bx2 + tail_len // 2, by1 - tail_len),
        "up-left":    (bx1 - tail_len // 2, by1 - tail_len),
    }
    bases = {
        "down":       [(cx - half_base, by2),  (cx + half_base, by2)],
        "up":         [(cx - half_base, by1),  (cx + half_base, by1)],
        "right":      [(bx2, cy - half_base),  (bx2, cy + half_base)],
        "left":       [(bx1, cy - half_base),  (bx1, cy + half_base)],
        "down-right": [(cx - half_base, by2),  (bx2, cy + half_base)],
        "down-left":  [(cx + half_base, by2),  (bx1, cy + half_base)],
        "up-right":   [(cx - half_base, by1),  (bx2, cy - half_base)],
        "up-left":    [(cx + half_base, by1),  (bx1, cy - half_base)],
    }

    # ① 角丸長方形（しっぽなし）
    radius = min(bw, bh) // 5
    draw.rounded_rectangle([bx1, by1, bx2, by2], radius=radius, fill=fill)

    # ③ 縦書きテキスト（右カラムから左へ）
    text_right = cx + text_w // 2
    text_top   = cy - text_h // 2

    for col_idx, col_chars in enumerate(columns):
        col_x = text_right - col_idx * (cw + col_gap) - cw
        cur_y = text_top
        for char in col_chars:
            draw_char_vertical(overlay, draw, char, col_x, cur_y, font, cw, ch)
            cur_y += ch

    return (bx1, by1, bx2, by2)


def draw_day_label(overlay, img_w, img_h, day_str, font):
    """左上に「N日目」ラベルを白い横長方形で描画する"""
    label = f"{day_str}日目"
    draw  = ImageDraw.Draw(overlay)

    # 横書き用フォントサイズ（吹き出しフォントより少し大きめ）
    font_size = max(20, int(img_w * 0.045))
    label_font = find_font(font_size)

    bbox = label_font.getbbox(label)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    pad_x = int(font_size * 0.6)
    pad_y = int(font_size * 0.4)
    margin = 10

    rx1 = margin
    ry1 = margin
    rx2 = rx1 + tw + pad_x * 2
    ry2 = ry1 + th + pad_y * 2

    radius = (ry2 - ry1) // 4
    # 白・不透明の長方形
    draw.rounded_rectangle([rx1, ry1, rx2, ry2], radius=radius,
                            fill=(255, 255, 255, 235), outline=(0, 0, 0, 180), width=2)

    tx = rx1 + pad_x - bbox[0]
    ty = ry1 + pad_y - bbox[1]
    draw.text((tx, ty), label, fill="black", font=label_font,
              stroke_width=0)
    return ry2  # 呼び出し元が吹き出し位置調整に使う


def process():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    font_cache = {}

    def get_font(img_w):
        # 画像幅の約4%をフォントサイズに
        size = max(18, int(img_w * 0.040))
        if size not in font_cache:
            font_cache[size] = find_font(size)
        return font_cache[size]

    with open(CSV_FILE, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("filename", "").strip()
            if not filename:
                continue

            src = os.path.join(INPUT_DIR, filename)
            if not os.path.exists(src):
                print(f"[スキップ] ファイルが見つかりません: {src}")
                continue

            print(f"[処理中] {filename} ...", end="", flush=True)

            # アニメ顔検出（顔回避用）
            faces = detect_faces(src)

            img  = Image.open(src).convert("RGBA")
            font = get_font(img.width)

            # 透明オーバーレイに描いてから合成することで半透明が正しく機能する
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw    = ImageDraw.Draw(overlay)

            # N日目ラベル
            day = row.get("day", "").strip()
            day_label_bottom = 0
            if day:
                day_label_bottom = draw_day_label(overlay, img.width, img.height, day, font)

            placed_bubbles = []
            for i in range(1, 4):
                text  = row.get(f"text{i}",  "").strip()
                pos   = row.get(f"pos{i}",   "").strip()
                color = row.get(f"color{i}", "white").strip() or "white"
                if text and pos:
                    rect = draw_bubble(overlay, draw, img.width, img.height, text, pos, color, font,
                                       day_label_bottom=day_label_bottom,
                                       placed_bubbles=placed_bubbles)
                    if rect:
                        placed_bubbles.append(rect)

            img = Image.alpha_composite(img, overlay)

            ext = os.path.splitext(filename)[1].lower()
            dst = os.path.join(OUTPUT_DIR, filename)
            if ext in (".jpg", ".jpeg"):
                img.convert("RGB").save(dst, quality=95)
            else:
                img.save(dst)
            print(f" 完了 -> {dst}")

    print("\n全て処理しました。")


if __name__ == "__main__":
    process()
