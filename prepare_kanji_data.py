import argparse
import os
import pickle
import random

import torch
from PIL import Image, ImageDraw, ImageFont

# Repo-local defaults (the pipeline is self-contained, no thesis dependency)
PKL_PATH = "data/samples_data.pkl"
SPLIT_PATH = "data/class_split.pt"
STROKE_PATH = "data/decompose-stroke-3755.txt"
FONT_DIR = "data/fonts"
FONTS = {"simsun": "simsun.ttf", "simfang": "simfang.ttf"}
CONTENT_DIR = "data/kanji_content"
STYLE_DIR = "data/kanji_style"
PROBE_CONTENT_DIR = "data/kanji_probe_content"
PROBE_STYLE_DIR = "data/kanji_probe_style"
FILL = 255  # light paper, dark ink


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare kanji content (synthetic fonts) and style (HWDB handwritten) "
                    "datasets for the ArtFlow fine-tuning pipeline."
    )
    parser.add_argument("--size", type=int, default=256,
                        help="side length of the square images prepared for ArtFlow")
    parser.add_argument("--max-style-samples", type=int, default=30000,
                        help="global cap for the style set")
    parser.add_argument("--max-samples-per-char", type=int, default=60,
                        help="per-class cap used during round-robin style sampling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pkl-path", type=str, default=PKL_PATH)
    parser.add_argument("--split-path", type=str, default=SPLIT_PATH)
    parser.add_argument("--stroke-path", type=str, default=STROKE_PATH)
    parser.add_argument("--font-dir", type=str, default=FONT_DIR)
    parser.add_argument("--content-dir", type=str, default=CONTENT_DIR)
    parser.add_argument("--style-dir", type=str, default=STYLE_DIR)
    parser.add_argument("--probe-content-dir", type=str, default=PROBE_CONTENT_DIR)
    parser.add_argument("--probe-style-dir", type=str, default=PROBE_STYLE_DIR)
    parser.add_argument("--probe-chars", type=str, default=None,
                        help="comma-separated characters for the probe content set "
                             "(default: pick a few seeded at random from the active chars)")
    parser.add_argument("--invert-style", action="store_true",
                        help="invert the handwritten samples (if they have dark paper / light ink)")
    return parser.parse_args()


# -----------------------------
# Split + valid characters
# -----------------------------

def load_train_chars(split_path):
    split = torch.load(split_path, weights_only=True)
    return set(split["train_chars"])


def load_valid_chars(stroke_path):
    valid = set()
    with open(stroke_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            char = line.split("|")[0].strip()
            if char:
                valid.add(char)
    return valid


# -----------------------------
# Rendering (content)
# -----------------------------

def render_char(char, font_path, img_size):
    img = Image.new("L", (img_size, img_size), color=FILL)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, img_size)
    bbox = draw.textbbox((0, 0), char, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (img_size - w) // 2 - bbox[0]
    y = (img_size - h) // 2 - bbox[1]
    draw.text((x, y), char, fill=0, font=font)
    return img


def render_content(args, chars):
    count = 0
    for char in sorted(chars):
        for font_name, font_file in FONTS.items():
            font_path = os.path.join(args.font_dir, font_file)
            img = render_char(char, font_path, args.size)
            out_path = os.path.join(args.content_dir, f"{char}_{font_name}.png")
            img.save(out_path)
            count += 1
        if count and count % 2000 == 0:
            print(f"  content rendered: {count}")
    return count


# -----------------------------
# Style extraction (HWDB pkl)
# -----------------------------

def load_samples(pkl_path):
    print(f"Loading {pkl_path} ...")
    with open(pkl_path, "rb") as f:
        samples = pickle.load(f)
    print(f"  loaded {len(samples)} samples")
    return samples


def resize_and_pad(img, size):
    w, h = img.size
    scale = size / max(w, h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("L", (size, size), color=FILL)
    x = (size - new_w) // 2
    y = (size - new_h) // 2
    canvas.paste(img, (x, y))
    return canvas


def stratified_style_samples(samples, active_chars, max_total, max_per_char, seed):
    rng = random.Random(seed)
    by_char = {}
    for s in samples:
        tag = s["tag_code"]
        if tag in active_chars:
            by_char.setdefault(tag, []).append(s)

    # round-robin over the classes so coverage is balanced
    chars = sorted(by_char.keys())
    for c in chars:
        rng.shuffle(by_char[c])

    picked = []
    idx = {c: 0 for c in chars}
    remaining = {c: max_per_char for c in chars}
    total = 0
    while total < max_total:
        progressed = False
        for c in chars:
            if total >= max_total:
                break
            if remaining[c] <= 0 or idx[c] >= len(by_char[c]):
                continue
            picked.append(by_char[c][idx[c]])
            idx[c] += 1
            remaining[c] -= 1
            total += 1
            progressed = True
        if not progressed:
            break
    return picked


def save_style(args, samples):
    rng = random.Random(args.seed)
    count = 0
    for i, s in enumerate(samples):
        img = Image.fromarray(s["image"]).convert("L")
        if args.invert_style:
            img = Image.eval(img, lambda v: 255 - v)
        img = resize_and_pad(img, args.size)
        out_path = os.path.join(args.style_dir, f"style_{i:06d}.png")
        img.save(out_path)
        count += 1
        if count and count % 5000 == 0:
            print(f"  style saved: {count}")
    return count


# -----------------------------
# Probe sets
# -----------------------------

def make_probe(args, content_chars, style_samples):
    rng = random.Random(args.seed)
    probe_chars = content_chars
    if args.probe_chars:
        probe_chars = [c.strip() for c in args.probe_chars.split(",")]
    probe_chars = [c for c in probe_chars if c in content_chars][:3]

    for char in probe_chars:
        for font_name, font_file in FONTS.items():
            font_path = os.path.join(args.font_dir, font_file)
            img = render_char(char, font_path, args.size)
            img.save(os.path.join(args.probe_content_dir, f"{char}_{font_name}.png"))

    n_probe_style = min(8, len(style_samples))
    for s in rng.sample(style_samples, n_probe_style):
        img = Image.fromarray(s["image"]).convert("L")
        if args.invert_style:
            img = Image.eval(img, lambda v: 255 - v)
        img = resize_and_pad(img, args.size)
        img.save(os.path.join(args.probe_style_dir, f"probe_{img.size[0]:02d}x{img.size[1]:02d}_{s['tag_code']}.png"))


# -----------------------------
# Sanity checks
# -----------------------------

def report_polarity(args, content_chars, style_samples):
    import numpy as np
    content_img = render_char(sorted(content_chars)[0], os.path.join(args.font_dir, FONTS["simsun"]), 64)
    c_arr = np.asarray(content_img, dtype=np.float32)
    dark_content = float((c_arr <= 120).mean())

    dark_style = []
    for s in rng_sample(style_samples):
        a = np.asarray(s["image"], dtype=np.float32)
        dark_style.append(float((a <= 120).mean()))
    dark_style_avg = float(np.mean(dark_style)) if dark_style else 0.0

    print(f"Polarity check (fraction of dark pixels):")
    print(f"  content sample : {dark_content:.3%}")
    print(f"  style average  : {dark_style_avg:.3%}")
    if dark_style_avg < 0.001:
        print("WARNING: style seems to have almost no dark ink with paper=255.")
    if dark_style_avg > 0.5:
        print("WARNING: style seems to have dark PAPER instead of dark ink; consider --invert-style.")


def rng_sample(samples):
    return random.Random(0).sample(samples, min(5, len(samples)))


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()
    random.seed(args.seed)

    for d in (args.content_dir, args.style_dir, args.probe_content_dir, args.probe_style_dir):
        os.makedirs(d, exist_ok=True)

    print("1. Characters used by training (split x valid)")
    train_chars = load_train_chars(args.split_path)
    valid_chars = load_valid_chars(args.stroke_path)
    active_chars = train_chars & valid_chars
    print(f"  train chars: {len(train_chars)} | valid: {len(valid_chars)} | active: {len(active_chars)}")
    if not active_chars:
        raise SystemExit("No active characters: check split and stroke file.")

    print("2. Synthetic content")
    n_content = render_content(args, active_chars)
    print(f"  {n_content} content images -> {args.content_dir}")

    print("3. Handwritten style")
    samples = load_samples(args.pkl_path)
    style_samples = stratified_style_samples(samples, active_chars,
                                             args.max_style_samples,
                                             args.max_samples_per_char,
                                             args.seed)
    print(f"  selected {len(style_samples)} style samples (stratified, cap {args.max_style_samples})")
    n_style = save_style(args, style_samples)
    print(f"  {n_style} style images -> {args.style_dir}")

    print("4. Probe subsets")
    make_probe(args, active_chars, style_samples)
    print(f"  probe content/ -> style dirs filled in {args.probe_content_dir}, {args.probe_style_dir}")

    print("5. Sanity / polarity")
    report_polarity(args, active_chars, style_samples)

    print("DONE.")


if __name__ == "__main__":
    main()