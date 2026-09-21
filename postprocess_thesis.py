import argparse
import os
import pickle
import sys

import numpy as np
from pathlib import Path
from PIL import Image

INPUT_DIR = "output_kanji"
OUTPUT_PKL = "data/synthetic_handwritten.pkl"
FONTS = ("simsun", "simfang")
SUFFIX = "_stylized"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the stylized kanji produced by infer_kanji.py into "
                    "the thesis dataset format (list of {'tag_code', 'image'}, "
                    "same schema as samples_data.pkl)."
    )
    parser.add_argument("--input", type=str, default=INPUT_DIR,
                        help="directory with the stylized PNGs")
    parser.add_argument("--output", type=str, default=OUTPUT_PKL,
                        help="path of the output pickle")
    parser.add_argument("--fonts", type=str, default=",".join(FONTS),
                        help="comma-separated font suffixes used in the filenames")
    return parser.parse_args()


def parse_stem(stem, fonts):
    """Parse '<char>_<font>_stylized' into (char, font). Returns (None, None)."""
    if not stem.endswith(SUFFIX):
        return None, None
    body = stem[: -len(SUFFIX)]
    for font in fonts:
        token = "_{:s}".format(font)
        if body.endswith(token):
            return body[: -len(token)], font
    return None, None


def collect_stylized(input_dir, fonts):
    samples = []
    skipped = []
    for path in sorted(Path(input_dir).glob("*")):
        if path.suffix.lower() != ".png":
            continue
        char, font = parse_stem(path.stem, fonts)
        if char is None:
            skipped.append(path.name)
            continue
        img = Image.open(str(path)).convert("L")
        samples.append({"tag_code": char, "image": np.asarray(img, dtype=np.uint8),
                        "_font": font})
    return samples, skipped


def sanity_report(samples):
    n = len(samples)
    by_char = {}
    for s in samples:
        by_char.setdefault(s["tag_code"], []).append(s)
    n_chars = len(by_char)
    print("Sanity report:")
    print(f"  total samples : {n}")
    print(f"  unique chars  : {n_chars}")
    if n_chars:
        per_char = sorted((len(v) for v in by_char.values()))
        print(f"  samples/char  : min {per_char[0]} | median {per_char[n_chars // 2]} "
              f"| max {per_char[-1]}")
    single = [c for c, v in by_char.items() if len(v) < len(FONTS)]
    if single:
        print(f"  WARNING: {len(single)} chars with fewer than {len(FONTS)} samples "
              f"(missing a font): {single[:10]}...")
    if not n:
        print("  WARNING: no samples produced; the output pickle would be empty.")

    arrays_ok = all(isinstance(s["image"], np.ndarray) and s["image"].dtype == np.uint8
                    and s["image"].ndim == 2 for s in samples)
    keys_ok = all(set(s.keys()) == {"tag_code", "image"} for s in samples)
    print(f"  dtype/ndim ok : {arrays_ok}")
    print(f"  keys exact    : {keys_ok}")
    return arrays_ok and keys_ok


def main():
    args = parse_args()
    fonts = tuple(f.strip() for f in args.fonts.split(",") if f.strip())
    if not fonts:
        print("ERROR: --fonts must list at least one font")
        sys.exit(1)

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"ERROR: input dir not found: {input_dir}")
        sys.exit(1)

    samples, skipped = collect_stylized(input_dir, fonts)
    print(f"Found {len(samples)} stylized samples in {input_dir} "
          f"(skipped {len(skipped)} files)")

    for s in samples:
        s.pop("_font", None)

    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    if not sanity_report(samples):
        print("ERROR: sanity check failed, not writing the pickle.")
        sys.exit(1)

    with open(out_path, "wb") as f:
        pickle.dump(samples, f)
    print(f"Saved {len(samples)} samples -> {out_path}")


if __name__ == "__main__":
    main()