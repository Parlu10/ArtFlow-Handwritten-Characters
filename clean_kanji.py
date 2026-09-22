import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter, label
from tqdm import tqdm

# Calibrated against the real HWDB samples (data/samples_data.pkl):
#   median ink level ~127, paper median = 255, paper grain ~2.2, dark coverage ~0.11.
# See the comparison in the analysis: stylized output keeps a gray veil/halo around
# the strokes and washed-out ink; this script removes the veil without binarizing.
DEFAULT_OUTPUT = "output_kanji_clean"


def otsu_threshold(a):
    hist, _ = np.histogram(a.ravel(), bins=256, range=(0, 255))
    hist = hist.astype(np.float64)
    total = hist.sum()
    sum_all = np.dot(np.arange(256), hist)
    sum_b, w_b = 0.0, 0.0
    best_t, best_var = 0, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        mean_b = sum_b / w_b
        mean_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (mean_b - mean_f) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return best_t


def deterministic_seed(name):
    import hashlib
    digest = hashlib.md5(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def clean(image, amt=1.0, blur_radius=1.4, dilate=1,
          paper_frac=0.30, paper_sigma=4.5,
          min_comp_rel=0.01, min_comp_abs=24, seed=0):
    """Remove the gray veil/halo around the strokes and remap the ink tone.

    Steps:
      1. unsharp mask: narrows the low-pass smear around the strokes;
      2. Otsu ink core + small-component removal (ghosts / border artifacts);
      3. 'support' = ink core dilated to keep the natural anti-aliased edge;
      4. tone remap inside the support so the median ink lands near 127
         (target level of the real handwritten samples);
      5. paper -> 255 with the sparse speckle grain of the real scans.
    The texture of the strokes is preserved (no binarization).
    """
    f = image.astype(np.float32)
    sharp = np.clip(f + amt * (f - gaussian_filter(f, blur_radius)), 0, 255)

    core = sharp < otsu_threshold(sharp)
    labels, n = label(core)
    if n > 1:
        sizes = np.bincount(labels.ravel())[1:]
        big = sizes >= max(sizes.max() * min_comp_rel, min_comp_abs)
        core = np.isin(labels, np.nonzero(big)[0] + 1)

    support = binary_dilation(core, iterations=dilate)

    paper_level = float(np.percentile(sharp, 97.5))
    med_core = float(np.median(sharp[core])) if core.any() else paper_level
    gain = np.clip((255 - 127) / max(255 - med_core, 1), 1.0, 3.0)

    out = np.where(support, 255.0 - gain * (paper_level - sharp), 255.0)
    out = np.clip(out, 0, 255)

    rng = np.random.default_rng(seed)
    paper = out >= 255.0
    speck = rng.random(paper.sum()) < paper_frac
    drops = np.abs(rng.normal(0, paper_sigma, size=paper.sum()))
    tone = np.full(paper.sum(), 255.0)
    tone[speck] = 255.0 - drops[speck]
    out[paper] = tone

    return np.clip(out, 0, 255).astype(np.uint8)


def batch_summary(paths):
    grains, bgs = [], []
    for path in paths:
        a = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        border = np.concatenate([a[:8, :].ravel(), a[-8:, :].ravel(),
                                 a[:, :8].ravel(), a[:, -8:].ravel()])
        bg = float(np.median(border))
        bgs.append(bg)
        clear = a[a > bg - 20]
        grains.append(float(clear.std()) if clear.size else float("nan"))
    grains = np.array(grains)
    grains = grains[~np.isnan(grains)]
    return {
        "n": len(paths),
        "bg_median": float(np.median(bgs)),
        "grain_median": float(np.median(grains)),
        "grain_std": float(np.std(grains)),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean the stylized kanji: remove the gray veil around the "
                    "strokes and restore real-handwriting ink/paper levels. "
                    "Keeps the originals untouched (writes to a new folder)."
    )
    parser.add_argument("--input", type=str, default="output_kanji",
                        help="directory with the stylized PNGs (default: output_kanji)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="directory for the cleaned PNGs (default: output_kanji_clean)")
    # Cleaning knobs (defaults calibrated on the real HWDB statistics)
    parser.add_argument("--amt", type=float, default=1.0,
                        help="unsharp-mask amount (veil suppression)")
    parser.add_argument("--blur-radius", type=float, default=1.4,
                        help="unsharp-mask gaussian radius in px")
    parser.add_argument("--dilate", type=int, default=1,
                        help="dilation iterations of the ink core (keeps the edge)")
    parser.add_argument("--paper-frac", type=float, default=0.30,
                        help="fraction of paper pixels receiving the real-scan speckle")
    parser.add_argument("--paper-sigma", type=float, default=4.5,
                        help="speckle intensity (target paper grain ~2.2)")
    parser.add_argument("--min-comp-rel", type=float, default=0.01,
                        help="drop components smaller than this fraction of the largest")
    parser.add_argument("--min-comp-abs", type=float, default=24,
                        help="absolute pixel floor for dropped components")
    parser.add_argument("--no-paper-noise", action="store_true",
                        help="leave the paper at flat 255 (no scan grain)")
    return parser.parse_args()


def main():
    args = parse_args()
    src = args.input
    dst = args.output
    if not os.path.isdir(src):
        print(f"ERROR: input dir not found: {src}", file=sys.stderr)
        sys.exit(1)
    paths = sorted(glob.glob(os.path.join(src, "*.png")))
    if not paths:
        print(f"ERROR: no PNGs in {src}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(dst, exist_ok=True)
    paper_noise = 0.0 if args.no_paper_noise else args.paper_sigma

    done = []
    for path in tqdm(paths, desc="cleaning"):
        stem = os.path.splitext(os.path.basename(path))[0]
        a = np.asarray(Image.open(path).convert("L"))
        clean_img = clean(
            a,
            amt=args.amt,
            blur_radius=args.blur_radius,
            dilate=args.dilate,
            paper_frac=args.paper_frac,
            paper_sigma=paper_noise,
            min_comp_rel=args.min_comp_rel,
            min_comp_abs=args.min_comp_abs,
            seed=deterministic_seed(stem),
        )
        out_path = os.path.join(dst, os.path.basename(path))
        Image.fromarray(clean_img).save(out_path)
        done.append(out_path)

    total = len(done)
    print(f"cleaned {total} images -> {dst}  (originals in {src} untouched)")
    if total:
        samples = done[:: max(1, total // 300)]
        summary = batch_summary(samples)
        print("batch summary (sampled):")
        print(f"  n            : {summary['n']}")
        print(f"  paper bg med : {summary['bg_median']:.1f}  (target 255)")
        print(f"  paper grain  : median {summary['grain_median']:.2f} "
              f"(target ~2.2, real HWDB range 1.5-2.8)")
        print(f"  grain spread : {summary['grain_std']:.2f}")


if __name__ == "__main__":
    main()