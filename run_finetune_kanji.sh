#!/usr/bin/env bash
# Fine-tune ArtFlow (AdaIN) on the kanji datasets:
#   content = synthetic kanji rendered with fonts (data/kanji_content)
#   style   = handwritten HWDB samples          (data/kanji_style)
# Output checkpoint: experiments/ArtFlow-Kanji/glow.pth
set -e

RESUME="experiments/ArtFlow-AdaIN/glow.pth"

if [ ! -f "$RESUME" ]; then
    echo "ERROR: pretrained checkpoint not found at $RESUME"
    echo "Download the 'experiments' folder (ArtFlow-AdaIN) from Google Drive and"
    echo "place it under the repository root. See Readme.md -> Pretrained Models."
    exit 1
fi

if [ ! -d "data/kanji_content" ] || [ ! -d "data/kanji_style" ]; then
    echo "ERROR: prepared datasets missing."
    echo "Run first: python prepare_kanji_data.py"
    exit 1
fi

python -u train.py \
    --content_dir data/kanji_content \
    --style_dir data/kanji_style \
    --save_dir experiments/ArtFlow-Kanji \
    --resume "$RESUME" \
    --n_flow 8 \
    --n_block 2 \
    --operator adain \
    --batch_size 8 \
    --lr 1e-5 \
    --max_iter 50000 \
    --print_interval 100 \
    --save_model_interval 10000