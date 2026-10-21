#!/usr/bin/env bash
# Fine-tune ArtFlow on the kanji datasets:
#   content = synthetic kanji rendered with fonts (data/kanji_content)
#   style   = handwritten HWDB samples          (data/kanji_style)
#
# Usage: ./run_finetune_kanji.sh [--operator {wct,adain}]
#   wct   (default) -> resume experiments/ArtFlow-WCT/glow.pth
#                      save    experiments/ArtFlow-Kanji-WCT/glow.pth
#   adain           -> resume experiments/ArtFlow-AdaIN/glow.pth
#                      save    experiments/ArtFlow-Kanji/glow.pth
set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OPERATOR="wct"

usage() {
    echo "Usage: $0 [--operator {wct,adain}]"
    echo "  --operator wct     fine-tune the WCT variant (default)"
    echo "  --operator adain   fine-tune the AdaIN variant"
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --operator)
            if [ -z "$2" ]; then
                echo "ERROR: --operator requires a value (wct or adain)"
                exit 1
            fi
            OPERATOR="$2"
            shift 2
            ;;
        --operator=*) OPERATOR="${1#*=}"; shift ;;
        -h|--help) usage ;;
        *) echo "ERROR: unknown argument: $1"; usage ;;
    esac
done

case "$OPERATOR" in
    wct)
        RESUME="experiments/ArtFlow-WCT/glow.pth"
        SAVE_DIR="experiments/ArtFlow-Kanji-WCT"
        ;;
    adain)
        RESUME="experiments/ArtFlow-AdaIN/glow.pth"
        SAVE_DIR="experiments/ArtFlow-Kanji"
        ;;
    *)
        echo "ERROR: unknown operator '$OPERATOR' (expected 'wct' or 'adain')"
        exit 1
        ;;
esac
echo "Operator: $OPERATOR"

if [ ! -f "$RESUME" ]; then
    echo "ERROR: pretrained checkpoint not found at $RESUME"
    echo "Download the 'experiments' folder from Google Drive and"
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
    --save_dir "$SAVE_DIR" \
    --resume "$RESUME" \
    --n_flow 8 \
    --n_block 2 \
    --operator "$OPERATOR" \
    --batch_size 2 \
    --lr 5e-5 \
    --max_iter 85000 \
    --mse_weight 0.3 \
    --content_weight 0.5 \
    --print_interval 100 \
    --save_model_interval 10000
