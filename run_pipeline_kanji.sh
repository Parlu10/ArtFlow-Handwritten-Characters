#!/usr/bin/env bash
# End-to-end kanji pipeline:
#   1. prepare   -> render synthetic content + extract handwritten style (prepare_kanji_data.py)
#   2. finetune  -> fine-tune ArtFlow / AdaIN from the pretrained checkpoint (run_finetune_kanji.sh)
#   3. infer     -> stylize all synthetic kanji with the medoid style reference (infer_kanji.py)
#   4. postproc  -> convert the stylized PNGs to data/synthetic_handwritten.pkl (postprocess_thesis.py)
set -e

SKIP_PREPARE=0
SKIP_TRAIN=0
SKIP_INFER=0
SKIP_POSTPROCESS=0

usage() {
    echo "Usage: $0 [flags]"
    echo "Flags:"
    echo "  --skip-prepare      skip step 1 (data preparation)"
    echo "  --skip-train        skip step 2 (fine-tuning)"
    echo "  --skip-infer        skip step 3 (stylization)"
    echo "  --skip-postprocess  skip step 4 (pickle conversion)"
    echo "  -h, --help          show this help"
    exit 0
}

for arg in "$@"; do
    case "$arg" in
        --skip-prepare) SKIP_PREPARE=1 ;;
        --skip-train) SKIP_TRAIN=1 ;;
        --skip-infer) SKIP_INFER=1 ;;
        --skip-postprocess) SKIP_POSTPROCESS=1 ;;
        -h|--help) usage ;;
        *) echo "ERROR: unknown argument: $arg"; usage ;;
    esac
done

# ----------------------------- sanity checks -----------------------------

# venv python
if [ -x "venv/bin/python" ]; then
    VENV_BIN="venv/bin"
elif [ -x "venv/Scripts/python.exe" ]; then
    VENV_BIN="venv/Scripts"
else
    echo "ERROR: no python found in venv (neither venv/bin nor venv/Scripts)."
    echo "Create it first: python3 -m venv venv && pip install -r requirements_kanji.txt"
    exit 1
fi
export PATH="$VENV_BIN:$PATH"
echo "Using python: $(command -v python)"

# pretrained checkpoint for the resume (needed by step 2)
PRETRAINED="experiments/ArtFlow-AdaIN/glow.pth"
if [ "$SKIP_TRAIN" -eq 0 ] && [ ! -f "$PRETRAINED" ]; then
    echo "ERROR: pretrained checkpoint not found at $PRETRAINED"
    echo "Download the 'experiments' folder (ArtFlow-AdaIN) from Google Drive and"
    echo "place it under the repository root. See Readme.md -> Pretrained Models."
    exit 1
fi

# handwritten source data (needed by step 1)
if [ "$SKIP_PREPARE" -eq 0 ] && [ ! -f "data/samples_data.pkl" ]; then
    echo "ERROR: data/samples_data.pkl not found (HWDB1.0 samples)."
    echo "Copy it into data/ or build it from the .gnt files. See Readme.md."
    exit 1
fi

# ----------------------------- pipeline steps -----------------------------

if [ "$SKIP_PREPARE" -eq 0 ]; then
    echo "[1/4] prepare "
    python prepare_kanji_data.py
else
    echo "[1/4] prepare SKIPPED"
fi

if [ "$SKIP_TRAIN" -eq 0 ]; then
    echo "[2/4] finetune"
    bash run_finetune_kanji.sh
else
    echo "[2/4] finetune SKIPPED"
fi

if [ "$SKIP_INFER" -eq 0 ]; then
    echo "[3/4] infer"
    python infer_kanji.py
else
    echo "[3/4] infer SKIPPED"
fi

if [ "$SKIP_POSTPROCESS" -eq 0 ]; then
    echo "[4/4] postprocess"
    python postprocess_thesis.py
else
    echo "[4/4] postprocess SKIPPED"
fi

echo "Pipeline done."