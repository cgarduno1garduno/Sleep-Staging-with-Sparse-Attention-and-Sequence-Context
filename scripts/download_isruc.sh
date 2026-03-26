#!/usr/bin/env bash
# Download ISRUC-Sleep Subgroup 1 (100 subjects) from dataset.isr.uc.pt
# Usage: bash scripts/download_isruc.sh
# Resumable: skips subjects already extracted.

set -euo pipefail

BASE_URL="https://dataset.isr.uc.pt/ISRUC_Sleep/subgroupI"
OUT_DIR="$(cd "$(dirname "$0")/.." && pwd)/data_ISRUC/subgroup1"
N_SUBJECTS=100

mkdir -p "$OUT_DIR"

ok=0
skip=0
fail=0
failed_subjects=()

for i in $(seq 1 $N_SUBJECTS); do
    subj_dir="$OUT_DIR/$i"

    # Skip if already extracted (subject dir with .rec file exists)
    if [ -d "$subj_dir" ] && ls "$subj_dir"/*.rec &>/dev/null 2>&1; then
        echo "[$i/$N_SUBJECTS] Already extracted — skipping"
        ((skip++)) || true
        continue
    fi

    echo -n "[$i/$N_SUBJECTS] Downloading subject $i ... "
    rar_file="$OUT_DIR/${i}.rar"

    if curl -L -k --max-time 120 --retry 3 --retry-delay 5 \
        -o "$rar_file" \
        "$BASE_URL/${i}.rar" 2>/dev/null; then
        echo -n "extracting ... "
        if bsdtar -xf "$rar_file" -C "$OUT_DIR" 2>/dev/null; then
            rm -f "$rar_file"
            echo "done"
            ((ok++)) || true
        else
            echo "EXTRACT FAILED"
            rm -f "$rar_file"
            ((fail++)) || true
            failed_subjects+=("$i")
        fi
    else
        echo "DOWNLOAD FAILED"
        rm -f "$rar_file" 2>/dev/null || true
        ((fail++)) || true
        failed_subjects+=("$i")
    fi
done

echo ""
echo "============================="
echo "Done: $ok downloaded, $skip skipped, $fail failed"
if [ ${#failed_subjects[@]} -gt 0 ]; then
    echo "Failed subjects: ${failed_subjects[*]}"
fi
echo "Output: $OUT_DIR"
