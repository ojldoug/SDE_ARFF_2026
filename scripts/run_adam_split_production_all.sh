#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

for experiment in ex1 ex2 ex3 ex4 ex5 ex6 ex7 ex8
do
    echo
    echo "================================================================"
    echo "STARTING ADAM SPLIT ${experiment}"
    date
    echo "================================================================"

    python scripts/run_production_batch.py \
        adam_split "${experiment}" \
        --device 0 \
        --start-seed 0 \
        --n-runs 30 \
        --resume

    echo
    echo "================================================================"
    echo "COMPLETED ADAM SPLIT ${experiment}"
    date
    echo "================================================================"
done

echo
echo "================================================================"
echo "ALL FOURIER SPLIT PRODUCTION BATCHES COMPLETE"
date
echo "================================================================"
