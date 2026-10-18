#!/bin/bash
# Run the whole benchmark of docs/research/2026-10-19-benchmark.md on devbox, in the mirror.
#
#   scripts/dx bench "tmux new-session -d -s bench 'bash tools/bench/run_devbox.sh'"
#
# Every compressor runs on processors 10-13 only, at the lowest priority, one compressor at a
# time. The log is ~/bench/run.log. The results are ~/bench/results.json. A second start
# continues where the first one stopped.
# Requirements: apt packages zstd, brotli, p7zip-full; paq8px v217 in ~/bench/src/paq8px/build
# (its build/build-linux-with-cmake.sh); the taco binary in ~/bench/target/release, built from
# commit 71a1192 (git archive 71a1192) with CARGO_TARGET_DIR=~/bench/target. A later
# build can have the --profile option (default auto) and gives other sizes.
set -euo pipefail
cd "$(dirname "$0")/../.."
# NOTE: if memory runs out, the kernel stops this job first, not the work of other people.
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true
B=$HOME/bench
PAQ=(--paq8px "$B/src/paq8px/build/paq8px")
TACO=(--taco "$B/target/release/taco" --model "$PWD/tools/data/out/gpt2.hsm")
run() { taskset -c 10-13 nice -n 19 python3 -I tools/bench/run.py measure --work "$B/work" \
  --out "$B/results.json" "$@" 2>&1 | tee -a "$B/run.log"; }
# 1. The general compressors, both sets. Each fast timing is the best of 3 runs.
run --compressors "gzip -9" "bzip2 -9" "xz -9e" "zstd -19" "zstd --ultra -22 --long=27" \
  "brotli -q 11 -w 24" "7z PPMd o16 256m"
# 2. paq8px -8 on the samples, one file at a time.
run "${PAQ[@]}" --compressors "paq8px -8" --sets samples
# 3. Taco on the samples: 2 files at a time, 2 threads each (a 32 KiB sample has 2 blocks).
run "${TACO[@]}" --compressors taco --sets samples --taco-jobs 2 --taco-threads 2
# 4. paq8px -8 on the whole files, one file at a time (single thread, up to 2.4 GB).
run "${PAQ[@]}" --compressors "paq8px -8" --sets whole --paq8px-jobs 1
# 5. Taco on whole files, one file at a time with 4 threads: the 6 logs, the 3 binary files,
#    alice29.txt and option.rs. Whole files of 32 KiB or less reuse the samples. The other
#    whole files (4,708,866 bytes) take about 4 h with the fast CPU engine of 2026-10-19
#    (est., from the sample speed), so this set is reduced.
run "${TACO[@]}" --compressors taco --sets whole --taco-jobs 1 --taco-threads 4 \
  --taco-whole HDFS_2k.log Apache_2k.log BGL_2k.log Linux_2k.log OpenSSH_2k.log Spark_2k.log \
  kennedy.xls ptt5 sum alice29.txt option.rs
echo "run_devbox: done" | tee -a "$B/run.log"
