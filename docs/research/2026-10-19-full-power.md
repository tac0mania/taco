# GPU speed at full power (RTX 3060 Laptop, 95 W)

Date: 2026-10-19. Machine: devbox (RTX 3060 Laptop GPU, 6 GB, driver 610.74). Rohit removed
the 45 W power cap. `nvidia-smi -q -d POWER`: current power limit 95.00 W, default 80.00 W.
During the engine runs the GPU used 94.6 to 94.9 W at 98 % load, 1,430 to 1,520 MHz, 83 to
88 °C (measured, sampled every 5 s). No thermal slowdown was reported. The benchmark held
`~/scratch/timing.lock` and ran on processors 0-5. One CPU-only job of another agent ran at
the same time on other processors.

All numbers in this note are measured. The 45 W numbers come from the cudawide agent
(2026-10-19, the same commit of `hardshell-cuda`) and from `2026-10-18-gpu-backend.md` §7.

## 1. Engine (`crates/hardshell-cuda`, `examples/gpt2 bench <model> 8 128 224`, commit dfee08c)

Tokens per second, coder op (24 bytes read back per token).

| Model | Prefill 1 x 512 | Prefill 8 x 1,024 | Decode, full window (batch 224, positions 0 to 1,023) | Decode batch 8 |
|---|---:|---:|---:|---:|
| `gpt2.hsm` v1, 45 W | 49,815 | 25,600 | 8,691 to 9,195 | 2,742 |
| **`gpt2.hsm` v1, 95 W** | **63,727** | **50,203** | **20,472** | **3,051** |
| `gpt2_wide.hsm` v2, 45 W | 30,177 | 15,068 | 7,788 (batch 128) | 3,584 |
| **`gpt2_wide.hsm` v2, 95 W** | **39,448** | **31,195** | **15,299** | **4,437** |
| `distil_logs.hsm` v1, 45 W | 78,632 | 42,673 | 16,745 (batch 128) | 4,690 |
| **`distil_logs.hsm` v1, 95 W** | **83,135** | **57,553** | **30,319** | **5,759** |
| `distil_logs_wide.hsm` v2, 45 W | 44,258 | 22,215 | 12,740 (batch 128) | 5,720 |
| **`distil_logs_wide.hsm` v2, 95 W** | **53,713** | **43,278** | **22,667** | **6,759** |

- The speed gate of the plan (GPT-2 small, 10,000 tokens/s or more, batched decode, bit-exact)
  passes: 20,472 tokens/s over a full window.
- Full-window decode is 2.2 times the 45 W value. Prefill 8 x 1,024 is 2.0 times.
- At position 512, attention takes 6.9 ms of the 10.6 ms step (GPT-2 v1, batch 224). The KV
  cache read is now the largest cost.

## 2. The command (`taco fold` and `taco unfold`, profile text, CUDA, 23 whole files)

Binary: the overlap working tree on top of dda9edd (not committed), the same binary as the
45 W run in `~/bench/whole-gpu`. 23 of 23 files restored byte for byte. Sizes are equal to
the 45 W run (the power changes no bit).

| File | Bytes | Fold 45 W | Fold 95 W | Unfold 45 W | Unfold 95 W |
|---|---:|---:|---:|---:|---:|
| lcet10.txt | 426,754 | 7.2 s (59 KB/s) | 3.3 s (129 KB/s) | 22.5 s (19 KB/s) | 16.5 s (26 KB/s) |
| plrabn12.txt | 481,861 | 9.2 s (52 KB/s) | 5.7 s (85 KB/s) | 20.7 s (23 KB/s) | 15.4 s (31 KB/s) |
| pg24264.txt (Chinese) | 2,663,455 | 85.6 s (31 KB/s) | 56.4 s (47 KB/s) | 173.5 s (15 KB/s) | 102.7 s (26 KB/s) |
| alice29.txt | 152,089 | 4.7 s | 2.5 s | 17.0 s | 11.8 s |
| cp.html | 24,603 | 3.0 s | 1.4 s | 16.3 s | 13.4 s |

Fold gains up to 2.2 times. Unfold gains only 1.2 to 1.7 times.

## 3. Why unfold gains less

A block decodes one token after the other. Thus the time of unfold has a lower limit: the
tokens of one block multiplied by the time of one step. A 16 KiB block of English is about
4,000 tokens (est.). A step at a small batch takes about 2.6 ms (batch 8, measured above).
Thus one block needs about 10 to 12 s (est.), and a small file cannot unfold faster than its
largest block. Examples: cp.html (2 blocks) 13.4 s, textwrap.py (2 blocks) 12.6 s.

Ways to lower this limit (not measured): smaller blocks for small files (more parallel
streams, larger output), speculative decoding (more tokens for each step), and a shorter step
at a small batch.
