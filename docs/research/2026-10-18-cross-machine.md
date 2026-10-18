# Research: cross-machine check, Mac against devbox (2026-10-18)

This note records the phase 2 gate (plan §11.3, §12). `scripts/xcheck` ran it.

## Setup

- Mac: Apple M2 (ARM), macOS, Rust 1.99.0. devbox: AMD Ryzen 7 5800H (x86, AVX2), Ubuntu
  24.04 in WSL2, Rust 1.99.0.
- Each machine made its own `gpt2.hsm` with `python -m convert.gpt2 --recipe clip`. Both files
  have the SHA-256 `aaa03eb96e9ad7957432168c8ac6b4a99ed55d57e637e4786a664b51001ba5c2`
  (measured).
- Nine inputs: English (4 KiB and 32 KiB), an HTML page, a server log, a Chinese text (Hong Lou Meng), an
  Excel file, 3,000 random bytes, `dasdasdasdada`, and an empty file.

## Results (all measured)

| Check | Result |
|---|---|
| Workspace tests on devbox, with the golden hashes that the Mac recorded | 234 of 234 pass |
| Mac folds, devbox unfolds | 9 of 9 restored byte for byte |
| devbox folds, Mac unfolds | 9 of 9 restored byte for byte |
| Same input, `.taco` file of the Mac against the `.taco` file of devbox | 9 of 9 identical |

| Input | Bytes | `.taco` bytes |
|---|---:|---:|
| english_32k.txt | 32,768 | 8,432 |
| page.html | 24,603 | 4,759 |
| server_8k.log | 8,192 | 1,574 |
| other_language_6k.txt | 6,144 | 2,863 |
| english_4k.txt | 4,096 | 1,237 |
| binary_4k.xls | 4,096 | 4,189 (raw) |
| random_3000.bin | 3,000 | 3,093 (raw) |
| dasdas.txt | 13 | 100 |
| empty.bin | 0 | 88 |

## Notes

- The run took 40 minutes, because the reference engine is slow (about 0.2 to 0.5 KB/s).
- The 88-byte header is large for very small inputs (13 bytes become 100 bytes). A shorter
  header for small inputs is a format question for phase 5.
- The Mac and devbox both use CPUs. A GPU is not yet in the check.
