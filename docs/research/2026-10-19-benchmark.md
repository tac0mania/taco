# Research: head-to-head benchmark (2026-10-19)

This note compares Taco with general compressors and with a strong context-mixing compressor
(paq8px). It gives the numbers for the README. All numbers in sections 1 to 7 are measured on
devbox on 2026-10-18, unless a number has a different label. Section 8 gives published numbers
from other people. Do not mix the two kinds of numbers.

The Taco build is commit `71a1192`: GPT-2 small alone, the fast CPU engine of 2026-10-19,
container version 2 with zstd blocks. The logs profile (match model and mixer) and the GPU
backends are not in this build. Section 7 gives their numbers from other notes.

## 1. Results in short

- **paq8px -8 is the strongest compressor in this test.** On the 23 samples of 32 KiB, it gives
  1.320 bits per byte (all files). On the whole files it gives 1.141 (measured).
- **Taco on the samples: 1.667 bits per byte for all 23 files** (measured). This is 6.5 %
  smaller than the best general compressor (brotli -q 11 -w 24, 1.783) and 26.3 % larger than
  paq8px.
- **Taco wins on English, code and HTML samples.** It is 22 % to 31 % smaller than the best
  general compressor and 4.6 % to 11.7 % smaller than paq8px on these classes. Taco is the
  smallest of all compressors on 8 of these 10 samples. The two other samples, asyoulik.txt
  and fields.c, have many tab characters.
- **Taco loses on other languages, logs, JSON and binary data.** On the samples, paq8px is 2.1
  times smaller on logs (0.422 against 0.866 bits per byte), 1.8 times on JSON and 1.3 times on
  German, French and Chinese. brotli is also smaller than Taco on these classes.
- **On whole files, Taco does not gain from the length of the file.** Each block of 16 KiB
  starts the model context again. On whole alice29.txt, Taco gives 2.048 bits per byte, 7z PPMd
  2.049 and paq8px 1.738. On the 16 whole files that Taco measured, paq8px is 2.6 times smaller
  and xz -9e is 22 % smaller (measured).
- **Speed (fast CPU engine, 4 processors):** on whole text, Taco folds at about 2.0 KB/s and
  unfolds at about 0.9 KB/s (alice29.txt, 627 and 274 tokens/s). paq8px -8 gives about 6 KB/s
  in each direction on 1 processor. xz -9e compresses at 671 KB/s. Taco used 245 to 481 MiB of
  memory, paq8px up to 2,276 MiB.
- **The GPU backends are faster** (other note, RTX 3060): 12,012 to 22,355 tokens/s for
  prefill and 7,229 tokens/s for decode over a full window. They are not in `taco fold` yet.
- **The logs profile is not in `taco fold` yet.** Its simulated result on the 6 whole logs is
  0.600 bits per byte (est., from the logs note). This is about 9 % smaller than 7z PPMd and
  xz, but 72 % larger than paq8px (0.348, measured).
- **cmix** did not run. It needs at least 32 GB of RAM (published, cmix README). devbox has
  12 GB. Section 8 gives its published enwik8 and enwik9 sizes.

## 2. Setup

| Item | Value |
|---|---|
| Machine | devbox: AMD Ryzen 7 5800H, Ubuntu 24.04 in WSL2 (Linux 6.18.54.1), 12 GB for Linux |
| Processors | `taskset -c 10-13` (4 logical processors), `nice -n 19`. No GPU. |
| Load | Other agents used devbox at the same time. The 1-minute load average on 14 processors was 11 to 22 for the fast compressors, 2 to 46 for paq8px, and 0.8 to 23.8 for Taco. Each timing has its load average in the results file and in section 6. |
| Order | One compressor at a time, except as stated in the notes below |
| Python | 3.12.3 (system), standard library only |
| Results files | `~/bench/results.json` (all compressors except paq8px on the whole files) and `~/bench/results-paq8px-whole.json` on devbox, copied to `tools/data/bench/` (ignored by git) |

| Compressor | Version | Source | Options |
|---|---|---|---|
| gzip | 1.12 | Ubuntu | `-9 -n` |
| bzip2 | 1.0.8 | Ubuntu | `-9` |
| xz | 5.4.5 | Ubuntu | `-9e -T1` |
| zstd | 1.5.5 | Ubuntu package `zstd` (installed for this test) | `-19`, and `--ultra -22 --long=27` (decompression with `--long=27`) |
| brotli | 1.1.0 | Ubuntu package `brotli` (installed for this test) | `-q 11 -w 24` |
| 7-Zip PPMd | 7-Zip 23.01 | Ubuntu package `p7zip-full` (installed for this test) | `7z a -t7z -m0=PPMd:mem=256m:o=16` |
| paq8px | v217 (commit `c84f576f`, 2026-09-28) | github.com/hxim/paq8px, built with its `build/build-linux-with-cmake.sh` | `-8` (2,433 MB level, published in its help text). No LSTM (`L`), no pre-training (`E`, `T`). |
| Taco | commit `71a1192` (the fast CPU engine of 2026-10-19 and container version 2), release build. The crates, `Cargo.toml` and `Cargo.lock` on devbox have the SHA-256 of `git archive 71a1192` (checked file by file). | this repository | `taco fold` and `taco unfold`, default block size 16,384 bytes, model `gpt2.hsm` (SHA-256 `aaa03eb9…a5c2`, checked). This build has no `--profile` option: every block is coded with GPT-2 alone, or stored as zstd or raw (ADR 0011 §4). |

Notes on the method:

- Sizes are the size of the compressed file. They include all headers: the gzip header, the 7z
  archive header and file name, the paq8px header and file name, and the Taco container (104
  bytes and 5 bytes for each block in version 2). Headers are important for the small files
  (3.7 KB to 4.2 KB).
- Each output was decompressed and compared with the input byte for byte. A difference stops
  the script. All outputs in this note were restored byte for byte, also all 23 Taco samples
  and all 11 Taco whole files (the logs `run-taco.log` and `run-taco-whole.log` show no
  error, and every record has `"restored": "byte for byte"`).
- Time is wall time, measured around the process, so it includes the process start and the
  load of the model file. For the fast compressors, each time is the best of 3 runs. paq8px and
  Taco ran 1 time.
- Peak memory is the maximum resident set size from GNU time (`%M`). The table gives the
  largest value over the files of the set.
- Speed is the total input bytes divided by the total time. 1 KB is 1,000 bytes. Token counts
  come from the Python BPE of `tools/baseline`, with a new start at each 16 KiB block (ADR
  0007).
- Taco on the samples: 2 files at a time, each with `--threads 2`. A 32 KiB sample has 2
  blocks, so 2 threads is the largest useful thread count for one sample. The speed in section
  6 is the speed of one process (2 threads). The 4 processors together gave about 2 times
  this speed.
- Taco on the whole files: 1 file at a time with `--threads 4`.
- The draft of this note had 2 Taco samples from the reference engine (crates as in `3cc17a6`,
  container version 1, load average 37 to 44). They are kept in
  `tools/data/bench/results-taco-reference.json`. The fast engine gives the same coded
  payloads: the two files are 8,448 and 9,923 bytes now and were 8,432 and 9,907 bytes, which
  is exactly the 16 more header bytes of container version 2.

## 3. Inputs

Two sets:

- **samples:** the phase 0 samples (`tools/baseline/corpus.toml`: 32 KiB from `offset`) and
  the first 32 KiB of 4 more Loghub 2k logs (BGL, Linux, OpenSSH, Spark). 23 files, 653,243
  bytes.
- **whole:** the 19 whole corpus files and the 4 more whole Loghub 2k logs. 23 files, 8,001,310
  bytes. Gutenberg files include their English licence text.

The logs are the Loghub 2k samples from github.com/logpai/loghub (`master`). Their sizes and
SHA-256 are in `tools/bench/run.py` and `tools/logs/logs.toml`. The classes:

| Class | Files |
|---|---|
| English | alice29.txt, asyoulik.txt, lcet10.txt, plrabn12.txt |
| Other languages | pg22367.txt (German), pg4650.txt (French), pg24264.txt (Chinese) |
| Code | fields.c, grammar.lsp, textwrap.py, option.rs, xargs.1 (troff source) |
| HTML | cp.html |
| Logs | HDFS, Apache, BGL, Linux, OpenSSH, Spark (Loghub 2k) |
| JSON | twitter.json |
| Binary | kennedy.xls, ptt5, sum |

**Taco on the whole files: a reduced set.** On whole text, Taco needs about 5.2 ms for each
token, fold and unfold together, on 4 processors (measured on alice29.txt). The 7 files that
Taco did not measure (below) have 2,780,757 tokens, so they take about 4 h (est.). pg24264.txt
alone takes about 2.9 h (est.). Thus Taco ran on a reduced set of 16 whole files (3,292,444
bytes). The run took 66 min (measured):

- the 6 logs, the 3 binary files, alice29.txt and option.rs (measured on the whole file);
- fields.c, grammar.lsp, textwrap.py, xargs.1 and cp.html: these files are 32 KiB or less, so
  the whole file is the sample. The script reuses the sample record.

Not measured with Taco: asyoulik.txt, lcet10.txt, plrabn12.txt, the 3 files in other languages
and twitter.json (4,708,866 bytes). Thus the whole-file rows for English, other languages,
JSON and the full overall rows have no Taco value. Section 5.9 compares all compressors on the
files that Taco measured.

<!-- TABLES START -->
## 4. Samples of 32 KiB: bits per byte (measured)

The best value in each row is in bold. n.m. means not measured. A class row and an overall row
need all files of the row.

### 4.1 English

| File | Bytes | gzip -9 | bzip2 -9 | xz -9e | zstd -19 | zstd --ultra -22 --long=27 | brotli -q 11 -w 24 | 7z PPMd o16 256m | paq8px -8 | taco |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| alice29.txt | 32,768 | 3.179 | 2.757 | 3.027 | 3.044 | 3.044 | 2.741 | 2.696 | 2.168 | **2.062** |
| asyoulik.txt | 32,768 | 3.317 | 2.844 | 3.155 | 3.177 | 3.177 | 2.902 | 2.817 | **2.276** | 2.423 |
| lcet10.txt | 32,768 | 3.076 | 2.724 | 2.914 | 2.943 | 2.943 | 2.395 | 2.672 | 2.079 | **1.810** |
| plrabn12.txt | 32,768 | 3.599 | 3.131 | 3.406 | 3.433 | 3.433 | 3.109 | 3.046 | 2.561 | **2.370** |
| **Class** | 131,072 | 3.293 | 2.864 | 3.126 | 3.149 | 3.149 | 2.787 | 2.808 | 2.271 | **2.166** |

### 4.2 Other languages

| File | Bytes | gzip -9 | bzip2 -9 | xz -9e | zstd -19 | zstd --ultra -22 --long=27 | brotli -q 11 -w 24 | 7z PPMd o16 256m | paq8px -8 | taco |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pg22367.txt | 32,768 | 3.310 | 2.826 | 3.133 | 3.159 | 3.159 | 3.058 | 2.797 | **2.265** | 3.109 |
| pg4650.txt | 32,768 | 3.278 | 2.867 | 3.095 | 3.136 | 3.136 | 2.999 | 2.815 | **2.284** | 2.979 |
| pg24264.txt | 32,768 | 4.143 | 3.470 | 3.908 | 3.914 | 3.914 | 3.802 | 3.545 | **2.941** | 3.844 |
| **Class** | 98,304 | 3.577 | 3.055 | 3.379 | 3.403 | 3.403 | 3.286 | 3.052 | **2.497** | 3.311 |

### 4.3 Code

| File | Bytes | gzip -9 | bzip2 -9 | xz -9e | zstd -19 | zstd --ultra -22 --long=27 | brotli -q 11 -w 24 | 7z PPMd o16 256m | paq8px -8 | taco |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fields.c | 11,150 | 2.244 | 2.180 | 2.175 | 2.166 | 2.166 | 1.949 | 2.167 | **1.456** | 2.241 |
| grammar.lsp | 3,721 | 2.653 | 2.758 | 2.778 | 2.610 | 2.610 | 2.417 | 2.748 | 1.860 | **1.739** |
| textwrap.py | 19,718 | 2.463 | 2.337 | 2.395 | 2.382 | 2.380 | 2.102 | 2.172 | 1.691 | **1.323** |
| option.rs | 32,768 | 2.115 | 2.001 | 2.030 | 2.026 | 2.026 | 1.821 | 1.961 | 1.386 | **1.189** |
| xargs.1 | 4,227 | 3.308 | 3.335 | 3.429 | 3.270 | 3.270 | 2.771 | 3.055 | 2.455 | **1.686** |
| **Class** | 71,584 | 2.329 | 2.240 | 2.275 | 2.250 | 2.249 | 2.005 | 2.157 | 1.568 | **1.448** |

### 4.4 HTML

| File | Bytes | gzip -9 | bzip2 -9 | xz -9e | zstd -19 | zstd --ultra -22 --long=27 | brotli -q 11 -w 24 | 7z PPMd o16 256m | paq8px -8 | taco |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cp.html | 24,603 | 2.593 | 2.479 | 2.488 | 2.509 | 2.509 | 2.242 | 2.337 | 1.758 | **1.553** |

### 4.5 Logs

| File | Bytes | gzip -9 | bzip2 -9 | xz -9e | zstd -19 | zstd --ultra -22 --long=27 | brotli -q 11 -w 24 | 7z PPMd o16 256m | paq8px -8 | taco |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HDFS_2k.log | 32,768 | 1.633 | 1.371 | 1.370 | 1.377 | 1.377 | 1.281 | 1.277 | **0.864** | 1.479 |
| Apache_2k.log | 32,768 | 0.468 | 0.410 | 0.409 | 0.394 | 0.394 | 0.367 | 0.356 | **0.163** | 0.477 |
| BGL_2k.log | 32,768 | 1.376 | 1.180 | 1.052 | 1.118 | 1.116 | 1.003 | 1.142 | **0.604** | 1.103 |
| Linux_2k.log | 32,768 | 0.570 | 0.628 | 0.458 | 0.474 | 0.474 | 0.439 | 0.493 | **0.250** | 0.573 |
| OpenSSH_2k.log | 32,768 | 0.796 | 0.757 | 0.661 | 0.672 | 0.672 | 0.609 | 0.738 | **0.298** | 0.825 |
| Spark_2k.log | 32,768 | 0.770 | 0.755 | 0.633 | 0.660 | 0.659 | 0.578 | 0.613 | **0.353** | 0.738 |
| **Class** | 196,608 | 0.936 | 0.850 | 0.764 | 0.783 | 0.782 | 0.713 | 0.770 | **0.422** | 0.866 |

### 4.6 JSON

| File | Bytes | gzip -9 | bzip2 -9 | xz -9e | zstd -19 | zstd --ultra -22 --long=27 | brotli -q 11 -w 24 | 7z PPMd o16 256m | paq8px -8 | taco |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| twitter.json | 32,768 | 1.355 | 1.321 | 1.185 | 1.283 | 1.282 | 1.079 | 1.458 | **0.747** | 1.323 |

### 4.7 Binary

| File | Bytes | gzip -9 | bzip2 -9 | xz -9e | zstd -19 | zstd --ultra -22 --long=27 | brotli -q 11 -w 24 | 7z PPMd o16 256m | paq8px -8 | taco |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kennedy.xls | 32,768 | 1.831 | 1.351 | 0.748 | 1.029 | 1.029 | 0.856 | 1.480 | **0.204** | 1.181 |
| ptt5 | 32,768 | 0.019 | 0.015 | 0.035 | 0.011 | 0.011 | 0.008 | 0.035 | **0.005** | 0.043 |
| sum | 32,768 | 2.840 | 2.881 | 2.125 | 2.503 | 2.503 | 2.258 | 2.573 | **1.509** | 2.562 |
| **Class** | 98,304 | 1.563 | 1.416 | 0.969 | 1.181 | 1.181 | 1.041 | 1.363 | **0.573** | 1.262 |

### 4.8 Overall

| File | Bytes | gzip -9 | bzip2 -9 | xz -9e | zstd -19 | zstd --ultra -22 --long=27 | brotli -q 11 -w 24 | 7z PPMd o16 256m | paq8px -8 | taco |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **All, not binary** | 554,939 | 2.238 | 1.996 | 2.081 | 2.101 | 2.101 | 1.915 | 1.944 | **1.452** | 1.739 |
| **All files** | 653,243 | 2.137 | 1.908 | 1.914 | 1.963 | 1.962 | 1.783 | 1.857 | **1.320** | 1.667 |

### 4.9 Taco against the others

Percentages are the change of the Taco size against the other size. A negative value means that
Taco is smaller.

| Group | Files | Bytes | Best general compressor | paq8px -8 | Taco | Taco against the best general | Taco against paq8px |
|---|---:|---:|---|---:|---:|---:|---:|
| English | 4 | 131,072 | 2.787 (brotli -q 11 -w 24) | 2.271 | 2.166 | −22.3 % | −4.6 % |
| Other languages | 3 | 98,304 | 3.052 (7z PPMd o16 256m) | 2.497 | 3.311 | +8.5 % | +32.6 % |
| Code | 5 | 71,584 | 2.005 (brotli -q 11 -w 24) | 1.568 | 1.448 | −27.8 % | −7.7 % |
| HTML | 1 | 24,603 | 2.242 (brotli -q 11 -w 24) | 1.758 | 1.553 | −30.7 % | −11.7 % |
| Logs | 6 | 196,608 | 0.713 (brotli -q 11 -w 24) | 0.422 | 0.866 | +21.5 % | +105.2 % |
| JSON | 1 | 32,768 | 1.079 (brotli -q 11 -w 24) | 0.747 | 1.323 | +22.6 % | +77.1 % |
| Binary | 3 | 98,304 | 0.969 (xz -9e) | 0.573 | 1.262 | +30.2 % | +120.4 % |
| **All, not binary** | 20 | 554,939 | 1.915 (brotli -q 11 -w 24) | 1.452 | 1.739 | −9.2 % | +19.7 % |
| **All files** | 23 | 653,243 | 1.783 (brotli -q 11 -w 24) | 1.320 | 1.667 | −6.5 % | +26.3 % |

## 5. Whole files: bits per byte (measured)

The best value in each row is in bold. n.m. means not measured. A class row and an overall row
need all files of the row.

### 5.1 English

| File | Bytes | gzip -9 | bzip2 -9 | xz -9e | zstd -19 | zstd --ultra -22 --long=27 | brotli -q 11 -w 24 | 7z PPMd o16 256m | paq8px -8 | taco |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| alice29.txt | 152,089 | 2.850 | 2.272 | 2.553 | 2.589 | 2.589 | 2.445 | 2.049 | **1.738** | 2.048 |
| asyoulik.txt | 125,179 | 3.120 | 2.529 | 2.850 | 2.885 | 2.885 | 2.730 | 2.451 | **2.012** | n.m. |
| lcet10.txt | 426,754 | 2.707 | 2.019 | 2.240 | 2.275 | 2.275 | 2.126 | 1.902 | **1.480** | n.m. |
| plrabn12.txt | 481,861 | 3.225 | 2.417 | 2.747 | 2.780 | 2.780 | 2.711 | 2.298 | **1.948** | n.m. |
| **Class** | 1,185,883 | 2.980 | 2.267 | 2.550 | 2.585 | 2.585 | 2.468 | 2.140 | **1.760** | n.m. |

### 5.2 Other languages

| File | Bytes | gzip -9 | bzip2 -9 | xz -9e | zstd -19 | zstd --ultra -22 --long=27 | brotli -q 11 -w 24 | 7z PPMd o16 256m | paq8px -8 | taco |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pg22367.txt | 145,983 | 2.994 | 2.474 | 2.705 | 2.751 | 2.751 | 2.635 | 2.228 | **1.894** | n.m. |
| pg4650.txt | 234,120 | 2.989 | 2.418 | 2.611 | 2.658 | 2.658 | 2.556 | 2.248 | **1.823** | n.m. |
| pg24264.txt | 2,663,455 | 3.453 | 2.490 | 2.683 | 2.719 | 2.719 | 2.696 | 2.215 | **1.969** | n.m. |
| **Class** | 3,043,558 | 3.395 | 2.484 | 2.678 | 2.716 | 2.716 | 2.683 | 2.218 | **1.954** | n.m. |

### 5.3 Code

| File | Bytes | gzip -9 | bzip2 -9 | xz -9e | zstd -19 | zstd --ultra -22 --long=27 | brotli -q 11 -w 24 | 7z PPMd o16 256m | paq8px -8 | taco |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fields.c | 11,150 | 2.244 | 2.180 | 2.175 | 2.166 | 2.166 | 1.949 | 2.167 | **1.456** | 2.241 |
| grammar.lsp | 3,721 | 2.653 | 2.758 | 2.778 | 2.610 | 2.610 | 2.417 | 2.748 | 1.860 | **1.739** |
| textwrap.py | 19,718 | 2.463 | 2.337 | 2.395 | 2.382 | 2.380 | 2.102 | 2.172 | 1.691 | **1.323** |
| option.rs | 81,530 | 1.702 | 1.511 | 1.577 | 1.594 | 1.594 | 1.460 | 1.381 | **1.006** | 1.130 |
| xargs.1 | 4,227 | 3.308 | 3.335 | 3.429 | 3.270 | 3.270 | 2.771 | 3.055 | 2.455 | **1.686** |
| **Class** | 120,346 | 1.963 | 1.811 | 1.869 | 1.866 | 1.866 | 1.686 | 1.684 | **1.237** | 1.303 |

### 5.4 HTML

| File | Bytes | gzip -9 | bzip2 -9 | xz -9e | zstd -19 | zstd --ultra -22 --long=27 | brotli -q 11 -w 24 | 7z PPMd o16 256m | paq8px -8 | taco |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cp.html | 24,603 | 2.593 | 2.479 | 2.488 | 2.509 | 2.509 | 2.242 | 2.337 | 1.758 | **1.553** |

### 5.5 Logs

| File | Bytes | gzip -9 | bzip2 -9 | xz -9e | zstd -19 | zstd --ultra -22 --long=27 | brotli -q 11 -w 24 | 7z PPMd o16 256m | paq8px -8 | taco |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HDFS_2k.log | 287,848 | 1.494 | 1.171 | 1.177 | 1.191 | 1.191 | 1.140 | 1.098 | **0.717** | 1.441 |
| Apache_2k.log | 171,239 | 0.429 | 0.333 | 0.315 | 0.337 | 0.337 | 0.313 | 0.281 | **0.116** | 0.470 |
| BGL_2k.log | 317,150 | 1.433 | 1.130 | 0.992 | 1.101 | 1.101 | 1.010 | 1.038 | **0.542** | 1.163 |
| Linux_2k.log | 216,485 | 0.541 | 0.526 | 0.370 | 0.420 | 0.420 | 0.379 | 0.419 | **0.193** | 0.548 |
| OpenSSH_2k.log | 225,216 | 0.553 | 0.427 | 0.346 | 0.386 | 0.386 | 0.359 | 0.382 | **0.110** | 0.575 |
| Spark_2k.log | 196,268 | 0.529 | 0.413 | 0.369 | 0.406 | 0.406 | 0.379 | 0.340 | **0.140** | 0.617 |
| **Class** | 1,414,206 | 0.922 | 0.738 | 0.663 | 0.712 | 0.712 | 0.664 | 0.662 | **0.348** | 0.872 |

### 5.6 JSON

| File | Bytes | gzip -9 | bzip2 -9 | xz -9e | zstd -19 | zstd --ultra -22 --long=27 | brotli -q 11 -w 24 | 7z PPMd o16 256m | paq8px -8 | taco |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| twitter.json | 631,514 | 0.621 | 0.434 | 0.432 | 0.478 | 0.474 | 0.422 | 0.439 | **0.250** | n.m. |

### 5.7 Binary

| File | Bytes | gzip -9 | bzip2 -9 | xz -9e | zstd -19 | zstd --ultra -22 --long=27 | brotli -q 11 -w 24 | 7z PPMd o16 256m | paq8px -8 | taco |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kennedy.xls | 1,029,744 | 1.629 | 1.012 | 0.403 | 0.542 | 0.542 | 0.478 | 1.416 | **0.062** | 0.860 |
| ptt5 | 513,216 | 0.816 | 0.776 | 0.621 | 0.677 | 0.677 | 0.638 | 0.757 | **0.264** | 0.722 |
| sum | 38,240 | 2.671 | 2.701 | 1.987 | 2.350 | 2.350 | 2.122 | 2.515 | **1.390** | 2.605 |
| **Class** | 1,581,200 | 1.391 | 0.976 | 0.512 | 0.630 | 0.629 | 0.570 | 1.229 | **0.160** | 0.858 |

### 5.8 Overall

| File | Bytes | gzip -9 | bzip2 -9 | xz -9e | zstd -19 | zstd --ultra -22 --long=27 | brotli -q 11 -w 24 | 7z PPMd o16 256m | paq8px -8 | taco |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **All, not binary** | 6,420,110 | 2.471 | 1.845 | 1.974 | 2.014 | 2.013 | 1.956 | 1.677 | **1.383** | n.m. |
| **All files** | 8,001,310 | 2.257 | 1.673 | 1.685 | 1.740 | 1.740 | 1.682 | 1.588 | **1.141** | n.m. |

### 5.9 Taco against the others

Percentages are the change of the Taco size against the other size. A negative value means that
Taco is smaller.

| Group | Files | Bytes | Best general compressor | paq8px -8 | Taco | Taco against the best general | Taco against paq8px |
|---|---:|---:|---|---:|---:|---:|---:|
| English (only alice29.txt) | 1 | 152,089 | 2.049 (7z PPMd o16 256m) | 1.738 | 2.048 | −0.1 % | +17.9 % |
| Code | 5 | 120,346 | 1.684 (7z PPMd o16 256m) | 1.237 | 1.303 | −22.7 % | +5.3 % |
| HTML | 1 | 24,603 | 2.242 (brotli -q 11 -w 24) | 1.758 | 1.553 | −30.7 % | −11.7 % |
| Logs | 6 | 1,414,206 | 0.662 (7z PPMd o16 256m) | 0.348 | 0.872 | +31.7 % | +150.4 % |
| Binary | 3 | 1,581,200 | 0.512 (xz -9e) | 0.160 | 0.858 | +67.4 % | +436.6 % |
| **All that Taco measured, not binary** | 13 | 1,711,244 | 0.882 (7z PPMd o16 256m) | 0.554 | 1.017 | +15.3 % | +83.4 % |
| **All files that Taco measured** | 16 | 3,292,444 | 0.736 (xz -9e) | 0.365 | 0.940 | +27.8 % | +157.6 % |

## 6. Speed and memory (measured)

A whole file of 32 KiB or less is the same input as its sample. Its record is reused from the
samples, and the speed tables of the set `whole` leave it out.

### 6.1 Set `samples`

| Compressor | Files | Bytes | Compress (KB/s) | Decompress (KB/s) | Peak memory, compress (MiB) | Peak memory, decompress (MiB) | Load average (1 min) |
|---|---:|---:|---:|---:|---:|---:|---|
| gzip -9 | 23 | 653,243 | 3,458 | 5,143 | 2 | 2 | 11.7 to 11.7 |
| bzip2 -9 | 23 | 653,243 | 3,163 | 4,511 | 2 | 2 | 11.7 to 11.7 |
| xz -9e | 23 | 653,243 | 598 | 4,128 | 33 | 2 | 11.5 to 11.7 |
| zstd -19 | 23 | 653,243 | 727 | 3,279 | 3 | 3 | 11.4 to 11.5 |
| zstd --ultra -22 --long=27 | 23 | 653,243 | 611 | 3,061 | 3 | 3 | 11.4 to 11.6 |
| brotli -q 11 -w 24 | 23 | 653,243 | 412 | 4,724 | 5 | 2 | 11.6 to 11.6 |
| 7z PPMd o16 256m | 23 | 653,243 | 1,925 | 1,974 | 6 | 6 | 11.6 to 11.6 |
| paq8px -8 | 23 | 653,243 | 1.16 | 1.22 | 2,019 | 2,019 | 11.6 to 46.3 |
| taco | 23 | 653,243 | 1.09 | 0.325 | 246 | 246 | 1.4 to 4.1 |

### 6.2 Set `whole`

| Compressor | Files | Bytes | Compress (KB/s) | Decompress (KB/s) | Peak memory, compress (MiB) | Peak memory, decompress (MiB) | Load average (1 min) |
|---|---:|---:|---:|---:|---:|---:|---|
| gzip -9 | 23 | 8,001,310 | 4,645 | 43,732 | 2 | 2 | 19.4 to 19.5 |
| bzip2 -9 | 23 | 8,001,310 | 5,635 | 7,332 | 8 | 5 | 19.5 to 20.0 |
| xz -9e | 23 | 8,001,310 | 671 | 23,098 | 89 | 5 | 20.0 to 21.0 |
| zstd -19 | 23 | 8,001,310 | 662 | 27,324 | 56 | 6 | 20.6 to 22.0 |
| zstd --ultra -22 --long=27 | 23 | 8,001,310 | 515 | 34,962 | 72 | 6 | 21.9 to 22.0 |
| brotli -q 11 -w 24 | 23 | 8,001,310 | 237 | 38,033 | 56 | 6 | 20.4 to 22.0 |
| 7z PPMd o16 256m | 23 | 8,001,310 | 2,204 | 2,028 | 61 | 60 | 19.5 to 20.1 |
| paq8px -8 | 23 | 8,001,310 | 5.77 | 6.23 | 2,276 | 2,276 | 2.2 to 6.2 |
| taco | 11 | 3,229,025 | 2.1 | 1.34 | 279 | 481 | 0.8 to 23.8 |

### 6.3 paq8px and Taco for each class, set `samples`

| Class | Compressor | Files | Bytes | Compress (KB/s) | Decompress (KB/s) |
|---|---|---:|---:|---:|---:|
| English | paq8px -8 | 4 | 131,072 | 0.835 | 0.851 |
| English | taco | 4 | 131,072 | 1.23 | 0.316 |
| Other languages | paq8px -8 | 3 | 98,304 | 1.43 | 1.72 |
| Other languages | taco | 3 | 98,304 | 0.728 | 0.188 |
| Code | paq8px -8 | 5 | 71,584 | 1.19 | 1.19 |
| Code | taco | 5 | 71,584 | 0.813 | 0.234 |
| HTML | paq8px -8 | 1 | 24,603 | 0.791 | 0.787 |
| HTML | taco | 1 | 24,603 | 0.758 | 0.185 |
| Logs | paq8px -8 | 6 | 196,608 | 1.74 | 1.74 |
| Logs | taco | 6 | 196,608 | 1.11 | 0.443 |
| JSON | paq8px -8 | 1 | 32,768 | 1.61 | 1.97 |
| JSON | taco | 1 | 32,768 | 0.78 | 0.176 |
| Binary | paq8px -8 | 3 | 98,304 | 0.874 | 0.948 |
| Binary | taco | 3 | 98,304 | 5.16 | 97.8 |

### 6.4 paq8px and Taco for each class, set `whole`

| Class | Compressor | Files | Bytes | Compress (KB/s) | Decompress (KB/s) |
|---|---|---:|---:|---:|---:|
| English | paq8px -8 | 4 | 1,185,883 | 6.1 | 5.87 |
| English | taco | 1 | 152,089 | 2.03 | 0.887 |
| Other languages | paq8px -8 | 3 | 3,043,558 | 5.51 | 6.37 |
| Code | paq8px -8 | 5 | 120,346 | 4.22 | 4.3 |
| Code | taco | 1 | 81,530 | 1.28 | 0.409 |
| HTML | paq8px -8 | 1 | 24,603 | 3.57 | 3.55 |
| Logs | paq8px -8 | 6 | 1,414,206 | 5.96 | 5.98 |
| Logs | taco | 6 | 1,414,206 | 1.09 | 0.696 |
| JSON | paq8px -8 | 1 | 631,514 | 6.29 | 7.29 |
| Binary | paq8px -8 | 3 | 1,581,200 | 5.91 | 6.41 |
| Binary | taco | 3 | 1,581,200 | 16.1 | 1,706 |
<!-- TABLES END -->

## 7. Discussion

### 7.1 Where Taco wins (measured)

- **English, code and HTML samples.** Taco is smaller than every general compressor on 9 of
  these 10 samples, and smaller than paq8px on 8 of 10. For each class (section 4.9): English
  −22.3 % against brotli and −4.6 % against paq8px; code −27.8 % and −7.7 %; HTML −30.7 % and
  −11.7 %. The largest gain is on xargs.1 (troff source): 1.686 bits per byte against 2.455 for
  paq8px.
- **The 5 whole code files** (120,346 bytes): Taco is 22.7 % smaller than 7z PPMd and 5.3 %
  larger than paq8px. On option.rs alone (81,530 bytes), paq8px is 11.0 % smaller than Taco.
- **All samples together:** Taco is 6.5 % smaller than brotli, the best general compressor
  (1.667 against 1.783 bits per byte).

### 7.2 Where Taco loses (measured)

- **Tabs.** asyoulik.txt and fields.c have many tab characters. GPT-2 gives tab tokens a very
  low probability (phase 0 note, §5). On fields.c the coded block is larger than the zstd frame,
  so Taco stores zstd (ADR 0011 §4). paq8px is smaller on both files.
- **Other languages** (German, French, Chinese samples): paq8px is 24.6 % smaller than Taco, and
  7z PPMd is 7.8 % smaller. GPT-2 small saw mostly English text in training.
- **Logs:** paq8px is 2.1 times smaller on the samples and 2.5 times smaller on the whole logs.
  brotli and xz are also smaller than Taco on every log, in both sets. On the Apache and Linux
  samples, the zstd frame is smaller than the coded block (unfold takes only 0.3 s), so Taco
  is near zstd there.
- **JSON** (twitter.json sample): paq8px is 43.5 % smaller than Taco, and brotli is 18.4 %
  smaller.
- **Binary:** Taco stores these blocks as zstd frames of 16 KiB. On the whole files, xz -9e is
  40 % smaller than Taco (0.512 against 0.858 bits per byte) and paq8px is 5.4 times smaller.
- **Whole files.** The other compressors use the whole file as context. Taco starts the model
  context again at each block of 16 KiB and after 1,024 tokens (plan §7.7). Thus Taco does not
  give fewer bits per byte for a longer file:

| File | Taco, sample | Taco, whole | 7z PPMd, sample | 7z PPMd, whole | paq8px, sample | paq8px, whole |
|---|---:|---:|---:|---:|---:|---:|
| alice29.txt | 2.062 | 2.048 | 2.696 | 2.049 | 2.168 | 1.738 |
| option.rs | 1.189 | 1.130 | 1.961 | 1.381 | 1.386 | 1.006 |
| Logs (6 files) | 0.866 | 0.872 | 0.770 | 0.662 | 0.422 | 0.348 |

All values are bits per byte (measured). A sample is 32 KiB of the file, so the sample and the
whole file are not the same text. The trend is clear: 7z PPMd and paq8px become 14 % to 30 %
smaller with the whole file. Taco changes by −5.0 % to +0.7 %.

### 7.3 Speed and memory (measured)

| Path | Input | Threads | Compress | Decompress | Load average |
|---|---|---:|---:|---:|---|
| Taco, fast CPU engine | 23 samples, all | 2 for each process, 2 processes | 1.09 KB/s for each process | 0.325 KB/s for each process | 1.4 to 4.1 |
| Taco, fast CPU engine | whole alice29.txt (47,030 tokens) | 4 | 2.03 KB/s (627 tokens/s) | 0.89 KB/s (274 tokens/s) | 0.8 to 3.2 |
| Taco, fast CPU engine | 6 whole logs | 4 | 1.09 KB/s | 0.70 KB/s | 1.8 to 23.8 |
| Taco, reference engine | 2 samples (draft of this note) | 2 for each process | 0.064 KB/s | 0.065 KB/s | 37 to 44 |
| paq8px -8 | 23 whole files | 1 | 5.77 KB/s | 6.23 KB/s | 2.2 to 6.2 |
| xz -9e | 23 whole files | 1 | 671 KB/s | 23,098 KB/s | 20 to 21 |
| 7z PPMd | 23 whole files | 1 | 2,204 KB/s | 2,028 KB/s | 19.5 to 20.1 |

- **Taco against paq8px.** On whole text, Taco folds about 3 times slower and unfolds about 7
  times slower than paq8px, with 4 times the processors (alice29.txt: fold 75 s, unfold 171 s;
  paq8px 24 s and 25 s, measured).
- **Unfold is slower than fold.** Fold knows all tokens of a block and uses the prefill path.
  Unfold must run one model step for each token. On the samples with coded blocks, fold gives
  250 to 460 tokens/s and unfold 64 to 143 tokens/s for each process of 2 threads.
- **zstd blocks are fast.** A block that Taco stores as zstd decodes without the model. For
  binary files, unfold gives 1,706 KB/s. The encoder also stops the model early on these blocks
  (ADR 0011 §4.5), so fold gives 16 KB/s.
- **The fast engine against the reference engine.** On alice29.txt (sample), the fast engine
  folds in 30.6 s and the reference engine in 525 s. Unfold: 111 s and 520 s. The load was very
  different (1.4 against 44), so the ratio (17 times and 4.7 times) is not exact.
- **Load.** A run of another agent raised the load average to 14 to 24 during the last 3 logs
  (Linux, OpenSSH, Spark). BGL unfold took 945 s at a load of 7.4. Thus the speeds for these logs
  are possibly too low.
- **Memory.** Taco used 245 to 481 MiB: the model file (127 MB) and the state of the threads.
  paq8px -8 used up to 2,276 MiB. The general compressors used 89 MiB or less.

### 7.4 The GPU backends (other note, measured there)

`docs/research/2026-10-18-gpu-backend.md` gives these numbers for `crates/hardshell-cuda` on the
RTX 3060 of devbox. The weights are the same as on the CPU, bit for bit (7 of 7 windows).

- Prefill (the fold path): 12,012 to 22,355 tokens/s.
- Decode (the unfold path): 18,565 tokens/s at short context (batch 128), and 7,229 tokens/s
  over a full window of 1,024 tokens (batch 224).

These values are 19 to 36 times the fold speed and 26 times the unfold speed of the CPU on
whole alice29.txt (est., different inputs). At 3.2 bytes for each token (alice29.txt), 7,229
tokens/s is about 23 KB/s (est.). The GPU path is not in `taco fold` and `taco unfold` of commit
`71a1192`, so this note does not measure it.

### 7.5 The logs profile (simulated in another note, not measured here)

The match model and the mixer (ADR 0010) are not in `taco fold` of commit `71a1192`. Work to add
them was in progress on 2026-10-18; `docs/research/2026-10-19-logs-profile.md` (not complete
when this note was written) is to give the real sizes. `docs/research/2026-10-18-logs.md` §5 gives the simulated
size with 1 MiB blocks: the exact table cost of the integer rules, and an estimate for the
container. The table below puts these values next to the measurements of this note. The
general compressors in the logs note ran on the Mac; this table uses the devbox values.

| File | Taco, measured (GPT-2 alone, 16 KiB blocks) | Logs profile, simulated (1 MiB blocks, est.) | xz -9e | 7z PPMd | paq8px -8 | Logs profile against paq8px |
|---|---:|---:|---:|---:|---:|---:|
| HDFS_2k.log | 1.441 | 1.155 | 1.177 | 1.098 | 0.717 | +61 % |
| Apache_2k.log | 0.470 | 0.221 | 0.315 | 0.281 | 0.116 | +90 % |
| BGL_2k.log | 1.163 | 0.900 | 0.992 | 1.038 | 0.542 | +66 % |
| Linux_2k.log | 0.548 | 0.280 | 0.370 | 0.419 | 0.193 | +45 % |
| OpenSSH_2k.log | 0.575 | 0.267 | 0.346 | 0.382 | 0.110 | +142 % |
| Spark_2k.log | 0.617 | 0.366 | 0.369 | 0.340 | 0.140 | +161 % |
| **6 logs** | 0.872 | 0.600 | 0.663 | 0.662 | **0.348** | +72 % |

All values are bits per byte. The xz, 7z PPMd and paq8px values are measured on devbox (section
5.5).

- The logs profile is smaller than xz on 6 of 6 logs and smaller than 7z PPMd on 4 of 6 logs
  (est.). For the 6 logs together, it is 9.4 % smaller than 7z PPMd and 9.5 % smaller than xz.
- paq8px is 1.4 to 2.6 times smaller than the logs profile on every log. Thus "Taco beats paq8px
  on logs" is not true with the logs profile of ADR 0010. A claim against xz and zstd is
  possible, with large blocks.
- paq8px is slow. It decompresses the whole logs at 5.98 KB/s, and xz decompresses the whole
  files at 23,098 KB/s (measured). A logs profile on a GPU can possibly give a size between xz
  and paq8px at a higher speed than paq8px (est., not measured).

## 8. Published reference numbers (not measured here)

WARNING: These numbers are not measurements of this note. They come from the authors' pages
(read again on 2026-10-18). The files, the machines and the method are different. Do not put
them in the same table as sections 4 to 6.

They are for enwik8 (100,000,000 bytes) and enwik9 (1,000,000,000 bytes) of English Wikipedia.

| Compressor | enwik8 (bytes) | enwik8 (bits per byte) | enwik9 (bytes) | enwik9 (bits per byte) | Notes (published) | Source |
|---|---:|---:|---:|---:|---|---|
| cmix v21 | 14,623,723 | 1.170 | 107,963,380 | 0.864 | 65,155 s for enwik8 and 622,950 s for enwik9; 20,987,016 KiB and 30,950,696 KiB of memory (20.0 GiB and 29.5 GiB) | http://www.byronknoll.com/cmix.html |
| nncp (2023-10-21) | 14,915,298 | 1.19 | 106,632,363 | 0.853 | enwik9 total with the zipped program (628,955 bytes): 107,261,318 bytes. The page lists v3.3 (2024-06-05) as the current download; the result table is for 2023-10-21. | https://bellard.org/nncp/ |
| ts_zip | 13,825,741 | 1.106 | 135,443,237 | 1.084 | RWKV 169M v4, 8-bit weights, BF16 arithmetic. "Up to 1 MB/s on a RTX 4090". Also alice29.txt: 21,713 bytes, 1.142 bits per byte (whole file). | https://bellard.org/ts_zip/ |
| xz (on the ts_zip page) | 24,865,244 | 1.989 | 213,370,900 | 1.707 | Also alice29.txt: 48,492 bytes, 2.551 bits per byte | https://bellard.org/ts_zip/ |

The bits per byte of cmix are calculated from the published sizes (8 × bytes / file size). The
other bits per byte values are as published. More results: the Large Text Compression
Benchmark, http://mattmahoney.net/dc/text.html.

One comparison is possible with care: whole alice29.txt. Taco gives 2.048 bits per byte
(measured, section 5.1). ts_zip gives 1.142 and xz 2.551 (published). Our xz -9e gives 2.553
(measured), so the xz values agree. ts_zip is 44 % smaller than Taco on this file. ts_zip
uses a larger model (RWKV, 169M parameters; GPT-2 small has 124M). Its page also says that the
result "does not depend on the exact GPU or CPU model nor on the number of configured threads"
(published). Thus Taco is not the only compressor with a
language model that decodes on other machines.

## 9. Limits

- **Load.** Other agents used devbox. The fast compressors ran at a load average of 11 to 22,
  and paq8px on the samples at up to 46. Taco and paq8px on the whole files ran at 0.8 to
  23.8. A loaded machine gives slower times, possibly by a factor of 2 or more for paq8px on
  the samples (est.). The sizes do not depend on the load.
- **Taco on the whole files: a reduced set** (section 3). 7 files (4,708,866 bytes) are not
  measured.
- **One build of Taco.** Commit `71a1192`: GPT-2 alone, the fast CPU engine, no GPU. The logs
  profile and the GPU backends are not in this build (sections 7.4 and 7.5).
- **The fast compressors are timed on small files.** The process start is a large part of
  their time, so their speed on the samples is lower than on large files. Use the whole-file
  speeds for these compressors.
- **One machine (x86) only.** Taco gives the same bytes on the Mac (phase 2 gate), so the sizes
  are the same on the Mac. The speeds are not.
- **One model.** The Taco sizes are for GPT-2 small (version 1). Other models are not in this
  test.
- **cmix** did not run: it needs 32 GB of RAM (published, cmix README), and devbox has 12 GB.

## 10. How to repeat the measurement

WARNING: A later build of `taco` can have a `--profile` option with the default `auto`. Then
the sizes are different from this note. Build commit `71a1192` to repeat this note.

1. On devbox: `sudo apt-get install -y zstd brotli p7zip-full` (through `scripts/dx`).
2. Build paq8px: `git clone --depth 1 --branch v217 https://github.com/hxim/paq8px.git
   ~/bench/src/paq8px`, then `cd ~/bench/src/paq8px/build && ./build-linux-with-cmake.sh`.
3. Build Taco from the commit: `git archive 71a1192 | ssh devbox "mkdir -p ~/bench/taco-src
   && tar -x -C ~/bench/taco-src"`, then `cargo build --release -p taco-cli` in that directory
   with `CARGO_TARGET_DIR=~/bench/target`.
4. Run `scripts/dx bench "tmux new-session -d -s bench 'bash tools/bench/run_devbox.sh'"`.
   The log is `~/bench/run.log`. A second start continues where the first one stopped.
5. Copy the results files to `tools/data/bench/`. Then run
   `python3 -I tools/bench/run.py table tools/data/bench/results.json
   tools/data/bench/results-paq8px-whole.json` to make the tables of sections 4 to 6. (In a
   new run, `run_devbox.sh` writes all records to `results.json`, so give only that file.)
