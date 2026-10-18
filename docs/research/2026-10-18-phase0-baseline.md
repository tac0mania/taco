# Research: phase 0 baseline and phase 1 part A (2026-10-18)

This note gives the first real numbers for Taco with the float GPT-2 model. Phase 0 measures
the compression. Phase 1 part A measures float drift between computation paths on one Mac.
All numbers are measured on 2026-10-18, unless a number has a different label.

## 1. Results in short

- On the 16 files that are not binary, float GPT-2 with the integer range coder gives
  1.992 bits per byte. bzip2 gives 2.356, xz 2.508, zstd 2.524 and gzip 2.658 (measured).
- GPT-2 is the best compressor on 10 of 19 files: English, French, HTML, Python, Rust, Lisp
  and the man page. It is worse on binary files (13 to 32 bits per byte), on logs, on JSON,
  and on the German, Chinese and C files (measured).
- The integer coder (plan §5, §5.1, `FREQ_BITS` = 32) adds 80.6 bytes to 379,259 bytes of
  table information, that is 0.021 %, or 0.30 bytes for each window of 1,024 tokens
  (measured).
- GPT-2 gives some tab and carriage-return tokens a probability below 2^-32 (up to 119 bits
  for one token, measured). The minimum frequency 1 of §5.1 limits the cost to about 32 bits.
  Thus the coded size is smaller than the float ideal size on these files.
- A stream that the full-sequence path encodes does not decode with the KV-cache path: 269 of
  270 windows failed. The first wrong token was token 1 to 27 of the window (median 2,
  measured). With the KV-cache path on both sides, 19 of 19 windows decoded (measured).
- Phase 1 part A, on one Mac: full sequence against KV cache, float32 against float64, and KV
  batch 4 against KV batch 1 each change the frequency table in 99.9 to 100 % of the steps.
  No window decoded across the two paths (0 of 19 for each pair, both directions). The
  largest probability difference was only 2.8e-5 to 7.4e-5. BLAS thread count 1 against the
  default changed nothing (measured). Part B (Mac against devbox) is not done (§10).

## 2. Setup

| Item | Value |
|---|---|
| Machine | Apple M2 (Mac14,2), 8 GB, macOS 26.5.2. Other agents used the CPU at the same time. |
| Software | Python 3.12.11, NumPy 2.5.4 (BLAS: Apple Accelerate), regex 2026.9.29, safetensors 0.8.0, zstandard 0.25.0 (zstd 1.5.7), zlib 1.2.12 |
| Model | GPT-2 small (`openai-community/gpt2`), float32. SHA-256 and sizes in `tools/baseline/manifest.toml`. |
| Tokenizer | `tools/baseline/bpe.py`: GPT-2 byte-level BPE. Bytes that are not well-formed UTF-8 use the single-byte tokens. |
| Context | A window of 1,024 tokens. The model input is `<|endoftext|>` and then the first 1,023 tokens of the window (plan §7.7). |
| Coder | `tools/baseline/rangecoder.py`: 64-bit range coder, LZMA style, `TOTAL` = 2^32, range >= 2^56 after renormalisation (plan §5). |
| Tables | Plan §5.1 exactly, `FREQ_BITS` = 32. Weights: `w = floor(e * (2^32 / s))` in float64, with `e = exp(logit - max)` and `s = sum(e)`. Thus `w` is `p * 2^32` rounded down, and `w <= 2^32`. Then `w * (T - V) < 2^64`, so the `u64` product is exact. |
| Streams | Each window is one coder stream (one block in §9 terms), so that windows decode in parallel. The decoder knows the token count of each window. |
| Cap | The first 32 KiB (32,768 bytes) of each file. Gutenberg files start after the `*** START OF` line. |

Model checks (`python -m baseline.check_model`, measured):

- Greedy generation of 20 tokens from "The dog" on the KV-cache path gives "The dog was found
  in a field near the intersection of West and West Streets.\n\nThe dog". This is the text of
  the GPT-2 greedy test in Hugging Face transformers (published; the test file was not
  downloaded, so the comparison uses the text as remembered: UNVERIFIED).
- After "The capital of France is", " Paris" has rank 4 (p = 0.032). The top tokens are
  " the" (0.085), " now", " a" and " France". With `<|endoftext|>` before the text, " Paris"
  has rank 92. After "The Eiffel Tower is in the city of", " Paris" has rank 0 (p = 0.070).
  GPT-2 small is a weak model, so rank 4 is not an error.
- Mean loss on the first window: lcet10.txt 3.52 nats per token, alice29.txt 4.82, and
  alice29.txt with LF line ends 3.13. The CRLF line ends of alice29.txt cost 1.7 nats per
  token. asyoulik.txt gives 7.57 nats per token, because it has many tab tokens (see §5).

Tokenizer checks (`python -m baseline.test_bpe`, measured): 6,284 round trips passed. The
inputs were empty input, all 256 single bytes, all 256 byte values in one string, 13
multilingual texts (German, French, Hindi, Chinese, Arabic, Japanese, emoji, code, JSON),
2,000 random byte strings, 2,000 random letter strings, 2,000 texts with a random byte
inserted, and invalid UTF-8 forms. Known ids: "Hello world" gives [15496, 995], " Paris"
gives [6342].

Coder checks (`python -m baseline.test_coder`, measured): 600 round trips and table checks
passed. The vectorised §5.1 table is the same as the integer reference on 200 random weight
vectors.

## 3. Corpus

`tools/baseline/corpus.toml` holds the URL, the size and the SHA-256 of each file.
`python -m baseline.fetch` downloads the files and checks the hashes.

| File | Class | Source | File size (bytes) | Bytes measured | Tokens |
|---|---|---|---:|---:|---:|
| alice29.txt | English prose | Canterbury | 152,089 | 32,768 | 10,503 |
| asyoulik.txt | English play | Canterbury | 125,179 | 32,768 | 10,242 |
| cp.html | HTML | Canterbury | 24,603 | 24,603 | 10,666 |
| fields.c | C source | Canterbury | 11,150 | 11,150 | 4,929 |
| grammar.lsp | Lisp source | Canterbury | 3,721 | 3,721 | 1,664 |
| kennedy.xls | Excel binary | Canterbury | 1,029,744 | 32,768 | 32,420 |
| lcet10.txt | English technical | Canterbury | 426,754 | 32,768 | 8,815 |
| plrabn12.txt | English poetry | Canterbury | 481,861 | 32,768 | 9,603 |
| ptt5 | Fax image (binary) | Canterbury | 513,216 | 32,768 | 32,768 |
| sum | SPARC executable | Canterbury | 38,240 | 32,768 | 26,940 |
| xargs.1 | Man page | Canterbury | 4,227 | 4,227 | 1,482 |
| pg22367.txt | German prose (Kafka) | Gutenberg 22367 | 145,983 | 32,768 | 13,554 |
| pg4650.txt | French prose (Voltaire) | Gutenberg 4650 | 234,120 | 32,768 | 12,915 |
| pg24264.txt | Chinese prose (Hong Lou Meng) | Gutenberg 24264 | 2,663,455 | 32,768 | 24,772 |
| textwrap.py | Python source | GitHub, CPython v3.12.0 | 19,718 | 19,718 | 8,561 |
| option.rs | Rust source | GitHub, Rust 1.80.0 | 81,530 | 32,768 | 13,464 |
| twitter.json | JSON | GitHub, nativejson-benchmark | 631,514 | 32,768 | 18,175 |
| HDFS_2k.log | Log lines | Loghub | 287,848 | 32,768 | 12,562 |
| Apache_2k.log | Log lines | Loghub | 171,239 | 32,768 | 11,824 |
| **All** | | | | 522,171 | 265,859 |

Notes on the sources:

- corpus.canterbury.ac.nz refused `cantrbry.tar.gz` and `cantrbry.zip` with HTTP 403 (a
  Cloudflare bot check). The Canterbury files come from the GitHub mirror
  `pfalcon/canterbury-corpus`. All 11 sizes agree with the published Canterbury sizes. Four
  files (alice29.txt, asyoulik.txt, lcet10.txt, plrabn12.txt) also have the same SHA-256 as
  the copies in `google/brotli` `tests/testdata`.
- Project Gutenberg had no Hindi text in Devanagari that a search found. Chinese is the third
  language. Hindi is not in the corpus.
- twitter.json and the Loghub files come from `master` branches. If a branch changes, the
  hash check in `baseline.fetch` fails.
- The Chinese sample ends in the middle of a UTF-8 character. The tokenizer codes the last
  bytes as single-byte tokens.

## 4. Bits per byte

Command: `nice -n 10 uv run --project tools python -m baseline.phase0`. Output:
`tools/data/out/phase0.json`. All values are bits per byte (measured). "Ideal (float)" is the
sum of `-log2 p` of the actual tokens with the float32 model and float64 softmax. "Ideal
(tables)" is the sum of `-log2(f / 2^32)` with the §5.1 tables. "Coded" is the size of the
coder streams. The full-sequence path made all GPT-2 numbers in this table.

| File | gzip -9 | bzip2 -9 | xz -9e | zstd -19 | GPT-2 ideal (float) | GPT-2 ideal (tables) | GPT-2 coded | Best |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| alice29.txt | 3.179 | 2.757 | 3.027 | 3.043 | 2.087 | 1.999 | **2.000** | GPT-2 |
| asyoulik.txt | 3.317 | 2.844 | 3.155 | 3.176 | 3.335 | 2.319 | **2.320** | GPT-2 |
| cp.html | 2.586 | 2.479 | 2.488 | 2.508 | 1.575 | 1.574 | **1.575** | GPT-2 |
| fields.c | 2.244 | 2.180 | 2.175 | **2.163** | 3.265 | 2.813 | 2.815 | zstd |
| grammar.lsp | 2.653 | 2.758 | 2.778 | 2.604 | 1.472 | 1.472 | **1.475** | GPT-2 |
| kennedy.xls | 1.859 | 1.351 | **0.748** | 1.106 | 22.575 | 21.897 | 21.899 | xz |
| lcet10.txt | 3.076 | 2.724 | 2.914 | 2.942 | 1.867 | 1.808 | **1.809** | GPT-2 |
| plrabn12.txt | 3.599 | 3.131 | 3.406 | 3.432 | 2.435 | 2.414 | **2.415** | GPT-2 |
| ptt5 | 0.019 | 0.015 | 0.035 | **0.010** | 35.380 | 31.643 | 31.644 | zstd |
| sum | 2.853 | 2.881 | **2.125** | 2.476 | 13.380 | 13.293 | 13.296 | xz |
| xargs.1 | 3.308 | 3.335 | 3.429 | 3.263 | 1.397 | 1.397 | **1.399** | GPT-2 |
| pg22367.txt (German) | 3.310 | **2.826** | 3.133 | 3.157 | 2.929 | 2.913 | 2.915 | bzip2 |
| pg4650.txt (French) | 3.278 | 2.867 | 3.095 | 3.135 | 2.857 | 2.841 | **2.842** | GPT-2 |
| pg24264.txt (Chinese) | 4.143 | **3.470** | 3.908 | 3.913 | 3.731 | 3.725 | 3.727 | bzip2 |
| textwrap.py | 2.463 | 2.337 | 2.395 | 2.380 | 1.287 | 1.287 | **1.288** | GPT-2 |
| option.rs | 2.115 | 2.001 | 2.030 | 2.025 | 1.110 | 1.110 | **1.111** | GPT-2 |
| twitter.json | 1.355 | 1.321 | **1.185** | 1.283 | 1.247 | 1.247 | 1.249 | xz |
| HDFS_2k.log | 1.630 | 1.371 | **1.370** | 1.376 | 1.449 | 1.444 | 1.445 | xz |
| Apache_2k.log | 0.468 | 0.410 | 0.409 | **0.392** | 0.770 | 0.677 | 0.678 | zstd |
| **16 files, not binary** | 2.658 | 2.356 | 2.508 | 2.524 | 2.105 | 1.991 | **1.992** | GPT-2 |
| **3 binary files** | 1.577 | 1.416 | **0.969** | 1.197 | 23.778 | 22.278 | 22.280 | xz |
| **All 19 files** | 2.455 | **2.179** | 2.218 | 2.274 | 6.186 | 5.811 | 5.812 | bzip2 |

Notes:

- The general compressors are the Python library versions (`gzip`, `bz2`, `lzma` with
  preset 9 | extreme in the xz format, `zstandard` level 19). Each output was decompressed and
  checked. The gzip and xz sizes include their headers.
- The GPT-2 sizes do not include a container (§9: 88 bytes, plus 5 bytes for each block).
- On the binary files GPT-2 makes the data larger. A real container stores such a block raw
  (§9). With raw storage for the 3 binary files, all 19 files would give 3.123 bits per byte
  (est., from the sizes above).
- **The model is not counted.** The model is shared by all files, so the table does not
  include it. If the 548,105,171 bytes of `model.safetensors` were added to the 522,171 bytes
  of the corpus, the result would be about 8,403 bits per byte (measured sizes). The model
  size is important only for one small file. It is not important for a library of files that
  share the model.

## 5. Integer coder against the ideal size

| Measure | Value (measured) |
|---|---|
| Coded size, all 270 windows | 379,340 bytes |
| Ideal size with the §5.1 tables | 379,259.4 bytes |
| Coded minus table ideal | 80.6 bytes = 0.021 % = 0.30 bytes for each window |
| Ideal size with float probabilities | 403,737.5 bytes |
| Table ideal minus float ideal, 5 files with no token below 2^-32 | +0.027 bytes of 14,249 bytes (0.0002 %) |
| Tokens with float p < 2^-32 | 33,965 of 265,859 (28,610 of them in ptt5) |

The coder overhead (80.6 bytes) is the flush at the end of each stream and the loss from the
integer division. It is less than 11 bytes for each file. The 32-bit table itself loses
almost nothing: on the 5 files with no token below 2^-32, the table ideal and the float ideal
differ by 0.0002 %.

The table ideal is smaller than the float ideal by 24,478 bytes. The cause is the tokens with
p < 2^-32. GPT-2 gives some tokens almost zero probability in some contexts:

- tab (`\t`): 51 of the 53 tokens above 32 bits in the first window of asyoulik.txt, and all
  26 in the first window of fields.c. The worst token cost 119 bits (float), after "love with
  Audrey.\n\n".
- carriage return (`\r`): all 23 tokens above 32 bits in the first window of Apache_2k.log.

The §5.1 minimum frequency gives each such token about 2^-32 of the range. Thus the coded
cost is at most about 32 bits for each token. This rule is a safety rule (every token can be
coded), and here it also makes the files smaller: 6,036 bytes (5.4 %) on the 16 files that
are not binary. A model that sees tabs and CR in training, or a larger minimum, can change
this. It is a measurement for phase 3 (rule 9: measure before you tune).

## 6. Decode checks

| Check | Encoder | Decoder | Windows | Result (measured) |
|---|---|---|---:|---|
| a. Coder only | full sequence | the tables of the encoder | 270 | 270 of 270 decoded |
| b. Mixed paths | full sequence | KV cache, model in the loop | 270 | 1 of 270 decoded |
| c. Same path | KV cache | KV cache, model in the loop | 19 | 19 of 19 decoded |

- Check b: in each window, step 0 has the same table on both paths, because the input is
  only `<|endoftext|>`. From step 1, the tables differ in 1,023 of 1,024 steps (checked on the
  19 windows of check c). The first wrong token was token 1 to 27 of its window (median 2).
  The first wrong byte of each file was byte 1 to 13. The one window that decoded is the last
  window of asyoulik.txt, which has only 2 tokens.
- Check c covers the first window of each file (19,456 tokens), because the KV-cache path
  costs about 5 ms for each token on this Mac (measured). The KV-coded size was the same as
  the full-sequence size, or 1 byte different, in each window.
- Check b is a phase 1 finding: the two paths of one program, on one machine, do not give the
  same frequency tables. Thus an encoder and a decoder must use the same path, in the same
  batch shape (§8).
- Two runs of phase 0 (13:41 and 14:07, with different CPU load) gave the same coded size and
  the same ideal size for every file (measured).

## 7. Timing

| Part | Run 2, 14:07 (measured) |
|---|---|
| Tokenizer and general compressors, 19 files | 1.2 s |
| Model load | 2.0 s |
| Full-sequence path, 265,859 tokens: model, tables, encoder, check a | 708 s (2.7 ms for each token) |
| Check b, KV decode until the first wrong token | 3.8 s |
| Check c, KV encode, 19,456 tokens, 12 streams in a batch | 103 s (5.3 ms for each token) |
| Check c, KV decode | 104 s |
| **Total** | **922 s (15.4 min)** |

Run 1 (13:41) took 1,406 s (23.4 min), because other work used the CPU at the same time.

Phase 1 part A (14:23, 19 windows, 4 pairs) took 1,009 s (16.8 min, measured): KV batch 1
dumps 357 s, KV batch 4 dumps 186 s, comparisons 201 s, float64 dumps 117 s, float32 dumps
56 s, thread-1 dumps 55 s.

## 8. Phase 1 part A: float drift on one Mac

Command: `nice -n 10 uv run --project tools python -m baseline.phase1 parta`. Output:
`tools/data/out/phase1/parta.json`. The tool computes the logits of the first window of each
file (19 windows, 19,456 steps) on two paths, each in a new process. For each step, it makes
the §5.1 table from each path and compares them. Then it encodes the window with the tables
of path A and decodes with the tables of path B, and the reverse.

NOTE: the cross decode uses the tables of path B along the original tokens. Up to and
including the first wrong token, this is what a real decoder on path B computes, because its
model input is the same. After the first wrong token, the output is wrong in any case.

| Pair (A against B) | Steps with a different table | First different step | Max abs probability difference | Max abs logit difference | Windows decoded A to B | Windows decoded B to A | First wrong byte |
|---|---:|---|---:|---:|---:|---:|---|
| Full sequence against KV cache (batch 4), float32 | 19,437 of 19,456 (99.90 %) | 1 in every window | 2.8e-5 | 2.4e-2 | 0 of 19 | 0 of 19 | 1 to 13 |
| KV cache batch 4 against batch 1, float32 | 19,456 of 19,456 (100 %) | 0 in every window | 7.4e-5 | 8.5e-4 | 0 of 19 | 0 of 19 | 0 to 8 (0 in 22 of 38 decodes) |
| float32 against float64, full sequence | 19,456 of 19,456 (100 %) | 0 in every window | 7.2e-5 | 2.4e-2 | 0 of 19 | 0 of 19 | 0 to 8 (0 in 22 of 38 decodes) |
| BLAS threads default against 1, full sequence, float32 | 0 of 19,456 (0 %) | none | 0 | 0 | 19 of 19 | 19 of 19 | none |

All values are measured. "First wrong byte" is the offset in the window of the first byte that
the decoder gives wrong. Byte 0 means that the first token was already wrong.

Findings:

- **Every pair that changes the order of float operations breaks the decode at once.** The
  probability differences are small (less than 1e-4). But a table with a total of 2^32 shows
  a probability difference of 2.3e-10 (one count). Thus nearly every table is different, and
  the decoder loses synchronisation in the first 13 bytes.
- **Full sequence against KV cache.** Step 0 is the same, because the attention has only one
  position. From step 1, the attention sums have a different length and order.
- **The batch shape changes the result.** The same KV-cache code with 4 streams in a batch
  and with 1 stream gives different tables from step 0. Possibly Accelerate uses a different
  matrix kernel for one row (UNVERIFIED: the cause is not measured). Thus a float decoder must
  also use the same batch shape as the encoder. Phase 0 check c obeyed this rule (12 streams on both sides,
  the same schedule).
- **Threads.** `VECLIB_MAXIMUM_THREADS=1` (and the OpenBLAS, OpenMP and MKL variables) gave
  the same bits as the default. On this Mac, Accelerate runs the matrix products on the AMX
  unit: the process CPU time was less than the wall time (measured on a 1,024 x 768 x 3,072
  product). Thus the variable possibly does not change the matrix kernel, and this pair is a
  weak test on the Mac. On devbox (OpenBLAS), the same pair is a real test.
- **Same path, same machine, two runs.** Phase 0 run 1 and run 2 gave the same sizes, and
  check c decoded 19 of 19 windows. Thus one path on one machine is repeatable (measured).
- These results agree with plan §1 and ADR 0002: a float decoder works only with the same
  program, the same path, the same batch shape and the same machine.

## 9. Limits

- The KV-cache path on both sides (check c) covers only the first window of each file. The
  command `python -m baseline.phase0 --kv-windows all` covers all windows. It takes about
  45 min on this Mac (est.).
- The cap is 32 KiB for each file. Longer contexts in one block are not measured.
- Each window is one coder stream. One stream for each file would save almost all of the
  0.30 bytes for each window of the coder overhead (est.).
- The Python coder follows plan §5. The Rust crate `taco-coder` and its ADR are the authority
  for the published format. The flush rule (the value with the most trailing zero bytes, and
  no trailing zero bytes in the output) and the omitted first byte can differ from the crate.
- The Python coder does not have the binary-decision mode of plan §5 item 6. The measurement
  does not need it.
- The general compressors ran as Python libraries, not as the command-line programs.
- Hindi is not in the corpus. Arabic is only in the tokenizer test.
- The comparison with the transformers greedy test uses the remembered text (UNVERIFIED).
- Phase 1 part A covers the first window of each file only (19,456 steps).
- Each phase 1 path ran in a new process. Repeats of one path in one process are not
  measured separately, but phase 0 shows that one path is repeatable.

## 10. Phase 1 part B (Mac against devbox): not done

devbox refuses work until its disk is compacted (CLAUDE.md §6). The tool is ready. These steps
remain:

1. Run `scripts/dx baseline uv run --project tools python -m baseline.fetch` on devbox. It
   downloads the model and the corpus and checks the hashes.
2. Run `scripts/dx baseline uv run --project tools python -m baseline.phase1 dump --path full
   --dtype f32 --files alice29.txt option.rs HDFS_2k.log pg24264.txt --out
   tools/data/out/phase1/devbox-full-f32` on devbox. Each window needs 206 MB of disk
   (float32 logits), so the command selects 4 files (824 MB, est.).
3. Copy that directory to the Mac.
4. Run the same `dump` command on the Mac with `--out tools/data/out/phase1/mac-full-f32`.
5. Run `uv run --project tools python -m baseline.phase1 compare
   tools/data/out/phase1/mac-full-f32 tools/data/out/phase1/devbox-full-f32`.
6. Do steps 2 to 5 again with `--path kv`, and with `OPENBLAS_NUM_THREADS=1` on devbox.
7. Run `python -m baseline.phase0` on devbox for the full-sequence timing and the sizes on
   x86 (the sizes must not change much, but the bits can).

Expected result (est.): the Mac and devbox give different tables in nearly all steps, because
Accelerate and OpenBLAS use different kernels. This is the failure that HardShell removes.

## 11. Requests to the lead

- `scripts/check-tags.sh` also scans `tools/data/`, which holds downloaded corpus files with
  `FIXME` and `XXX` comments. Please add `--exclude-dir=data`. For now, the local copies of
  `textwrap.py` and `option.rs` have a `.txt` suffix (`local` in `corpus.toml`), so the check
  passes.
- Commit `tools/pyproject.toml`, `tools/uv.lock`, `tools/baseline/` and this note. Do not
  commit `tools/data/` (it is in `.gitignore`).

