# Speculative decoding and the skip rule: engine speed and the expected unfold gain

Date: 2026-10-19. Machine: devbox (RTX 3060 Laptop GPU, 6 GB, driver 610.74, power limit
95 W). ADR: `docs/adr/0017-speculative-decoding-and-the-skip-rule.md`. Each number is
labelled `measured` or `est.`.

## 1. What was measured

1. **The engine speed of a verified draft** (`crates/hardshell-cuda/examples/draft.rs`): each
   call runs `k` rows for each of `S` slots (the known token and `k - 1` draft tokens) with
   the coder op or the mixed coder op, then truncates each slot. The slots decode at the
   positions 256 to 1,000. Each shape runs for 2 s after 8 calls of warm-up.
2. **The tokens for each sequential pass** of one block (`tools/logs/skip.py final`): the
   protocol of ADR 0017 §3 on the facts of `logs.gpt2pass` (18 logs, 6,556,027 bytes,
   2,798,749 tokens).

Conditions of the speed run (measured): the timing lock held, the program on processors 0
to 5, other agents used the other processors (load average 9.8). The GPU used 65 to 95 W at
87 to 99 % load, 1,250 to 1,890 MHz, 86 to 87 °C (sampled each 10 s). The power fell from
95 W to about 68 W during the run, with the temperature at 86 °C. Thus the numbers can be
about 10 % low against a cold GPU (est.). A first run (2026-10-19 08:05 PDT, shorter shapes,
a build on the other processors) gave the same values within about 10 %.

## 2. Engine speed (measured, `gpt2.hsm` `aaa03eb9…a5c2`)

Calls per second (one call is one sequential step of the decoder for every slot), and rows
per second.

| Op | Slots | Draft 1 | Draft 4 | Draft 8 | Draft 16 | Draft 32 |
|---|---:|---:|---:|---:|---:|---:|
| coder op | 1 | 646 calls/s | 412 | 352 | 410 | 388 |
| coder op | 8 | 417 | 346 | 310 | 241 | 154 |
| coder op | 32 | 343 | 142 | 114 | 74 | 41 |
| mixed coder op | 1 | 561 | 361 | 306 | 362 | 331 |
| mixed coder op | 8 | 357 | 294 | 262 | 195 | 119 |
| mixed coder op | 32 | 307 | 124 | 95 | 59 | 31 |

| Op | Slots | Rows/s, draft 1 | Draft 4 | Draft 8 | Draft 16 | Draft 32 |
|---|---:|---:|---:|---:|---:|---:|
| coder op | 1 | 646 | 1,646 | 2,817 | 6,558 | 12,429 |
| coder op | 8 | 3,338 | 11,080 | 19,846 | 30,798 | 39,364 |
| coder op | 32 | 10,971 | 18,188 | 29,227 | 37,623 | 41,499 |
| mixed coder op | 1 | 561 | 1,444 | 2,450 | 5,792 | 10,580 |
| mixed coder op | 8 | 2,855 | 9,422 | 16,775 | 25,013 | 30,526 |
| mixed coder op | 32 | 9,818 | 15,802 | 24,316 | 30,288 | 31,503 |

`distil_logs.hsm` (6 layers, the Route A model) is about 1.4 to 1.5 times faster: mixed coder
op, 1 slot: 779, 535, 445, 544 and 495 calls/s for the drafts 1, 4, 8, 16 and 32; 8 slots:
526, 446, 383, 282 and 168 calls/s (measured).

Findings:

- **At 1 slot, a draft of 32 rows costs 1.7 to 2.0 times one row** (2.6 to 3.0 ms against 1.6
  to 1.8 ms). The step at a small batch reads all the weights for each call, so more rows are
  almost free. Drafts of 4 and 8 rows are slower than 16 rows at 1 slot (measured in both
  runs): `MatKernel::for_rows` selects the dp4a products below 16 rows and the tensor cores
  from 16 rows (est. from the kernel selection; not profiled).
- **At 32 slots, the GPU is full**: the rows per second stay near 30,000 (mixed) to 41,000
  (plain) for drafts of 16 and 32. Speculative decoding then helps little: it gives more rows,
  not faster rows.
- The mixed coder op costs 10 % to 25 % more than the coder op for each call.

## 3. Tokens for each pass (simulation, 18 logs, 64 KiB blocks)

From ADR 0017 (measured with `tools/logs/skip.py final`; the full table for 16 KiB, 64 KiB and
1 MiB is in the ADR):

| Protocol | Tokens for each pass | Rows for each pass |
|---|---:|---:|
| One token for each pass (today) | 1.00 | 1.00 |
| Speculative, draft 8 | 3.35 | 8 |
| Speculative, draft 16 | 3.96 | 16 |
| Speculative, draft 16, with a redecode op | 5.31 | 16 |
| Skip rule only (rule 2) | 3.62 | 3.62 |
| Skip rule and speculative, draft 8 | 4.73 | 10.6 |
| Skip rule and speculative, draft 16, with a redecode op | 5.89 | 18.8 |

With 1 MiB blocks the values are higher (skip rule and draft 16 with redecode: 6.83 tokens
for each pass). By file (64 KiB, skip rule and draft 16 with redecode): 2.3 (HPC) to 14.1
(nginx JSON lines) tokens for each pass.

## 4. Expected unfold gain on logs (est.)

The expected decoder speed of the profile `logs` with the mixed coder op on CUDA is `S` times
the tokens for each pass (section 3) times the calls per second of the pass shape (section 2).
The logs have 2.34 bytes for each token (measured on the 18 logs). The host work (the match
model, the mixer, the decoder) is not in these values: about 50 µs for each token with a
prediction on one CPU thread (measured, `taco-match` `examples/cost.rs` on devbox, synthetic
log), and the skip rule removes the GPT-2 table of a skipped hit.

| Slots (blocks that decode at the same time) | Today: 1 token for each pass | Speculative, draft 8 | Skip rule only | Skip rule and draft 8 |
|---|---:|---:|---:|---:|
| 1 (a small file) | 561 tokens/s (1.3 KB/s) | 1,026 (1.8 times) | 1,307 (2.3 times) | 1,533 (2.7 times) |
| 8 | 2,854 (6.5 KB/s) | 7,024 (2.5 times) | 8,526 (3.0 times) | 9,097 (3.2 times) |
| 32 (a large file) | 9,818 (22.4 KB/s) | 10,184 (1.04 times) | 14,306 (1.5 times) | 12,623 (1.3 times) |

KB is 1,024 bytes.

How the values were made (est.):

- Speculative draft 8: `S * 3.35 * calls/s(S, 8)`.
- Skip rule only: `S * 3.62 * calls/s(S, 4)` (3.62 rows for each pass).
- Skip rule and draft 8: `S * 4.73 * calls/s(S, 10.6 rows)`, with the calls per second
  interpolated between the drafts 8 and 16.
- With a redecode op (ADR 0017 §7), the skip rule and draft 16, 1 slot: `5.89 * 357 = 2,100`
  tokens/s (3.7 times; 357 calls/s interpolated at 18.8 rows).

Conclusions (est.):

1. The gain is largest when few blocks decode at the same time (a small file or the last
   blocks of a file): 1.8 to 3.7 times. These are the cases where unfold is slow today
   (`2026-10-19-full-power.md` §3).
2. With many slots the GPU is full, and only the skip rule helps (about 1.5 times), because it
   needs no wrong draft rows. At 32 slots, drafts make the skip rule slower (1.3 times against
   1.5 times): a decoder must select the draft length from the number of slots.
3. The skip rule gives the most for its cost: one row for each token, no wrong rows, and a
   smaller file (ADR 0017: −4.8 % at 64 KiB). Speculative decoding adds 7 % to 17 % to it at 1
   to 8 slots, and 61 % at 1 slot with a redecode op.
4. The CPU path of unfold (no GPU) gains more from the skip rule than from drafts: a skipped
   hit needs no forward row output and no table of 50,257 symbols (est.).

## 5. Equality (measured)

- CUDA, real GPT-2, 2 slots, 3,400 tokens, drafts of up to 16 rows, `K = 0` and `K = 512`: every
  used row has the SHA-256 of the CPU row (546 and 555 calls).
- wgpu (OpenGL on D3D12, RTX 3060), the same test with 3,200 tokens: equal for `K = 0` and
  `K = 512` (533 and 563 calls).
- The 7 windows of `examples/gpt2.rs equiv` (`gpt2.hsm`): equal on CUDA and on wgpu, with the
  published CPU SHA-256 values (`a9b37a6c…809b` to `b0de8f4c…01ba`); 2,643 of 2,643 coder
  intervals equal on each.
