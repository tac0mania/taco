# Research: the logs profile in `taco fold` (2026-10-19)

This note records the first real measurement of the `logs` profile (ADR 0010, ADR 0012) with
`taco fold` and `taco unfold`. The simulation in `docs/research/2026-10-18-logs.md` (the "logs
note") predicted the sizes. This note gives the real sizes. "bpb" is bits per byte. All numbers
are measured, unless a number has a different label.

## 1. Results in short

- **The real sizes agree with the simulation.** With blocks of 16 KiB, the 16 Loghub files give
  0.657 bpb (simulation 0.658, −0.2 %). Each of the 18 logs is 0.0 % to 0.6 % smaller than
  its simulation. With blocks of 1 MiB, the 16 Loghub files give 0.489 bpb (simulation 0.497,
  −1.8 %), and each of the 18 logs is 0.2 % to 5.6 % smaller than its simulation. Section 7
  explains the gaps.
- **Against xz -9e and zstd -19 --long=27, 16 KiB blocks:** the 16 Loghub files are 32.0 %
  smaller than xz and 31.4 % smaller than zstd. Taco is the smallest on 18 of 18 logs: −14.4 %
  (HDFS) to −43.0 % (Apache access) against the better of xz and zstd.
- **Against xz and zstd, 1 MiB blocks:** the 16 Loghub files are 11.7 % smaller than xz and
  18.2 % smaller than zstd. Taco is the smallest on 17 of 18 logs. The Apache access log is
  0.397 bpb (−12.3 % against xz, −17.7 % against zstd). The nginx JSON lines are 0.167 bpb
  (−3.6 % against xz, −13.6 % against zstd).
- **OpenStack with 1 MiB blocks:** the mixed candidate was larger than the zstd frame, so the
  encoder stored the zstd block (40,801 bytes, 0.548 bpb). The Taco file is 5.5 % larger than
  the xz file. The logs note gave the mixed simulation 11.8 % larger than xz.
- **The logs profile against the text profile:** −24.9 % on the Loghub files with 16 KiB
  blocks, −18.5 % with 1 MiB blocks. With 1 MiB blocks, the text container is near the size of
  zstd (0.599 bpb), because GPT-2 alone (about 1.0 bpb, simulation) loses to zstd on large log
  blocks.
- **Text control:** on the 14 text files (16 KiB blocks), the logs profile gives 2.002 bpb and
  the text profile 2.207 bpb (−9.33 %). Every file becomes smaller (−1.7 % to −28.8 %). The
  simulation gave 2.009 bpb for the logs profile.
- **The text profile did not change.** The golden container is the same, and the real text
  containers have exactly the size that the logs fold computes for them: 18 of 18 logs with
  16 KiB blocks, 2 of 2 logs with 1 MiB blocks, 14 of 14 text files.
- **Round trips.** Each unfold restored the input exactly, byte for byte: 82 of 82 round trips, 0
  failures.
  The 82 round trips are 18 + 18 logs with 16 KiB blocks (logs and text profile), 18 logs
  with 1 MiB blocks, and 14 + 14 text files.
- **The committed code makes the same containers.** A clean build of commit 870ffaf gave the
  same SHA-256 as the run binary on 5 of 5 inputs (section 4).

## 2. What the code does now

1. `taco fold --profile text|logs|auto`. The default of the command is `auto` (ADR 0012). The
   default of the library (`FoldOptions::new()`) is `text`.
2. The profile `text` codes with HardShell alone. Its output did not change: the golden
   container `0b0ba3f1…7c07` (1,660 bytes) is the same.
3. The profile `logs` codes each block two times from the same HardShell weights: with GPT-2
   alone, and with the match model and the mixer of `taco-match` (mixer rule version 1). The
   smallest of raw, zstd, coded and mixed is stored. On a tie, the order is raw, zstd, coded,
   mixed. A mixed block is block kind 3 (ADR 0010 §5).
4. The profile `auto` codes as `logs`. If no block is mixed, it writes the `text` container.
5. `taco unfold` reads the profile and the coding parts from the header. It needs no flag.
6. The fast paths are the same as for `text`: prefill chunks of 32 tokens in fold, and lanes
   of a `Batch` in unfold. Each mixed block has its own match model and mixer, so a mixed
   block decodes in any lane.

## 3. The format (ADR 0010 §5, ADR 0011 §2 and §3)

| Field | Rule |
|---|---|
| Header offset 90, coding parts (`u16`) | Bit 0: the match model and the mixer. Bits 1 to 15: 0. |
| Header offset 92, byte 0 of the coding parameters | The mixer rule version (1) when bit 0 is set, else 0. Bytes 93 to 103: 0. |
| Profile and coding parts | The profile `text` has no coding part. The profile `logs` has bit 0 or no coding part. |
| Block kind 3, "mixed" | The payload of ADR 0007, coded with the match model and the mixer. Only with bit 0. Smaller than the original bytes. Not in the raw-only mode. |

A reader refuses each value that the table does not permit, with a typed error
(`UnknownCodingParts`, `UnknownMixerRule`, `ReservedNotZero`, `CodingPartsNotInProfile`,
`MixedWithoutCodingPart`, `UnknownBlockKind`, `CodedBlockInRawOnly`, `CodedNotSmaller`).

The per-block choice is a block kind and not a flag, because the block table already has a kind
byte for each block. Thus the choice costs no byte, and the encoder can keep the smaller of the
two forms of each block. A `logs` container is never larger than the `text` container of the same
input and block size.

## 4. Method

1. **Data.** The 18 log files of the logs note (`tools/data/logs`): the 16 Loghub files whole,
   and the first 1 MiB of `apache_access.log` and of `nginx_json.log`. The text control is the
   14 text files of the logs note, the first 32 KiB of each file from byte 0.
2. **Model.** `tools/data/out/gpt2.hsm`, SHA-256 `aaa03eb9…a5c2`, on both machines.
3. **Program.** `taco fold --profile logs` and `--profile text`, release build, one thread for
   each job, `--block-size 16384` and `--block-size 1048576`. `taco unfold` (one thread; the
   profile comes from the header) restored each container. A script compared the output with
   the input, byte for byte.
4. **The binary.** The runs used a release build of the tree of 2026-10-18 at 22:00 +04. To
   prove that this binary is the committed code, devbox built `taco-cli` from a clean export of
   commit 870ffaf (`git archive`). That build folded five inputs again: `grammar.lsp` (logs and
   text), `xargs.1` (logs), `HPC_2k.log` (logs, 16 KiB) and `Spark_2k.log` (logs, 1 MiB). All
   five containers have the same SHA-256 as the containers of the runs. Commit 92001b9 (HEAD
   now) changes only a script and a note, not the program.
5. **Machines.** devbox (Linux x86, Ryzen 7 5800H, `nice 15`, at most 10 jobs on cores 0 to 9,
   then 8 jobs on cores 0 to 7) ran the logs profile with blocks of 1 MiB, and seven Loghub
   files and the two Elastic files with blocks of 16 KiB, and the two text folds with blocks of
   1 MiB. The Mac (Apple M2, `nice`) ran the
   text files, the text profile of the 18 logs with blocks of 16 KiB, and the nine smallest
   Loghub files with the logs profile and blocks of 16 KiB. The xcheck of commit 870ffaf (11
   of 11 containers identical) and the HPC check above show that both machines make the same
   containers.
6. **The text profile with blocks of 1 MiB.** The logs profile also computes the size of the
   text container (`FoldStats::text_container_len`), because the coded candidate of each block
   is the same in both profiles. For blocks of 16 KiB, a real text fold of each of the 18
   logs gave exactly this size. For blocks of 1 MiB, real text folds of `Apache_2k.log` (7,270
   bytes) and `Linux_2k.log` (11,435 bytes) gave exactly this size too. Thus the
   1 MiB text column (marked `*`) uses this size.
7. **General compressors.** xz 5.8.3 (`-9e`) and zstd 1.5.7 (`-19 --long=27`) on the Mac.
   Each block is compressed alone, and each output is decompressed and compared.
8. **Simulation.** The exact upper bound of the table cost of each file and block size
   (`tools/data/out/logs/final_o32-16-8-4-2-1_h20_lr160_lim255_wm65536_min16_d64_a1.json`, the
   source of the logs note), plus the container estimate of the logs note: 88 bytes, and 9
   bytes for each block. The simulation codes every block as mixed.

## 5. Logs with blocks of 1 MiB

"Simulation" is the GPT-2 + match column of the logs note, section 5. "Blocks" gives the
number of mixed, coded (GPT-2 alone) and other (zstd or raw) blocks of the logs container.

| File | Bytes | xz -9e | zstd -19 --long=27 | Taco text | Taco logs | Simulation | Real against simulation | logs against best of xz, zstd | logs against text | Blocks: mixed, coded, other | Round trip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| HDFS_2k.log | 287,848 | 1.177 | 1.190 | 1.193 * | **1.143** | 1.155 | -1.0 % | -2.9 % | -4.2 % | 1, 0, 0 | ok |
| Apache_2k.log | 171,239 | 0.315 | 0.335 | 0.340 | **0.218** | 0.221 | -1.2 % | -30.8 % | -35.7 % | 1, 0, 0 | ok |
| BGL_2k.log | 317,150 | 0.992 | 1.100 | 1.103 * | **0.891** | 0.900 | -1.0 % | -10.2 % | -19.2 % | 1, 0, 0 | ok |
| Linux_2k.log | 216,485 | 0.370 | 0.419 | 0.423 | **0.275** | 0.280 | -1.9 % | -25.7 % | -34.9 % | 1, 0, 0 | ok |
| OpenSSH_2k.log | 225,216 | 0.346 | 0.386 | 0.389 * | **0.259** | 0.267 | -3.0 % | -25.2 % | -33.5 % | 1, 0, 0 | ok |
| Spark_2k.log | 196,268 | 0.370 | 0.403 | 0.408 * | **0.361** | 0.366 | -1.4 % | -2.3 % | -11.5 % | 1, 0, 0 | ok |
| Zookeeper_2k.log | 279,891 | 0.425 | 0.474 | 0.478 * | **0.379** | 0.385 | -1.7 % | -11.0 % | -20.7 % | 1, 0, 0 | ok |
| Android_2k.log | 279,076 | 0.441 | 0.471 | 0.474 * | **0.364** | 0.367 | -0.6 % | -17.5 % | -23.2 % | 1, 0, 0 | ok |
| Windows_2k.log | 285,433 | 0.239 | 0.256 | 0.259 * | **0.215** | 0.217 | -0.8 % | -10.0 % | -16.9 % | 1, 0, 0 | ok |
| Mac_2k.log | 319,414 | 0.856 | 0.909 | 0.911 * | **0.713** | 0.715 | -0.4 % | -16.7 % | -21.8 % | 1, 0, 0 | ok |
| HealthApp_2k.log | 187,456 | 0.520 | 0.570 | 0.574 * | **0.514** | 0.515 | -0.2 % | -1.2 % | -10.6 % | 1, 0, 0 | ok |
| Proxifier_2k.log | 236,962 | 0.584 | 0.626 | 0.613 * | **0.423** | 0.429 | -1.3 % | -27.5 % | -31.0 % | 1, 0, 0 | ok |
| Hadoop_2k.log | 384,948 | 0.242 | 0.278 | 0.280 * | **0.220** | 0.224 | -1.9 % | -9.1 % | -21.5 % | 1, 0, 0 | ok |
| HPC_2k.log | 151,178 | 0.999 | 1.032 | 1.038 * | **0.814** | 0.821 | -0.8 % | -18.5 % | -21.6 % | 1, 0, 0 | ok |
| Thunderbird_2k.log | 325,192 | 0.484 | 0.570 | 0.573 * | **0.349** | 0.353 | -1.2 % | -27.9 % | -39.0 % | 1, 0, 0 | ok |
| OpenStack_2k.log | 595,119 | 0.520 | 0.547 | 0.548 * | **0.548** | 0.581 | -5.6 % | +5.5 % | +0.0 % | 0, 0, 1 | ok |
| **Loghub, 16 files** | 4,458,875 | 0.554 | 0.597 | 0.599 | **0.489** | 0.497 | -1.8 % | xz -11.7 %, zstd -18.2 % | -18.5 % | | |
| apache_access.log | 1,048,576 | 0.453 | 0.482 | 0.483 * | **0.397** | 0.400 | -0.8 % | -12.3 % | -17.9 % | 1, 0, 0 | ok |
| nginx_json.log | 1,048,576 | 0.173 | 0.193 | 0.194 * | **0.167** | 0.170 | -1.8 % | -3.6 % | -14.0 % | 1, 0, 0 | ok |

## 6. Logs with blocks of 16 KiB

| File | Bytes | xz -9e | zstd -19 --long=27 | Taco text | Taco logs | Simulation | Real against simulation | logs against best of xz, zstd | logs against text | Blocks: mixed, coded, other | Round trip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| HDFS_2k.log | 287,848 | 1.488 | 1.467 | 1.441 | **1.255** | 1.257 | -0.2 % | -14.4 % | -12.9 % | 18, 0, 0 | ok |
| Apache_2k.log | 171,239 | 0.503 | 0.465 | 0.470 | **0.282** | 0.283 | -0.1 % | -39.3 % | -39.9 % | 11, 0, 0 | ok |
| BGL_2k.log | 317,150 | 1.322 | 1.360 | 1.163 | **0.998** | 1.001 | -0.3 % | -24.5 % | -14.2 % | 20, 0, 0 | ok |
| Linux_2k.log | 216,485 | 0.581 | 0.582 | 0.548 | **0.349** | 0.351 | -0.4 % | -39.9 % | -36.2 % | 14, 0, 0 | ok |
| OpenSSH_2k.log | 225,216 | 0.586 | 0.571 | 0.575 | **0.335** | 0.336 | -0.5 % | -41.4 % | -41.8 % | 14, 0, 0 | ok |
| Spark_2k.log | 196,268 | 0.636 | 0.620 | 0.617 | **0.453** | 0.454 | -0.1 % | -26.9 % | -26.5 % | 12, 0, 0 | ok |
| Zookeeper_2k.log | 279,891 | 0.705 | 0.710 | 0.703 | **0.483** | 0.485 | -0.4 % | -31.5 % | -31.3 % | 17, 1, 0 | ok |
| Android_2k.log | 279,076 | 1.077 | 1.079 | 0.966 | **0.639** | 0.639 | -0.0 % | -40.7 % | -33.8 % | 18, 0, 0 | ok |
| Windows_2k.log | 285,433 | 0.473 | 0.443 | 0.438 | **0.307** | 0.308 | -0.4 % | -30.8 % | -30.0 % | 18, 0, 0 | ok |
| Mac_2k.log | 319,414 | 2.068 | 2.089 | 1.574 | **1.255** | 1.256 | -0.1 % | -39.3 % | -20.3 % | 20, 0, 0 | ok |
| HealthApp_2k.log | 187,456 | 0.779 | 0.782 | 0.781 | **0.627** | 0.629 | -0.3 % | -19.5 % | -19.8 % | 12, 0, 0 | ok |
| Proxifier_2k.log | 236,962 | 0.882 | 0.886 | 0.630 | **0.516** | 0.517 | -0.2 % | -41.5 % | -18.1 % | 15, 0, 0 | ok |
| Hadoop_2k.log | 384,948 | 0.575 | 0.567 | 0.555 | **0.356** | 0.357 | -0.3 % | -37.2 % | -35.8 % | 24, 0, 0 | ok |
| HPC_2k.log | 151,178 | 1.159 | 1.117 | 1.069 | **0.851** | 0.853 | -0.2 % | -23.8 % | -20.5 % | 10, 0, 0 | ok |
| Thunderbird_2k.log | 325,192 | 0.796 | 0.785 | 0.711 | **0.462** | 0.464 | -0.4 % | -41.2 % | -35.0 % | 20, 0, 0 | ok |
| OpenStack_2k.log | 595,119 | 1.153 | 1.125 | 1.125 | **0.848** | 0.850 | -0.2 % | -24.6 % | -24.6 % | 37, 0, 0 | ok |
| **Loghub, 16 files** | 4,458,875 | 0.966 | 0.958 | 0.874 | **0.657** | 0.658 | -0.2 % | xz -32.0 %, zstd -31.4 % | -24.9 % | | |
| apache_access.log | 1,048,576 | 1.215 | 1.230 | 0.950 | **0.693** | 0.694 | -0.3 % | -43.0 % | -27.1 % | 64, 0, 0 | ok |
| nginx_json.log | 1,048,576 | 0.419 | 0.404 | 0.403 | **0.246** | 0.247 | -0.6 % | -39.1 % | -38.9 % | 64, 0, 0 | ok |

## 7. Real sizes against the simulation

The simulation adds an estimate of the container to the exact table cost: 88 bytes, and 9
bytes for each block. The real container has 104 bytes of header (version 2), 5 bytes of block
table for each block, and the token count of each coded or mixed block (a varint: 2 bytes for
a block of 16 KiB, 3 bytes for a block of 1 MiB). The coder flush is part of the coder bytes.
Thus the real container costs 112 bytes for one block of 1 MiB (est. 97) and 230 bytes for 18
blocks of 16 KiB (est. 250). This difference is less than 0.1 % of every log file. It is
+1.4 % and +1.5 % for the two smallest text files (`xargs.1`, `grammar.lsp`, 4 KiB each).

To remove the container term, the next table compares the coder bytes of the mixed blocks
(the stored bytes less the token count) with the exact table cost of the simulation. Only
containers in which every block is mixed are in the table (`payload.py`, measured).

| Block | Files | Coder bytes against the exact cost of the simulation | Total |
|---|---:|---|---:|
| 16 KiB | 15 Loghub files (not Zookeeper: 1 coded block) | −0.33 % (OpenSSH) to +0.06 % (Android) | −0.12 % |
| 16 KiB | `apache_access.log`, `nginx_json.log` | −0.13 %, −0.30 % | |
| 16 KiB | 11 text files with the same bytes | −0.63 % (option.rs) to +0.34 % (asyoulik) | −0.05 % |
| 1 MiB | 15 Loghub files (not OpenStack: zstd block) | −3.26 % (OpenSSH) to −0.34 % (HealthApp) | −1.2 % |
| 1 MiB | `apache_access.log`, `nginx_json.log` | −0.86 %, −1.83 % | |

Findings:

1. **16 KiB blocks: the real coder bytes and the simulation agree within 0.7 % on every file.** The
   coder overhead of phase 0 (0.021 %) explains only a small part of the residual. The
   probable cause is the position of the GPT-2 restarts inside a block (est., UNVERIFIED).
   The simulation starts a new window at each 1,024 tokens from the block start. HardShell
   starts the context again when the context is full (§7.7). A shift of one token at each
   restart changes the cost of the first tokens of each window in both directions.
2. **1 MiB blocks: the real coder bytes are 0.3 % to 3.3 % smaller than the simulation.** The logs note
   (section 8) gave the cause before the measurement: the simulation starts the GPT-2 context
   again at each 16 KiB step for every block size, and Taco with a block of 1 MiB starts it
   again only when the context is full. The logs note estimated less than 1 %. The real gap is
   larger (total −1.2 %). The match model and the mixer do not restart in either case.
3. **Tokens at the 16 KiB boundaries.** The simulation tokenizes each 1 MiB block as a sequence of
   16 KiB pieces. Taco tokenizes the whole block. The real block of 1 MiB has 4 to 20 fewer tokens
   than the simulation on the 16 Loghub files (for example HDFS: 112,478 against 112,491).
   This effect is very small (est. less than 0.01 % of the bits).
4. **OpenStack, 1 MiB.** The simulation codes every block as mixed (43,217 bytes with the
   container estimate). Taco stores the smaller zstd frame (40,801 bytes). Thus the real file
   is 5.6 % smaller than the simulation. The size of the mixed candidate is not in the
   container, so this note cannot compare it.
5. **The text column of the simulation is not comparable with the real text profile.** The
   simulation gives "GPT-2 alone" for every block (about 1.0 bpb on the Loghub files). The
   real text profile stores the smallest of raw, zstd and coded for each block. With 1 MiB
   blocks the real text size is near the zstd size (0.599 bpb against 0.597 bpb for zstd
   --long on the Loghub files). The logs fold does not record which candidate the text
   container uses.

## 8. Text files (control)

The simulation of the text files read the Gutenberg files `pg22367.txt`, `pg4650.txt` and
`pg24264.txt` from the offset of `tools/baseline/corpus.toml` (828 to 950 bytes, after the
licence header). The real runs read them from byte 0. Thus these three rows compare different
bytes, and the real run has fewer tokens (for example 13,463 against 13,555 for `pg22367.txt`).
The other 11 files have the same bytes and the same token counts.

| File | Bytes | xz -9e | zstd -19 --long=27 | Taco text | Taco logs | Simulation, logs | Real against simulation | logs against text | Blocks: mixed, coded, other | Round trips (logs, text) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| alice29.txt | 32,768 | 3.271 | 3.244 | 2.062 | **1.956** | 1.954 | +0.1 % | -5.2 % | 2, 0, 0 | ok, ok |
| asyoulik.txt | 32,768 | 3.398 | 3.371 | 2.423 | **1.992** | 1.982 | +0.5 % | -17.8 % | 2, 0, 0 | ok, ok |
| cp.html | 24,603 | 2.793 | 2.776 | 1.553 | **1.474** | 1.472 | +0.2 % | -5.1 % | 2, 0, 0 | ok, ok |
| fields.c | 11,150 | 2.181 | 2.165 | 2.241 | **1.654** | 1.651 | +0.2 % | -26.2 % | 1, 0, 0 | ok, ok |
| grammar.lsp | 3,721 | 2.795 | 2.610 | 1.739 | **1.668** | 1.643 | +1.5 % | -4.1 % | 1, 0, 0 | ok, ok |
| lcet10.txt | 32,768 | 3.184 | 3.171 | 1.810 | **1.734** | 1.730 | +0.2 % | -4.2 % | 2, 0, 0 | ok, ok |
| plrabn12.txt | 32,768 | 3.632 | 3.589 | 2.370 | **1.972** | 1.969 | +0.1 % | -16.8 % | 2, 0, 0 | ok, ok |
| xargs.1 | 4,227 | 3.445 | 3.272 | 1.686 | **1.658** | 1.635 | +1.4 % | -1.7 % | 1, 0, 0 | ok, ok |
| pg22367.txt | 32,768 | 3.420 | 3.392 | 3.064 | **2.975** | 3.027 | -1.7 % | -2.9 % | 2, 0, 0 | ok, ok |
| pg4650.txt | 32,768 | 3.383 | 3.371 | 2.948 | **2.802** | 2.827 | -0.9 % | -5.0 % | 2, 0, 0 | ok, ok |
| pg24264.txt | 32,768 | 4.204 | 4.164 | 3.810 | **3.660** | 3.697 | -1.0 % | -3.9 % | 2, 0, 0 | ok, ok |
| textwrap.py.txt | 19,718 | 2.574 | 2.507 | 1.323 | **1.226** | 1.222 | +0.3 % | -7.3 % | 2, 0, 0 | ok, ok |
| option.rs.txt | 32,768 | 2.199 | 2.163 | 1.189 | **1.045** | 1.049 | -0.3 % | -12.1 % | 2, 0, 0 | ok, ok |
| twitter.json | 32,768 | 1.522 | 1.617 | 1.323 | **0.941** | 0.939 | +0.3 % | -28.8 % | 2, 0, 0 | ok, ok |
| **14 files** | 358,331 | 3.051 | 3.030 | 2.207 | **2.002** | 2.009 | -0.4 % | -9.33 % | | |

## 9. Speed

Warning: these speeds are not a benchmark. Each job used one thread, at low priority, on
machines that other agents used at the same time (devbox load 13 to 38, measured). Up to 10
jobs ran together on devbox and up to 6 together on the Mac. Use them only for orders of
size.

| Job | Machine | Fold, tokens/s for each job | Unfold, tokens/s for each job |
|---|---|---:|---:|
| logs, 1 MiB (one block) | devbox | 29 to 102 (median 78) | 12 to 23 (median 19) |
| logs, 16 KiB | devbox | 46 to 79 | 39 to 61 |
| logs, 16 KiB | Mac | 92 to 153 | 65 to 188 |
| text, 16 KiB | Mac | 65 to 190 | 48 to 3,824 |

1. The unfold of a block of 1 MiB is slow: it is one block, thus one lane of the `Batch`, and
   it does one HardShell step for each token. The block of 1 MiB of `apache_access.log` (480,406
   tokens) took 23,870 s (6.6 hours) on the shared devbox (measured). Fold is faster, because it uses prefill
   chunks of 32 tokens.
2. With blocks of 16 KiB, unfold decodes many blocks together in the lanes of a `Batch`.
3. The unfold of a text container is fast when most blocks are zstd blocks (for example
   OpenStack: 3,824 tokens/s), because a zstd block needs no model.
4. The logs profile computes the coded and the mixed candidates from the same HardShell weights.
   The match model and the mixer add about 0.1 ms for each token (ADR 0010 section 7, est.).
   This measurement does not isolate that cost.

## 10. Limits

- **Round trips.** Each unfold restored the input exactly, byte for byte: 82 of 82 round trips, 0
  failures.
  The 82 round trips are 18 + 18 logs with 16 KiB blocks (logs and text profile), 18 logs
  with 1 MiB blocks, and 14 + 14 text files.
- **The text profile with 1 MiB blocks** is a real fold for 2 files and `text_container_len`
  for 16 files (marked `*`). Both methods gave the same size on all 34 files where both
  exist.
- **One run for each job.** The sizes are deterministic (the xcheck and the HEAD check prove
  it). The speeds are not reproducible (section 9).
- **The cause of the 16 KiB residual (less than 0.7 %)** is an estimate (section 7, finding 1). A
  simulation with the restart rule of HardShell can confirm it.
- **The Gutenberg files** of the text control compare different bytes with the simulation
  (section 8).
- **bzip2** is not in this note. The logs note measured bzip2 -9 at 0.157 bpb on
  `nginx_json.log` with 1 MiB blocks. That is 5.9 % smaller than the real Taco file (0.167
  bpb). It is the only log of the 18 where a general compressor of the logs note is smaller
  than Taco with 1 MiB blocks, except OpenStack.

## 11. Commands

Run from the repository root. The runner scripts of this measurement are not in the
repository (`run.py`, `baselines.py`, `tables.py` in the scratch directory of the session).
They call these commands for each file, block size and profile.

```
scripts/mx cargo build --release -p taco-cli
taco fold FILE -o FILE.taco --model tools/data/out/gpt2.hsm --profile logs --block-size 1048576 --threads 1
taco unfold FILE.taco -o FILE.back --model tools/data/out/gpt2.hsm --threads 1 && cmp FILE FILE.back
xz -9e -c BLOCK | wc -c          # each block alone
zstd -19 --long=27 -c BLOCK | wc -c
git archive --prefix=head-src/ 870ffaf | ssh devbox tar -x   # the clean build of the check
scripts/xcheck
```
