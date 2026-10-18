# Taco: working rules

Taco is a lossless compressor. A language model predicts the next token. A range coder keeps
only the part that the model did not predict. HardShell is the engine that runs the model. It
uses only integer arithmetic. Thus every machine computes the same predictions, bit for bit,
and a file that one machine compresses decompresses on every other machine.

Read these documents before you change anything:

- `HANDOFF.md`: the current state and the next steps. Read it first.
- `docs/design/plan.md`: the design. Section numbers below (§) refer to it.
- `docs/ARCHITECTURE.md`: the crates, the tools, and the permitted dependencies.
- `docs/adr/`: the decisions that are already made. Add an ADR when you make a decision that
  other people must obey.
- `docs/STYLE.md`: how to write documents (ASD-STE100).

If the code and the plan do not agree, stop and tell the lead. Do not select one of them
yourself.

## 1. The rules that you must not break

These rules come from the plan (§3). If a change must break a rule, first change the rule in an
ADR.

1. **No floating-point arithmetic in the prediction path.** The prediction path is all the code
   from the model weights to the frequency table that the coder receives. This includes
   HardShell, the frequency conversion and the coder. Do not use `f32`, `f64`, `half`, or a
   function from `libm` in it. Tables of non-linear functions are data in the model file. A
   tool calculates them one time, before publication.
2. **Every integer operation has one defined result.** Use `wrapping_*` or `checked_*`
   operations, or write a bound (`INVARIANT:`) that shows that overflow cannot occur. Do not
   use `/` or `%` on signed values. Use the floor-division helpers in `hardshell::int`. Use only
   the rounding rules in §7.2. Do not use saturating operations unless the spec names them.
3. **The model file is the model.** The encoder and the decoder load the same published file.
   Do not convert or quantize a model on the machine that decodes. The container stores the
   SHA-256 of the model file. The decoder refuses a file whose model hash it does not have.
4. **Lossless is tested, not assumed.** Every container stores a checksum of the original data.
   Decompression verifies it. A failed check is an error, never a silent wrong output.
5. **Every symbol can be coded.** Every token in a frequency table has a frequency of 1 or more.
   Any byte string can be tokenized and restored exactly (§10).
6. **The worst case is bounded.** A block that does not become smaller is stored raw. The
   container grows by its header and a flag per block only (§9).
7. **Conformance vectors are the contract.** A HardShell implementation (CPU, GPU, a new
   language) is correct only when it produces the reference outputs bit for bit (§11). Run the
   cross-machine check (Mac ARM and devbox x86) before you call a HardShell change done.
8. **Formats have versions.** The model file and the container each have a format version. A
   change to a persisted shape adds a new version, a reader for the old version, and a test
   that reads the old version.
9. **Measure before you tune.** Do not add a constant that changes the compression ratio or the
   speed without a test or a benchmark that produced it. A comment names that source.
10. **One engine.** Do not add code paths that work differently on different machines. A faster
    path (SIMD, GPU) must give the same bits as the reference path. A test proves it.

## 2. How the code is organised

The dependency direction is one way. A crate may depend only on crates above it in this list.

```
taco-coder     range coder and frequency tables: integers in, bytes out (pure, no I/O)
hardshell      integer inference engine and the model-file reader (pure, no I/O except reads)
taco-format    the .taco container: header, blocks, checksums, raw fallback
taco           the library: tokenizer, model, coder and container wired together
taco-cli       the `taco` command: `taco fold` and `taco unfold`
```

- `taco-coder` and `hardshell` do not depend on each other. Both are kernel crates: no floats,
  no network, no clock, no randomness, no global state.
- `tools/` holds Python tools that run before publication: model conversion, table
  calculation, float baselines and measurements. Tools may use floats. Nothing in `crates/`
  calls a tool at run time.
- The CLI contains no logic. It translates arguments to library calls.

## 3. Rust conventions

**Language and safety**

- Edition 2024. The toolchain is pinned in `rust-toolchain.toml`.
- `unsafe` is denied in the workspace. A module that needs it (a SIMD kernel, for example)
  allows it at module level, and every block has a `// SAFETY:` comment. Unsafe code has its
  own tests and a Miri run.
- No panics in library code: no `unwrap`, `expect`, `panic!`, `todo!`, `unimplemented!`, or
  indexing that can go out of bounds on untrusted data. Lints enforce most of this. Tests may
  panic.
- Kernel crates add `#![deny(clippy::float_arithmetic)]` at the crate root. `scripts/check.sh`
  also refuses `f32` and `f64` in their `src/` directories.

**Errors**

- A library defines its own error enum with `thiserror`, marked `#[non_exhaustive]`, with one
  variant for each thing that a caller can act on. No `anyhow` in a library's public API.
- Binaries and tests may use `anyhow`.
- Errors carry offsets, lengths, ids and hashes. They never carry the user's data.
- Do not ignore an error. There is no `let _ =` on a `Result`.

**Types**

- Use newtypes for values with a meaning: `TokenId`, `Freq`, `ModelHash`, `ByteLen`. Do not
  pass a bare `u32` or `u64` across a function boundary when it has a meaning.
- Make invalid states impossible to represent: enums, validating constructors, private fields.
- Mark public enums and structs that can grow `#[non_exhaustive]`. Put `#[must_use]` on
  results and builders.

**Modules and functions**

- A module has one job. Its `//!` comment states that job. A file of more than about 500 lines
  must be divided.
- A function does one thing. More than six parameters means a parameter struct.
- Keep visibility as small as possible: private, then `pub(crate)`, then `pub`.

**Performance**

- State the budget before you optimise. Measure after. Benchmarks are in `benches/` and use
  `criterion`.
- A fast path is compared with the reference path in a test. The test fails on any difference.

**Documentation**

- Every public item has a doc comment. It says what the item does and what it guarantees.
- Say what a function costs when that is not clear ("one pass over the input").
- Comments explain why, not what. Numbers in comments are labelled `measured`, `published` or
  `est.`.
- Documents in `docs/`, `HANDOFF.md` and status files obey `docs/STYLE.md` (ASD-STE100).

## 4. Comment tags

`scripts/check-tags.sh` checks one format. The tag is in upper case, then a colon and a space.

| Tag | Meaning | Permitted on main |
|---|---|---|
| `TODO(phase-N): …` | Work for a later phase of the plan | Yes |
| `TODO(#123): …` | Work with an issue | Yes |
| `TODO(owner): …` | Short work that one person owns | Yes, for a short time |
| `FIXME: …` | Wrong now | No. Repair it or make it a tagged TODO |
| `HACK: …` | Works, but for a bad reason. Say what removes it | Yes, with the reason |
| `PERF: …` | A known cost, with the measurement or the bound | Yes |
| `SAFETY: …` | Why an unsafe block is sound | Necessary on every unsafe block |
| `INVARIANT: …` | A condition that the code around it needs | Yes |
| `DETERMINISM: …` | Why this operation gives the same bits on every machine | Yes |
| `NOTE: …` | Context that a reader can otherwise miss | Yes |
| `PLAN(§N): …` | Connects code to a section of the plan | Yes |
| `UNVERIFIED: …` | An assumption or an estimate that is not measured | Yes, until its phase gate |

A `TODO` without a parenthesis fails the check. So does `XXX`. Do not keep code that is in
comments.

## 5. Testing

| Kind | Where | Rule |
|---|---|---|
| Unit | Next to the code, `#[cfg(test)]` | Fast, no network, no clock |
| Property | `proptest` | For the coder, the container and every integer kernel |
| Round trip | `crates/taco/tests/` | Every input class in §11.2 comes back byte for byte |
| Conformance | `conformance/` | Reference vectors with SHA-256 of every output (§11) |
| Cross-machine | `scripts/xcheck` | Mac (ARM) and devbox (x86) give the same hashes |
| Quality | `tools/` | Bits per byte against the committed baseline. A change that makes it worse fails |

- A bug fix starts with a test that fails.
- Test names describe the behaviour: `decode_refuses_unknown_model_hash`, not `test_3`.
- Test data includes several languages, code, random bytes and empty input.
- No test depends on another test, on the clock, or on the order of a hash map.

## 6. Building and running

- **devbox** (Linux x86, RTX 3060) is the build and compute server. Heavy work runs there:
  workspace builds, GPU work, model conversion, benchmarks, quality runs.
  `scripts/dx <lane> <command>` copies the tree to the server and runs the command there. The
  lane names a private build directory, so parallel work does not wait on one cargo lock.
- **The Mac** (Apple Silicon, 8 GB) holds the code. It also runs the ARM half of the
  cross-machine check. `scripts/mx <command>` runs a light command on the Mac, one at a time.
- Before you call a change done, `cargo fmt` and `cargo clippy -- -D warnings` are clean, and
  `scripts/check.sh` passes.

```
scripts/dx main cargo test --workspace
scripts/dx main scripts/check.sh
scripts/mx cargo test -p taco-coder
```

`scripts/dx` refuses to run when the Windows C: drive has less than 15 GB free. The Linux disk
of devbox is a file on C:, and it stops Linux when C: is full. Only Rohit can compact it.

## 7. Dependencies

- Versions are in the root `Cargo.toml` under `[workspace.dependencies]`. A crate uses
  `name.workspace = true`.
- Before you add a dependency, ask: Is it maintained? Is its licence compatible with Apache-2.0
  or MIT? Does it use floats in a place that we call from the prediction path? Prefer the
  standard library.
- Kernel crates have almost no dependencies.

## 8. Changing behaviour

- Version control is the rollback. Do not add a switch that selects an old or a new
  implementation.
- A change to the model file, the tables, or the coder changes the compressed bytes. It needs a
  new format version and new conformance vectors.
- A change to the compression ratio comes with numbers: bits per byte before and after, on the
  quality corpus, in the commit message.

## 9. Commits

- Small and easy to read: one logical change per commit. Each commit builds and passes its
  tests.
- Conventional prefixes: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `build`, `chore`,
  with the crate as scope: `feat(coder): add 64-bit range coder`.
- The subject says what changed, in the imperative, in less than 72 characters. The body says
  why, what was measured, and which plan section it implements.
- Do not commit secrets, model weights, test corpora, build output or `.env`.

## 10. Working with agents

- An agent owns the crates or tool directories in its brief. It does not edit other files. It
  does not edit the root `Cargo.toml`, `Cargo.lock`, `CLAUDE.md` or another crate. It writes a
  request for such a change in its status file and in its report.
- An agent keeps `.agent-status/<name>.md` current: started at, current step, steps done, steps
  left, estimated minutes remaining, blockers, last updated. Less than 15 lines. It rewrites the
  file after each step and before a command that takes more than about two minutes.
- An agent builds in its own lane: `scripts/dx <name> cargo test -p <its crate>`.
- An agent does not commit. The lead integrates and commits after each step.
- If a format, a type or a rule is wrong or missing, say so. Do not work around it.

## 11. Definition of done

A change is done when:

- It compiles with no warnings, and `fmt` and `clippy -D warnings` are clean.
- Its tests exist and pass, including property tests for codecs and integer kernels.
- For HardShell and the coder: the cross-machine check gives the same hashes on the Mac and on
  devbox.
- Public items have documentation.
- No rule in section 1 is broken, no `FIXME` remains, and every `TODO` has a tag.
- What was not done is written down: in a tagged comment, the status file, or the report.
- Results are reported as they are. If a test fails, say so and show it. If something was
  skipped, say so.
