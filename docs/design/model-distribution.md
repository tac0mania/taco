# Model distribution: how `taco` finds, downloads and verifies model files

Status: proposed, 2026-10-19 (the release agent). Owner: Rohit. This design obeys
`docs/STYLE.md`. It implements the owner decision of `docs/design/product-v1.md` §4 item 2.
The CLI agent implements it. Nothing in this document is implemented yet.

Decisions that Rohit must make are in §10. Requests to the lead are in §11.

## §1 Requirements

1. **The hash is the identity.** A model file is identified by its SHA-256 (`CLAUDE.md` §1
   rule 3). `taco` uses a model file only after it calculates the SHA-256 of the bytes that it
   loads and compares it with the expected hash.
2. **Unfold needs no user knowledge.** The container records the model hash and the profile
   (ADR 0012). `taco unfold` finds the model by that hash. The user does not name a model.
3. **One download, then offline.** If no model directory has the file, `taco` downloads it one
   time and keeps it (owner decision, product-v1 §4.2). After that, `taco` works without a
   network.
4. **Machines without a network.** The release has a model archive. A user copies it to the
   machine and imports the file (§6.4). A setting prevents every download (§5.5).
5. **A model file never changes.** A new model is a new file with a new hash. Old model files
   stay available, so that old `.taco` files still unfold (ADR 0014 point 7).
6. **No telemetry.** A download is one HTTPS GET of a public file. `taco` sends no other data.

## §2 The model registry

The `taco` binary contains a table of the published models: the registry. A new release of
`taco` can add entries. It never changes or removes an entry.

| Field | Type | Meaning |
|---|---|---|
| `name` | `&'static str` | A short name, for example `gpt2`. Unique in the registry. |
| `hash` | `ModelHash` | The SHA-256 of the file. |
| `size` | `ByteLen` (u64) | The size of the file in bytes. A download stops when it gets more bytes. |
| `file_name` | `&'static str` | The name of the published file, for example `gpt2.hsm`. |
| `urls` | `&'static [&'static str]` | The download locations, in order of preference (§7). |
| `profiles` | `&'static [Profile]` | The profiles that use this model by default in `fold`. |
| `licence` | `&'static str` | An SPDX-like name of the licence of the source weights. |
| `source` | `&'static str` | The source of the weights (a URL). |

The entries of version 1:

| `name` | `hash` | `size` | `profiles` | `licence` |
|---|---|---|---|---|
| `gpt2` | `aaa03eb96e9ad7957432168c8ac6b4a99ed55d57e637e4786a664b51001ba5c2` | 127,237,568 bytes (measured) | `text`, `logs` | `LicenseRef-OpenAI-GPT-2-Modified-MIT` |

NOTE: ADR 0012 permits a separate `logs` model (Route A). If the lead publishes one, it gets
its own entry, and `logs` moves to it. The `logs` model has an open licence question (§8.3).

## §3 Model directories

### §3.1 The layout

A model directory holds files with the name `<hash>.hsm`, where `<hash>` is the SHA-256 in 64
lower-case hexadecimal digits. Thus a lookup by hash needs no hash calculation of other files.
Other files in the directory are ignored, except the temporary files of §5.3.

### §3.2 The search order

`taco` looks for `<hash>.hsm` in these directories, in this order. It uses the first file that
it finds and that has the correct hash (§4). A file with a wrong hash is an error (§4.3); `taco`
does not continue to the next directory.

1. Each directory in `TACO_MODEL_PATH`. The separator is `:` on Linux and macOS and `;` on
   Windows (`std::env::split_paths`).
2. The user model directory (§3.3). `taco model fetch` and `taco model import` write here.
3. The directory `models` next to the `taco` program (`std::env::current_exe()`, with
   symbolic links resolved). This supports a portable directory: the program and the model
   archive in one place.
4. The system model directories (§3.3). `taco` only reads them.

### §3.3 The directories of each platform

| Platform | User model directory (default) | System model directories |
|---|---|---|
| Linux and other Unix | `$XDG_DATA_HOME/taco/models`, else `~/.local/share/taco/models` | `/usr/local/share/taco/models`, `/usr/share/taco/models` |
| macOS | `~/Library/Application Support/taco/models` | `/Library/Application Support/taco/models` |
| Windows | `%LOCALAPPDATA%\taco\models` | `%PROGRAMDATA%\taco\models` |

`TACO_MODEL_DIR` replaces the user model directory on every platform.

Reason for a data directory, not a cache directory: cache cleaners delete caches. A deleted
model file is a new download of 127 MB (measured size), or an error on a machine without a
network. A model file is data that old `.taco` files need.

`taco` creates the user model directory when it writes the first file. It creates no other
directory. If `taco` cannot find the home directory (`HOME` on Unix, `LOCALAPPDATA` on Windows)
and `TACO_MODEL_DIR` is not set, the user model directory does not exist. Then a fetch or an
import is an error (`StoreError::NoUserDir`), and the read-only search still works.

### §3.4 Environment variables

| Variable | Meaning |
|---|---|
| `TACO_MODEL` | A model file, as in version 0.1 (§4.1). Kept for compatibility. |
| `TACO_MODEL_DIR` | The user model directory. |
| `TACO_MODEL_PATH` | More directories, searched first. |
| `TACO_OFFLINE` | `1`: no download. Any other value, or not set: downloads are permitted. |
| `TACO_MODEL_URL` | A base URL. It replaces the URLs of the registry: `taco` downloads `<TACO_MODEL_URL>/<hash>.hsm`. For a mirror on a private network. |
| `HTTPS_PROXY`, `NO_PROXY` | The usual proxy settings. The HTTP client reads them. |

## §4 Which model a command uses

### §4.1 `taco fold`

1. With `--model <value>`: if `<value>` is a path to a file, `taco` uses that file. Else, if it
   is a registry name or a hash (64 digits, or a unique prefix of 8 or more digits), `taco`
   uses the store (§3.2, then §5). Else it is an error.
2. Else, with `TACO_MODEL` set: `taco` uses that file.
3. Else: `taco` uses the registry model of the profile. With `--profile auto`, `taco` needs the
   models of `text` and `logs`. In version 1 both are `gpt2`.

A model file given by a path is not checked against the registry. A customer model (for
example a fine-tune) thus works without a registry entry. Its hash goes into the container as
usual.

### §4.2 `taco unfold`

1. `taco` reads the header of the container and gets the model hash. A container with only raw
   blocks needs no model.
2. With `--model <value>` or `TACO_MODEL`: `taco` uses that file. If its hash is not the hash of
   the container, that is an error (`TacoError::ModelHashMismatch`, as now). `taco` does not
   then search the store.
3. Else: `taco` finds `<hash>.hsm` in the store (§3.2).
4. If no directory has it and the hash is in the registry: `taco` downloads it (§5), unless
   downloads are off. If the hash is not in the registry, `taco` stops with an error that names
   the hash and says: "import the model file with `taco model import <file>`".

### §4.3 Verification before use

`taco` verifies the bytes that it loads, not the file at an earlier time:

1. `taco` reads the full file into memory (it does this now for `Model::load`).
2. It calculates the SHA-256 of those bytes and compares it with the expected hash (from the
   container, the registry, or the file name in the store).
3. On a mismatch it stops with `StoreError::HashMismatch` and names the path. It never uses the
   file. It does not delete the file: the user decides (`taco model verify` reports it).

The cost is one SHA-256 pass over 127 MB: about 0.1 to 0.5 s (est.). `Model::load` already
calculates this hash (`hardshell` `hsm/reader.rs`), so the store must pass the bytes and the
expected hash to one function and not hash two times.

## §5 Download

### §5.1 When

- `taco model fetch` downloads. It is the explicit command.
- `taco fold` and `taco unfold` download a missing registry model one time, unless downloads
  are off. Before the download, `taco` prints one line to stderr:
  `taco: downloading the model gpt2 (aaa03eb9…a5c2, 127,237,568 bytes) from <url>`.
- No other command downloads.

### §5.2 From where

`taco` tries the URLs of the registry entry in order. With `TACO_MODEL_URL`, it tries only
`<TACO_MODEL_URL>/<hash>.hsm`. A failure of one URL (a connection error, an HTTP status other
than 200, a wrong size, a wrong hash) moves to the next URL. When all fail, `taco` stops with
`StoreError::Download` and lists each URL with its reason.

### §5.3 How

1. `taco` creates the user model directory if necessary.
2. It writes to a temporary file in that directory: `.<hash>.<process id>.partial`. Thus the
   rename in step 5 stays on one file system.
3. It calculates the SHA-256 while it writes. It stops when it gets more bytes than the
   registry size.
4. At the end it compares the size and the hash. On a mismatch it deletes the temporary file.
5. It flushes the file to the disk (`File::sync_all`) and renames it to `<hash>.hsm`. If the
   target exists (another `taco` process was faster), `taco` deletes its own temporary file and
   uses the existing file after the check of §4.3.
6. A temporary file older than 24 hours is deleted by the next fetch (an interrupted download).

Version 1 does not resume a download. A new download starts again (est. 10 to 60 s for
127 MB on a normal connection).

### §5.4 Progress and limits

- Progress goes to stderr when stderr is a terminal: the bytes received, the total, and the
  rate, at most 4 times a second. Without a terminal, `taco` prints only the start line and the
  end line.
- The connection timeout is 30 s. A read that gets no bytes for 60 s stops the download.
- The client follows at most 5 redirects, and only to HTTPS URLs. GitHub release assets
  redirect to a storage host.
- The User-Agent is `taco/<version>`.

### §5.5 Offline

`--offline` (on `fold`, `unfold` and `model fetch`) or `TACO_OFFLINE=1` prevents every
download. A missing model is then the error `StoreError::Offline`, which names the hash, the
model name, and the directories that `taco` searched.

## §6 The `taco model` commands (the exact interface)

All output for a person goes to stderr, as now. Output for a script (the lists, the path) goes
to stdout. A hash in a list is printed in full; a message may print the short form
`aaa03eb9…a5c2` (8 and 4 digits).

A `<model>` argument is a registry name (`gpt2`), a full hash, or a unique hash prefix of 8 or
more digits. `verify` also accepts a path.

### §6.1 `taco model list`

```
taco model list [--all]
```

Prints one line for each registry model, and with `--all` also one line for each other
`.hsm` file in the model directories:

```
<name>\t<hash>\t<size in bytes>\t<state>\t<path or ->
```

`<state>` is `present` (the file exists in a model directory; not hashed), `missing`, or
`unknown` (a file that is not in the registry; `<name>` is `-`). Exit status 0.

### §6.2 `taco model fetch`

```
taco model fetch [<model>...] [--all] [--offline]
```

- No argument: the models of the profiles `text` and `logs`. `--all`: every registry model.
- A model that a directory already has is not downloaded. `taco` verifies its hash (§4.3).
- Prints one line to stderr for each model: `taco: gpt2 aaa03eb9…a5c2: present at <path>` or
  `taco: gpt2 aaa03eb9…a5c2: downloaded to <path> (127,237,568 bytes, 12.3 s)`.
- Exit status 0 when each model is present and verified, 1 when one failed.

### §6.3 `taco model verify`

```
taco model verify [<model>|<path>...]
```

- No argument: every `.hsm` file in every model directory.
- For each file: `ok`, `WRONG HASH` (the file name says one hash and the content has another;
  the content hash is printed), `not in the registry` (a path with a hash that the registry does
  not have; this is not a failure), or `missing`.
- One line for each file to stdout: `<state>\t<hash of the content>\t<path>`.
- Exit status 0 when no file has a wrong hash and no named model is missing, else 1.

### §6.4 `taco model import`

```
taco model import <path> [--expect <hash>]
```

- `taco` hashes the file. With `--expect`, a different hash is an error.
- It copies the file into the user model directory as `<hash>.hsm`, with the temporary-file
  steps of §5.3. A file that is already there is not copied again.
- A file that is not a valid model file is refused: `taco` parses the header with `hardshell`
  before it copies the file.
- It prints `taco: imported <hash> (<name or "not in the registry">) to <path>`. Exit status 0.

### §6.5 `taco model path`

```
taco model path <model>
```

Prints the path of the file that §3.2 finds, to stdout. It does not hash the file and does not
download. Exit status 1 if no directory has the file.

### §6.6 Changes to `fold`, `unfold` and the top level

- `--model <value>` accepts a path, a registry name or a hash (§4.1).
- `--offline` on `fold` and `unfold`.
- `taco --version` (and `-V`) prints:
  ```
  taco 0.1.0 (<commit>, <target>)
  container: reads versions 1, 2; writes version 2
  models: gpt2 aaa03eb9…a5c2
  ```
  The commit comes from the build (`TACO_COMMIT` of `scripts/release/build.sh`, through
  `option_env!`; `unknown` if not set).
- The exit status 2 stays for argument errors.

## §7 Hosting

### §7.1 The facts

- GitHub: "Each file included in a release must be under 2 GiB" and "There is no limit on the
  total size of a release, nor bandwidth usage" (published, GitHub documentation "About
  releases", read 2026-10-19). `gpt2.hsm` is 127 MB (measured), so the file limit is not a
  problem for version 1. A 3B to 4B model (anymodel plan) at 8 bits is about 3 to 4 GB (est.),
  more than the limit: it needs two parts or another host.
- The repository `tac0mania/taco` is private. A release asset of a private repository needs a
  token with read access to the repository. Thus a public user cannot download it, and `taco`
  must never contain a token.
- The hash makes the host untrusted: a wrong or changed file fails the check of §4.3. Thus any
  host is safe for integrity. The host matters only for availability and for the licence.

### §7.2 The options

| Option | Public download | Cost (published or est.) | Notes |
|---|---|---|---|
| A. Make `tac0mania/taco` public; model files as release assets | Yes | No cost | Publishes the source code. An owner decision. |
| B. A separate public repository, for example `tac0mania/taco-models`, with releases that hold only model files | Yes | No cost | The source stays private. Release assets are permanent while the repository exists. |
| C. Hugging Face Hub, a public model repository | Yes | No cost for public models (est.) | The usual place for model weights. A second URL in the registry. |
| D. Object storage with a CDN (for example Cloudflare R2) | Yes | Storage cost; R2 has no egress fee (published by Cloudflare; amount UNVERIFIED) | Needs an account and a domain. |

### §7.3 Recommendation

Use B as the first URL and C as the second URL. Do not use A only for the model files. Each
registry entry has both URLs, so one host can fail.

The URL form for B: `https://github.com/tac0mania/taco-models/releases/download/<name>-<hash prefix of 8>/<hash>.hsm`.
The asset name is the hash, so that `TACO_MODEL_URL` mirrors use the same name.

## §8 Licences

### §8.1 GPT-2 small (the model of version 1)

- OpenAI published the GPT-2 code and weights under the "Modified MIT License", Software
  Copyright (c) 2019 OpenAI (published, `https://github.com/openai/gpt-2/blob/master/LICENSE`,
  read 2026-10-19). The Hugging Face model card of `openai-community/gpt2` gives the licence
  `mit` (published, read 2026-10-19).
- The licence permits use, copy, modification, distribution, sublicence and sale. It is
  compatible with the Apache-2.0 licence of Taco for distribution together (est.; not legal
  advice).
- Condition: the copyright notice and the permission notice must be included in all copies or
  substantial portions. `gpt2.hsm` is a converted copy of the weights (and of the tokenizer
  files). Thus every distribution of `gpt2.hsm` includes the licence text: the model archive has
  `LICENSE-GPT2.txt` (`scripts/release/model-archive.sh`), and the model hosting page shows it.
- The licence also says that the notices "need not be included with content created by the
  Software". A `.taco` file is the user's data, coded with the help of the model. It needs no
  notice (est.; not legal advice).
- The licence has a request, not a condition: "use GPT-2 responsibly and clearly indicate your
  content was created using GPT-2". Taco does not generate content. The README and the man page
  say that Taco uses GPT-2.

### §8.2 What each release contains

| Release file | Licence files in it |
|---|---|
| `taco-<version>-<target>.tar.gz` / `.zip` | `LICENSE` (Apache-2.0), `THIRD-PARTY-LICENSES.txt` (each crate in the binary, made by `scripts/release/third_party.py`) |
| `taco-model-gpt2-aaa03eb9.tar` | `LICENSE-GPT2.txt` (the OpenAI licence), `MODEL.txt` (source, hash, conversion) |

### §8.3 The `logs` model of Route A (not in version 1 yet)

- The base model distilgpt2: the model card gives `apache-2.0` (published, Hugging Face
  `distilbert/distilgpt2`, read 2026-10-19).
- The training data Loghub: its LICENSE says that the datasets "are freely available for
  research or academic work", with a condition to cite the paper (published,
  `https://github.com/logpai/loghub`, read 2026-10-19). It does not say that commercial use is
  permitted. A model that is fine-tuned on Loghub and that Taco sells or ships with a product is
  thus a licence risk. Rohit must decide (§10 item 4) before such a model is published.

## §9 The library API (for the CLI agent)

### §9.1 Where the code goes

The CLI contains no logic (`CLAUDE.md` §2). The store logic goes into a new library crate,
`taco-store`, below `taco-cli` and beside `taco`:

```
taco-store   model registry, model directories, verification, import, download (I/O)
```

- `taco-store` depends on `sha2` and `thiserror`, and on an HTTP client (§9.4) behind the
  default feature `fetch`. It does not depend on `hardshell` or `taco`. The check "the file is
  a valid model file" (§6.4) is a function that the caller gives, so the crate stays small.
- `taco-cli` depends on `taco` and `taco-store`. `taco` does not depend on `taco-store`.
- `taco` needs one new function: the model hash of a container from the header only (§9.3).
- This is a new crate and new dependencies: the lead must approve it, and
  `docs/ARCHITECTURE.md` gets a row (§11).

### §9.2 `taco-store` types and functions

```rust
/// The SHA-256 of a model file. 32 bytes. Display: 64 lower-case hex digits.
pub struct ModelHash([u8; 32]);
impl ModelHash {
    pub const fn from_bytes(b: [u8; 32]) -> Self;
    pub const fn as_bytes(&self) -> &[u8; 32];
    /// Parses 64 hex digits. A prefix is not accepted here (see `Store::resolve`).
    pub fn from_hex(s: &str) -> Result<Self, StoreError>;
    /// "aaa03eb9…a5c2".
    pub fn short(&self) -> String;
}

/// One published model (§2).
#[non_exhaustive]
pub struct ModelInfo {
    pub name: &'static str,
    pub hash: ModelHash,
    pub size: u64,
    pub file_name: &'static str,
    pub urls: &'static [&'static str],
    pub profiles: &'static [ProfileName],   // ProfileName::Text | ProfileName::Logs
    pub licence: &'static str,
    pub source: &'static str,
}

/// The registry of this build. Sorted by name.
pub fn registry() -> &'static [ModelInfo];
pub fn by_name(name: &str) -> Option<&'static ModelInfo>;
pub fn by_hash(hash: &ModelHash) -> Option<&'static ModelInfo>;
/// The registry model of a profile (§4.1 step 3).
pub fn for_profile(profile: ProfileName) -> &'static ModelInfo;

/// Where the model files are (§3).
#[must_use]
pub struct Store { /* search: Vec<PathBuf>, user: Option<PathBuf>, policy: NetPolicy */ }

#[non_exhaustive]
pub enum NetPolicy { Allowed, Offline }

impl Store {
    /// The directories of §3.2 and §3.3 from the environment of the process, and the policy
    /// from `TACO_OFFLINE`. It reads the environment once; it does no I/O.
    pub fn from_env() -> Store;
    /// Explicit directories (tests): `user` is the write directory, `read` the others, in order.
    pub fn new(user: Option<PathBuf>, read: Vec<PathBuf>, policy: NetPolicy) -> Store;
    pub fn with_policy(self, policy: NetPolicy) -> Store;
    /// The directories in search order, with their kind (for messages and `model list`).
    pub fn dirs(&self) -> Vec<(DirKind, &Path)>;          // DirKind::{Path, User, Program, System}
    /// A name, a full hash or a unique prefix of 8 or more hex digits -> a hash (§6).
    pub fn resolve(&self, model: &str) -> Result<ModelHash, StoreError>;
    /// The first `<hash>.hsm` of §3.2. No hash calculation, no download.
    pub fn locate(&self, hash: &ModelHash) -> Option<PathBuf>;
    /// Locates the file, downloads it if it is missing and permitted (§5), reads it, and
    /// verifies the bytes (§4.3). The bytes are the bytes that the caller loads.
    pub fn load(&self, hash: &ModelHash, fetcher: &dyn Fetch, progress: &mut dyn Progress)
        -> Result<LoadedFile, StoreError>;
    /// §6.2 for one model. Returns the path and whether it downloaded.
    pub fn fetch(&self, info: &ModelInfo, fetcher: &dyn Fetch, progress: &mut dyn Progress)
        -> Result<Fetched, StoreError>;
    /// §6.4. `check` parses the header (the CLI gives `|b| hardshell::... .map(|_| ())`).
    pub fn import(&self, path: &Path, expect: Option<ModelHash>,
                  check: &dyn Fn(&[u8]) -> Result<(), String>) -> Result<Imported, StoreError>;
    /// §6.1: every `.hsm` file in the directories, with its hash from the file name.
    pub fn list(&self) -> Result<Vec<Entry>, StoreError>;
}

/// The bytes of a verified model file, and where they came from.
pub struct LoadedFile { pub bytes: Vec<u8>, pub hash: ModelHash, pub path: PathBuf, pub downloaded: bool }

/// Hashes a file with one read pass (§6.3).
pub fn hash_file(path: &Path) -> Result<ModelHash, StoreError>;
/// Hashes bytes and compares with `expected`. The one place that decides "verified".
pub fn verify_bytes(bytes: &[u8], expected: &ModelHash) -> Result<(), StoreError>;

/// A download source. The `fetch` feature gives `HttpFetch`; tests give a fake.
pub trait Fetch {
    /// GETs `url` into `sink`. Stops with `FetchError::TooLarge` after `limit` bytes.
    fn get(&self, url: &str, limit: u64, sink: &mut dyn std::io::Write,
           progress: &mut dyn Progress) -> Result<u64, FetchError>;
}
/// The fetcher that always fails with `FetchError::Disabled` (for `NetPolicy::Offline` and
/// builds without the `fetch` feature).
pub struct NoFetch;
#[cfg(feature = "fetch")]
pub struct HttpFetch { /* the client, timeouts of §5.4, User-Agent */ }
#[cfg(feature = "fetch")]
impl HttpFetch { pub fn new(user_agent: &str) -> Result<Self, StoreError>; }

/// Progress of a download (bytes so far, total). The CLI prints it; the library never prints.
pub trait Progress { fn update(&mut self, done: u64, total: u64); }
```

A real `ModelHash` already exists in `hardshell` and in `taco-format` (the same 32 bytes).
`taco-store` defines its own, as `taco-format` does, so that it depends on neither. The CLI
converts with `from_bytes(*h.as_bytes())`.

### §9.3 New function in `taco`

```rust
/// The model hash in the header of a container, or `None` for a container that needs no model
/// (only raw blocks). Reads the header only; no block is decoded.
pub fn container_model(container: &[u8]) -> Result<Option<ModelHash>>;   // TacoError::Format on a bad header
```

### §9.4 The HTTP client

Proposal: `ureq` (MIT OR Apache-2.0) with `rustls` (Apache-2.0 OR ISC OR MIT), the `ring`
crypto provider (Apache-2.0 AND ISC), and `rustls-platform-verifier` (MIT OR Apache-2.0), so
that `taco` uses the certificate store of the system and works behind a company proxy with its
own certificate authority. The versions are UNVERIFIED: the lead checks them before the
change to the root `Cargo.toml`.

Checks for the lead (`CLAUDE.md` §7): all are maintained (est.); the licences are compatible;
none is in the prediction path. `ring` compiles C and assembly with the C compiler that
`zstd-sys` already needs. The build of `ring` on `x86_64-pc-windows-gnu` with w64devkit is
UNVERIFIED.

The alternative: no HTTP crate, and `taco` runs the system `curl` program (macOS, Windows 10
version 1803 and later, and most Linux systems have it). This adds no dependency, but a
minimal Linux container can lack `curl`, and errors are harder to report. Not recommended.

### §9.5 Errors

```rust
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// No model directory has the file, and downloads are off.
    #[error("the model {hash} is not in a model directory, and downloads are off")]
    Offline { hash: ModelHash, searched: Vec<PathBuf> },
    /// No model directory has the file, and the registry does not know the hash.
    #[error("the model {hash} is not in a model directory and not in the registry: import it with `taco model import <file>`")]
    Unknown { hash: ModelHash, searched: Vec<PathBuf> },
    /// The bytes of a file do not have the expected hash.
    #[error("{path}: the SHA-256 is {actual}, not {expected}")]
    HashMismatch { path: PathBuf, expected: ModelHash, actual: ModelHash },
    /// A download gave a different number of bytes than the registry size.
    #[error("{url}: {actual} bytes, not {expected}")]
    SizeMismatch { url: String, expected: u64, actual: u64 },
    /// Every URL failed. One reason for each URL, in order.
    #[error("the download of the model {hash} failed: {}", .attempts.len())]
    Download { hash: ModelHash, attempts: Vec<(String, FetchError)> },
    /// The name is not in the registry, or the prefix matches no model or more than one.
    #[error("no model {0:?}: give a registry name, a hash, or a unique hash prefix of 8 or more digits")]
    NoSuchModel(String),
    /// A hash text is not 64 hex digits.
    #[error("not a SHA-256: {0:?}")]
    BadHash(String),
    /// `import` refused the file: it is not a model file.
    #[error("{path}: not a model file: {reason}")]
    NotAModel { path: PathBuf, reason: String },
    /// No user model directory (no home directory and no TACO_MODEL_DIR).
    #[error("no user model directory: set TACO_MODEL_DIR")]
    NoUserDir,
    /// A file operation failed.
    #[error("{path}: {source}")]
    Io { path: PathBuf, source: std::io::Error },
}

#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    #[error("downloads are off")] Disabled,
    #[error("HTTP status {0}")] Status(u16),
    #[error("more than {0} bytes")] TooLarge(u64),
    #[error("not an HTTPS URL")] NotHttps,
    #[error("timeout")] Timeout,
    #[error("{0}")] Transport(String),   // the text of the client error; no user data
    #[error("write: {0}")] Write(std::io::Error),
}
```

The errors carry paths, URLs, hashes and sizes. They never carry the content of a file
(`CLAUDE.md` §3).

### §9.6 Tests that the CLI agent writes

- `resolve`: a name, a full hash, a prefix of 8, a prefix of 7 (refused), an ambiguous prefix
  (two files in a test directory), an unknown name.
- The search order: the same hash in two directories; `TACO_MODEL_PATH` wins; the user
  directory wins over the program directory and the system directory.
- `load` refuses a file whose name is one hash and whose content is another
  (`load_refuses_a_file_with_a_wrong_hash`), and does not search the next directory.
- `load` with `NetPolicy::Offline` and no file gives `StoreError::Offline` and never calls the
  fetcher (a fake fetcher that counts calls).
- Download with a fake fetcher: a correct file is renamed into place; a wrong hash, a short
  file and a long file (`TooLarge`) leave no file and no temporary file; the second URL is used
  when the first fails; two loads at the same time give one file.
- `import`: a valid file, a file that `check` refuses, `--expect` with a wrong hash, a file
  that is already there.
- The CLI: `taco unfold x.taco` with no `--model` finds the model in `TACO_MODEL_DIR`;
  `taco model verify` exit status 1 on a wrong file; `taco model list` output columns;
  `taco --version` names the registry hashes.
- No test uses the network. The HTTP fetcher has one test that is `#[ignore]` and runs by hand.

## §10 Decisions for Rohit

1. **The public host of the model files** (§7): a separate public repository
   (`tac0mania/taco-models`, recommended), Hugging Face, the main repository made public, or
   object storage. Until a host exists, the registry has no URL, `taco model fetch` cannot
   work, and users get the file from the model archive.
2. **Automatic download in `fold` and `unfold`** (§5.1). Product-v1 §4.2 says yes. The
   alternative is to download only with `taco model fetch`. Recommendation: keep the
   automatic download, with the one-line message and `TACO_OFFLINE`.
3. **A new crate and new dependencies** (§9.1, §9.4): `taco-store` with `ureq`, `rustls`,
   `ring`, `rustls-platform-verifier`.
4. **The licence of a log model fine-tuned on Loghub** (§8.3): research use only, a licence
   from the Loghub authors, or a fine-tune on data with a commercial licence.
5. **The legal check of the GPT-2 licence for a commercial product** (§8.1). This design is not
   legal advice.

## §11 Requests to the lead

- Approve the crate `taco-store` and its dependencies; add the rows to `docs/ARCHITECTURE.md`
  and the versions to the root `Cargo.toml`.
- Ask the CLI agent to implement §6 and §9, and to change the usage text in
  `crates/taco-cli/src/args.rs`. Then remove the PLANNED marks in `docs/man/taco.1`.
- Make an ADR for the model directories and the download (this document), because other
  agents must obey it.
