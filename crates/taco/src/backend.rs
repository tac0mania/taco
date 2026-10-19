//! The backend that runs HardShell: the CPU, an NVIDIA GPU (CUDA) or any other GPU (wgpu), and
//! the run-time selection between them (ADR 0016).
//!
//! DETERMINISM: every backend gives the same weights and the same coder intervals, bit for bit
//! (CLAUDE.md §1 rule 10, ADR 0008). Thus a container does not record the backend, and every
//! backend unfolds every container. Before a GPU is used, [`with_engine`] runs a self-test on
//! the loaded model: a few steps on the GPU must give the weights and the intervals of the CPU
//! session. A GPU that fails the test is not used.

use core::fmt;

use hardshell::TokenId;
use hardshell::gpt2::{Model, Session};
use hardshell_cuda::{CudaDevice, CudaModel};
use hardshell_gpu::{AdapterChoice, GpuDevice, GpuModel};
use taco_coder::{FreqBits, FreqTable, Symbol};

use crate::codec::FREQ_BITS;
use crate::error::{Result, TacoError};
use crate::lanes::{Answer, CudaLanes, GpuLanes, Lanes, Query};

/// The backend that the caller asks for (`taco --backend`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BackendChoice {
    /// The first backend that works: CUDA if an NVIDIA driver and a device of compute
    /// capability 8.0 or more are found and pass the self-test, else a discrete GPU of wgpu
    /// that passes the self-test, else the CPU. An integrated GPU is not used: the CPU path
    /// is faster there (measured on the Apple M2, ADR 0016).
    #[default]
    Auto,
    /// The CUDA backend (`hardshell-cuda`). An error if it does not work.
    Cuda,
    /// The portable GPU backend (`hardshell-gpu`, wgpu). An error if it does not work.
    Gpu,
    /// The CPU (`hardshell`, SIMD kernels and threads).
    Cpu,
}

/// The backend that ran.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Backend {
    /// The CPU.
    #[default]
    Cpu,
    /// An NVIDIA GPU through CUDA.
    Cuda,
    /// A GPU through wgpu (Vulkan, Metal, DX12 or OpenGL).
    Gpu,
}

impl Backend {
    /// The name of the backend, as `--backend` takes it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Gpu => "gpu",
        }
    }

    /// A [`TacoError::Backend`] of this backend.
    pub(crate) fn fault(self, reason: impl fmt::Display) -> TacoError {
        TacoError::Backend { backend: self, reason: reason.to_string() }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Why [`BackendChoice::Auto`] did not use a backend.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct BackendNote {
    /// The backend that was not used.
    pub backend: Backend,
    /// `true` if the machine does not have the backend (no driver, no device, no adapter, or
    /// an integrated GPU). `false` if the backend was there and failed (for example its
    /// self-test or its memory).
    pub missing: bool,
    /// What the driver or the self-test said.
    pub reason: String,
}

/// The model on the selected backend. [`crate::fold_with`] and [`crate::unfold_with`] run on
/// it. Make one with [`with_engine`].
pub struct Engine<'a> {
    kind: Kind<'a>,
    device: String,
    notes: Vec<BackendNote>,
}

enum Kind<'a> {
    Cpu,
    Cuda(&'a CudaModel<'a>),
    Gpu(&'a GpuModel),
}

impl fmt::Debug for Engine<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Engine")
            .field("backend", &self.backend())
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl Engine<'_> {
    /// The CPU engine. [`crate::fold`] and [`crate::unfold`] use it.
    #[must_use]
    pub fn cpu() -> Engine<'static> {
        Engine { kind: Kind::Cpu, device: "CPU".to_owned(), notes: Vec::new() }
    }

    /// The backend of this engine.
    #[must_use]
    pub fn backend(&self) -> Backend {
        match self.kind {
            Kind::Cpu => Backend::Cpu,
            Kind::Cuda(_) => Backend::Cuda,
            Kind::Gpu(_) => Backend::Gpu,
        }
    }

    /// The device name, for example "NVIDIA GeForce RTX 3060 Laptop GPU" or
    /// "Apple M2 (Metal, IntegratedGpu, ...)".
    #[must_use]
    pub fn device(&self) -> &str {
        &self.device
    }

    /// Why [`BackendChoice::Auto`] did not select a faster backend: one line for each backend
    /// that it tried and refused.
    #[must_use]
    pub fn notes(&self) -> &[BackendNote] {
        &self.notes
    }

    /// Makes the GPU slots for a job of `jobs` independent sequences. `rows` is the most rows
    /// of one pass on CUDA. `None` for the CPU engine.
    pub(crate) fn lanes(&self, jobs: usize, rows: usize) -> Result<Option<Box<dyn Lanes + '_>>> {
        match self.kind {
            Kind::Cpu => Ok(None),
            Kind::Cuda(m) => {
                let slots = cuda_slots(m, jobs, rows)?;
                Ok(Some(Box::new(CudaLanes::new(m, slots, rows)?)))
            }
            Kind::Gpu(m) => {
                // NOTE: halve the slots until the device accepts the buffers (the largest
                // binding of a device limits them).
                let mut slots = jobs.clamp(1, MAX_GPU_SLOTS);
                loop {
                    match GpuLanes::new(m, slots) {
                        Ok(l) => return Ok(Some(Box::new(l))),
                        Err(e) if slots == 1 => return Err(e),
                        Err(_) => slots /= 2,
                    }
                }
            }
        }
    }
}

/// The most slots of the wgpu backend. The portable path is fastest at about 32 slots
/// (measured: 1,311 to 1,388 tokens/s at 32, 1,138 to 1,166 at 128, RTX 3060 on GL,
/// `docs/research/2026-10-18-gpu-backend.md` section 3).
const MAX_GPU_SLOTS: usize = 32;

/// The most slots of the CUDA backend. 224 slots fill the 6 GB of the RTX 3060 (measured, the
/// research note section 4.1).
const MAX_CUDA_SLOTS: usize = 224;

/// The slots that fit in the free memory of the device: three quarters of the free memory,
/// less the work buffers of `rows` rows (about `20 * vocab` bytes each, est., `CudaBatch::new`),
/// divided by the KV cache of one slot.
fn cuda_slots(model: &CudaModel<'_>, jobs: usize, rows: usize) -> Result<usize> {
    let (free, _) = model.device().memory().map_err(|e| Backend::Cuda.fault(e))?;
    let free = u64::try_from(free).unwrap_or(u64::MAX);
    let vocab = u64::try_from(model.vocab_size()).unwrap_or(u64::MAX);
    let rows = u64::try_from(rows).unwrap_or(u64::MAX);
    let budget = (free / 4 * 3).saturating_sub(rows.saturating_mul(24).saturating_mul(vocab));
    let per_slot = model.cache_bytes_per_slot().max(1);
    let fit = usize::try_from(budget / per_slot).unwrap_or(usize::MAX);
    if fit == 0 {
        return Err(Backend::Cuda.fault("not enough free GPU memory for one slot"));
    }
    Ok(jobs.clamp(1, MAX_CUDA_SLOTS).min(fit))
}

/// Opens the backend that `choice` selects, loads `model` on it, checks it, and calls `f`
/// with the engine. `model_file` is the model file that `model` was loaded from; a GPU
/// backend loads it again into GPU memory.
///
/// With [`BackendChoice::Auto`], a backend that is missing or that fails its self-test is
/// skipped, and [`Engine::notes`] says why. With an explicit choice, that failure is the
/// result.
///
/// Cost: for a GPU, opening the device (the driver compiles the kernels the first time, some
/// hundred milliseconds), one upload of the model, and the self-test (6 steps on the CPU and
/// on the GPU).
///
/// # Errors
///
/// - [`TacoError::Backend`] if the chosen GPU backend does not work, or fails the self-test.
/// - [`TacoError::ModelHashMismatch`] if `model_file` is not the file of `model`.
/// - The errors of `f`.
pub fn with_engine<R>(
    choice: BackendChoice,
    model_file: &[u8],
    model: &Model,
    f: impl FnOnce(&Engine<'_>) -> Result<R>,
) -> Result<R> {
    let mut notes: Vec<BackendNote> = Vec::new();
    let mut f = Some(f);
    let mut call =
        |engine: &Engine<'_>| f.take().map_or(Err(TacoError::WorkerFailed), |f| f(engine));
    // A refusal: the error, and `true` if the machine does not have the backend.
    let note = |backend: Backend, (e, missing): (TacoError, bool)| BackendNote {
        backend,
        missing,
        reason: e.to_string(),
    };
    if matches!(choice, BackendChoice::Auto | BackendChoice::Cuda) {
        let tried = (|| {
            let dev = CudaDevice::open(0).map_err(|e| {
                let missing = matches!(e, hardshell_cuda::Error::NoDevice { .. });
                (Backend::Cuda.fault(e), missing)
            })?;
            let failed = |e: TacoError| (e, false);
            let gpu =
                CudaModel::load(&dev, model_file).map_err(|e| failed(Backend::Cuda.fault(e)))?;
            check_hash(model, gpu.hash()).map_err(failed)?;
            let mut engine =
                Engine { kind: Kind::Cuda(&gpu), device: dev.name().to_owned(), notes: Vec::new() };
            self_test(&engine, model).map_err(failed)?;
            engine.notes = std::mem::take(&mut notes);
            Ok(call(&engine))
        })();
        match (tried, choice) {
            (Ok(r), _) => return r,
            (Err((e, _)), BackendChoice::Cuda) => return Err(e),
            (Err(refusal), _) => notes.push(note(Backend::Cuda, refusal)),
        }
    }
    if matches!(choice, BackendChoice::Auto | BackendChoice::Gpu) {
        let tried = (|| {
            let dev = GpuDevice::open(&AdapterChoice::BestHardware).map_err(|e| {
                let missing = matches!(e, hardshell_gpu::Error::NoAdapter { .. });
                (Backend::Gpu.fault(e), missing)
            })?;
            if choice == BackendChoice::Auto && dev.adapter().device_type != DISCRETE {
                let reason = format!("{} is not a discrete GPU; the CPU is faster", dev.adapter());
                return Err((Backend::Gpu.fault(reason), true));
            }
            let failed = |e: TacoError| (e, false);
            let gpu =
                GpuModel::load(&dev, model_file).map_err(|e| failed(Backend::Gpu.fault(e)))?;
            check_hash(model, gpu.hash()).map_err(failed)?;
            let mut engine = Engine {
                kind: Kind::Gpu(&gpu),
                device: dev.adapter().to_string(),
                notes: Vec::new(),
            };
            self_test(&engine, model).map_err(failed)?;
            engine.notes = std::mem::take(&mut notes);
            Ok(call(&engine))
        })();
        match (tried, choice) {
            (Ok(r), _) => return r,
            (Err((e, _)), BackendChoice::Gpu) => return Err(e),
            (Err(refusal), _) => notes.push(note(Backend::Gpu, refusal)),
        }
    }
    let mut engine = Engine::cpu();
    engine.notes = notes;
    call(&engine)
}

/// The wgpu device type of a discrete GPU (`AdapterSummary::device_type`).
///
/// NOTE: `auto` uses wgpu only on a discrete GPU. The portable path has no prefill, so fold runs
/// one token of each window in each step. On the Apple M2 (integrated, Metal), 4,096 bytes of
/// `lcet10.txt` fold in 25.0 s with wgpu and in 7.1 s on the CPU with 6 threads (measured,
/// 2026-10-19, the Mac with memory pressure). `--backend gpu` still selects any adapter.
const DISCRETE: &str = "DiscreteGpu";

/// Refuses a GPU model whose file hash is not the hash of the CPU model.
fn check_hash(model: &Model, gpu: hardshell::ModelHash) -> Result<()> {
    if model.hash() == gpu {
        return Ok(());
    }
    let hash = |h: hardshell::ModelHash| taco_format::ModelHash::from_bytes(*h.as_bytes());
    Err(TacoError::ModelHashMismatch { container: hash(model.hash()), loaded: hash(gpu) })
}

/// The tokens of the self-test: fixed ids spread over the vocabulary (never `<|endoftext|>`).
fn test_tokens(model: &Model) -> Vec<TokenId> {
    let vocab = u32::try_from(model.vocab_size()).unwrap_or(u32::MAX).max(1);
    (0..SELF_TEST_STEPS)
        .map(|i| {
            let t = TokenId::new((i * 7919 + 464) % vocab);
            if t == model.eot_token() { TokenId::new((t.get() + 1) % vocab) } else { t }
        })
        .collect()
}

/// The number of tokens of the self-test.
const SELF_TEST_STEPS: u32 = 6;

/// Runs the first steps of a block on the GPU and on the CPU, and compares the weights of one
/// step and the coder answers (encode and decode) of every step. A difference is an error:
/// the GPU does not give the bits of the reference path (CLAUDE.md §1 rule 10).
fn self_test(engine: &Engine<'_>, model: &Model) -> Result<()> {
    let backend = engine.backend();
    let Some(mut lanes) = engine.lanes(1, 16)? else { return Ok(()) };
    let bits = FreqBits::new(FREQ_BITS)?;
    let tokens = test_tokens(model);
    let mut session = Session::new(model);
    let mut cpu_weights = vec![session.start_block()?.to_vec()];
    for &t in tokens.iter().take(tokens.len() - 1) {
        cpu_weights.push(session.step(t)?.to_vec());
    }
    let fail = |what: &str| backend.fault(format!("self-test: {what} differ from the CPU"));
    let mut input = model.eot_token();
    lanes.reset(0);
    for (i, (w, &t)) in cpu_weights.iter().zip(&tokens).enumerate() {
        let table = FreqTable::from_weights(w, bits)?;
        let want = table.code_interval(Symbol::new(t.get())).ok_or_else(|| fail("tokens"))?;
        // NOTE: step 0 also checks the full weights (the read back of the logs profile); the
        // later steps check the coder op, for the encoder and for the decoder.
        let got = if i == 0 {
            let gw = lanes.weights(&[(0, input)])?;
            if gw != *w {
                return Err(fail("the weights"));
            }
            Answer { token: t, interval: want }
        } else {
            let a = lanes.coded(&[(0, input)], &[Query::Encode(t)])?;
            a.first().copied().ok_or_else(|| fail("the answers"))?
        };
        if got.token != t || got.interval != want {
            return Err(fail("the encoder intervals"));
        }
        input = t;
    }
    // The decoder queries, each from a new context (`<|endoftext|>` only): the first target,
    // one inside the table, and the last target (the last symbol takes the remainder).
    let w = cpu_weights.first().ok_or_else(|| fail("the weights"))?;
    let table = FreqTable::from_weights(w, bits)?;
    for target in [0, table.total() / 3, table.total() - 1] {
        let (sym, iv) = table.symbol_at(target).ok_or_else(|| fail("targets"))?;
        lanes.reset(0);
        let a = lanes.coded(&[(0, model.eot_token())], &[Query::Decode(target)])?;
        let a = a.first().copied().ok_or_else(|| fail("the answers"))?;
        if a.token.get() != sym.get() || a.interval != iv {
            return Err(fail("the decoder intervals"));
        }
    }
    Ok(())
}
