//! The GPU device: adapter choice, buffers, shader modules, and reads back to the host.
//!
//! wgpu gives one API over Vulkan (NVIDIA, AMD, Intel, and the Mesa drivers), Metal (Apple),
//! DX12 (Windows) and OpenGL. The user needs only the GPU driver (ADR 0008).

use std::sync::{Arc, Mutex, PoisonError, mpsc};

use crate::block_on::block_on;
use crate::error::{Error, Result};

/// Which adapter to use.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum AdapterChoice {
    /// The best hardware adapter: a discrete GPU first, then an integrated GPU, then any other
    /// GPU. A software (CPU) adapter is never selected.
    #[default]
    BestHardware,
    /// The first adapter whose name or backend contains this text, case-insensitive. This can
    /// select a software adapter (for example "llvmpipe").
    NameContains(String),
}

/// A description of the adapter that a device uses.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct AdapterSummary {
    /// The adapter name, for example "Apple M2".
    pub name: String,
    /// The backend: Vulkan, Metal, Dx12 or Gl.
    pub backend: String,
    /// Discrete, integrated, virtual, CPU or other.
    pub device_type: String,
    /// The driver name and version, as the driver gives them.
    pub driver: String,
}

impl std::fmt::Display for AdapterSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({}, {}, {})", self.name, self.backend, self.device_type, self.driver)
    }
}

/// A GPU device and its queue. A clone shares the same device.
#[derive(Clone)]
pub struct GpuDevice {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    summary: AdapterSummary,
    limits: wgpu::Limits,
    errors: Arc<Mutex<Vec<String>>>,
}

impl std::fmt::Debug for GpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDevice").field("adapter", &self.summary).finish_non_exhaustive()
    }
}

fn summary(a: &wgpu::Adapter) -> AdapterSummary {
    let i = a.get_info();
    AdapterSummary {
        name: i.name,
        backend: format!("{:?}", i.backend),
        device_type: format!("{:?}", i.device_type),
        driver: format!("{} {}", i.driver, i.driver_info).trim().to_owned(),
    }
}

/// `true` if the adapter runs compute shaders. The kernels need them; some OpenGL adapters do
/// not have them.
fn has_compute(a: &wgpu::Adapter) -> bool {
    a.get_downlevel_capabilities().flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
}

fn rank(t: wgpu::DeviceType) -> Option<u8> {
    match t {
        wgpu::DeviceType::DiscreteGpu => Some(0),
        wgpu::DeviceType::IntegratedGpu => Some(1),
        wgpu::DeviceType::Other => Some(2),
        wgpu::DeviceType::VirtualGpu => Some(3),
        wgpu::DeviceType::Cpu => None,
    }
}

/// All adapters that wgpu finds on this machine. Cost: the drivers enumerate their devices.
#[must_use]
pub fn adapters() -> Vec<AdapterSummary> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    block_on(instance.enumerate_adapters(wgpu::Backends::all())).iter().map(summary).collect()
}

impl GpuDevice {
    /// Opens a device on the adapter that `choice` selects. The device gets the full limits of
    /// the adapter. Cost: the driver makes a device (some milliseconds).
    ///
    /// # Errors
    /// [`Error::NoAdapter`] if no adapter agrees with `choice` ([`AdapterChoice::BestHardware`]
    /// skips adapters with no compute shaders); [`Error::Device`] if the chosen adapter has no
    /// compute shaders or refuses the device.
    pub fn open(choice: &AdapterChoice) -> Result<Self> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let all = block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        let picked = match choice {
            AdapterChoice::BestHardware => all
                .into_iter()
                .filter(has_compute)
                .filter_map(|a| rank(a.get_info().device_type).map(|r| (r, a)))
                .min_by_key(|(r, _)| *r)
                .map(|(_, a)| a),
            AdapterChoice::NameContains(text) => {
                let t = text.to_lowercase();
                all.into_iter().find(|a| {
                    let s = summary(a);
                    s.name.to_lowercase().contains(&t) || s.backend.to_lowercase().contains(&t)
                })
            }
        };
        let adapter = picked.ok_or_else(|| Error::NoAdapter { choice: format!("{choice:?}") })?;
        if !has_compute(&adapter) {
            return Err(Error::Device {
                reason: format!("{} has no compute shaders", summary(&adapter)),
            });
        }
        let limits = adapter.limits();
        // NOTE: timestamps are only for measurements (`GpuBatch::profile_step`); they do not
        // change a result. The device asks for them when the adapter has them.
        let timestamps =
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
        let features = adapter.features() & timestamps;
        let desc = wgpu::DeviceDescriptor {
            label: Some("hardshell-gpu"),
            required_features: features,
            required_limits: limits.clone(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::default(),
        };
        let (device, queue) = block_on(adapter.request_device(&desc))
            .map_err(|e| Error::Device { reason: e.to_string() })?;
        let errors = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&errors);
        // NOTE: the default handler of wgpu panics. This handler keeps the messages, and every
        // operation of this crate checks them, so a GPU error is an `Error`, never a panic.
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            sink.lock().unwrap_or_else(PoisonError::into_inner).push(e.to_string());
        }));
        Ok(Self { device, queue, summary: summary(&adapter), limits, errors })
    }

    /// The adapter of this device.
    #[must_use]
    pub fn adapter(&self) -> &AdapterSummary {
        &self.summary
    }

    /// `true` if the device can write timestamps inside a compute pass.
    #[must_use]
    pub fn has_timestamps(&self) -> bool {
        self.device.features().contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
    }

    /// Picoseconds for each timestamp tick, rounded down. wgpu gives the period as an IEEE 754
    /// single; this decodes its bits with integer operations, so the crate has no float
    /// arithmetic.
    pub(crate) fn timestamp_period_ps(&self) -> u64 {
        let bits = self.queue.get_timestamp_period().to_bits();
        let exp = (bits >> 23) & 0xff;
        let mant = u128::from((bits & 0x7f_ffff) | if exp == 0 { 0 } else { 0x80_0000 });
        // value = mant * 2^(exp - 150) nanoseconds; times 1000 for picoseconds.
        let ps = mant * 1000;
        let v = if exp >= 150 { ps << (exp - 150).min(40) } else { ps >> (150 - exp).min(127) };
        u64::try_from(v).unwrap_or(u64::MAX)
    }

    /// The largest storage buffer that one binding can have, in bytes.
    #[must_use]
    pub fn max_binding_bytes(&self) -> u64 {
        self.limits.max_storage_buffer_binding_size.min(self.limits.max_buffer_size)
    }

    pub(crate) fn max_workgroups(&self) -> u32 {
        self.limits.max_compute_workgroups_per_dimension
    }

    /// Gives the first GPU error since the last check, if there is one.
    pub(crate) fn check(&self) -> Result<()> {
        let mut e = self.errors.lock().unwrap_or_else(PoisonError::into_inner);
        if e.is_empty() {
            return Ok(());
        }
        let reason = e.join("; ");
        e.clear();
        Err(Error::Gpu { reason })
    }

    /// Checks that a buffer of `bytes` fits in one binding.
    pub(crate) fn check_binding(&self, what: &'static str, bytes: u64) -> Result<()> {
        let limit = self.max_binding_bytes();
        if bytes > limit {
            return Err(Error::DeviceLimit { what, needed: bytes, limit });
        }
        Ok(())
    }

    /// A storage buffer with these bytes. An empty slice gives a buffer of 16 zero bytes,
    /// because a binding cannot be empty.
    pub(crate) fn storage(&self, label: &str, bytes: &[u8]) -> Result<wgpu::Buffer> {
        use wgpu::util::DeviceExt;
        self.check_binding("a storage buffer", bytes.len() as u64)?;
        let pad;
        let contents = if bytes.len() < 16 {
            pad = {
                let mut v = bytes.to_vec();
                v.resize(16, 0);
                v
            };
            &pad[..]
        } else {
            bytes
        };
        Ok(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        }))
    }

    /// A storage buffer of `bytes` zero bytes (at least 16). wgpu sets new buffers to zero.
    pub(crate) fn zeroed(&self, label: &str, bytes: u64) -> Result<wgpu::Buffer> {
        self.check_binding("a storage buffer", bytes)?;
        Ok(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: round4(bytes.max(16)),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    }

    /// A uniform buffer with these words (padded to a multiple of 16 bytes).
    pub(crate) fn uniform(&self, label: &str, words: &[u32]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        let mut bytes = words_to_bytes(words);
        bytes.resize(bytes.len().div_ceil(16).max(1) * 16, 0);
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: &bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// A buffer that the host can map to read.
    pub(crate) fn staging(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: round4(bytes.max(16)),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Compiles a WGSL module from parts (the parts are joined in order).
    pub(crate) fn module(&self, label: &str, parts: &[&str]) -> Result<wgpu::ShaderModule> {
        let source = parts.join("\n");
        let m = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        self.check()?;
        Ok(m)
    }

    /// A compute pipeline with the layout that the shader declares.
    pub(crate) fn pipeline(
        &self,
        module: &wgpu::ShaderModule,
        entry: &str,
    ) -> Result<wgpu::ComputePipeline> {
        let p = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        self.check()?;
        Ok(p)
    }

    /// A bind group for group 0 of `pipeline`, with each buffer at its binding number.
    pub(crate) fn bind(
        &self,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[(u32, &wgpu::Buffer)],
    ) -> wgpu::BindGroup {
        let layout = pipeline.get_bind_group_layout(0);
        let entries: Vec<wgpu::BindGroupEntry<'_>> = buffers
            .iter()
            .map(|&(i, b)| wgpu::BindGroupEntry { binding: i, resource: b.as_entire_binding() })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        })
    }

    /// Submits `encoder` after a copy of `bytes` bytes of `src` into `staging`, waits, and gives
    /// the bytes. Cost: one submission and one wait for the GPU.
    pub(crate) fn finish_and_read(
        &self,
        encoder: wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        staging: &wgpu::Buffer,
        bytes: u64,
        out: &mut Vec<u8>,
    ) -> Result<()> {
        let mut outs = [std::mem::take(out)];
        self.finish_and_read_all(encoder, &[(src, staging, bytes)], &mut outs)?;
        let [o] = outs;
        *out = o;
        Ok(())
    }

    /// As `finish_and_read`, for several (source, staging, bytes) copies in one submission.
    pub(crate) fn finish_and_read_all(
        &self,
        mut encoder: wgpu::CommandEncoder,
        copies: &[(&wgpu::Buffer, &wgpu::Buffer, u64)],
        outs: &mut [Vec<u8>],
    ) -> Result<()> {
        for &(src, staging, bytes) in copies {
            encoder.copy_buffer_to_buffer(src, 0, staging, 0, round4(bytes));
        }
        self.queue.submit([encoder.finish()]);
        self.check()?;
        let (tx, rx) = mpsc::channel();
        for &(_, staging, bytes) in copies {
            let tx = tx.clone();
            staging.slice(0..round4(bytes)).map_async(wgpu::MapMode::Read, move |r| {
                // NOTE: the receiver waits below; a failed send means that it stopped waiting.
                if tx.send(r).is_err() {}
            });
        }
        drop(tx);
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu { reason: e.to_string() })?;
        for _ in copies {
            match rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(Error::Gpu { reason: e.to_string() }),
                Err(e) => return Err(Error::Gpu { reason: e.to_string() }),
            }
        }
        for (&(_, staging, bytes), out) in copies.iter().zip(outs.iter_mut()) {
            {
                let slice = staging.slice(0..round4(bytes));
                let view =
                    slice.get_mapped_range().map_err(|e| Error::Gpu { reason: e.to_string() })?;
                out.clear();
                out.extend_from_slice(view.get(..usize_of(bytes)).unwrap_or(&[]));
            }
            staging.unmap();
        }
        self.check()
    }

    /// Submits `encoder` and waits until the GPU finishes it.
    pub(crate) fn finish_and_wait(&self, encoder: wgpu::CommandEncoder) -> Result<()> {
        self.queue.submit([encoder.finish()]);
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu { reason: e.to_string() })?;
        self.check()
    }

    /// Writes `words` to `buffer` at offset 0, in queue order.
    pub(crate) fn write_words(&self, buffer: &wgpu::Buffer, words: &[u32]) {
        self.queue.write_buffer(buffer, 0, &words_to_bytes(words));
    }
}

fn round4(x: u64) -> u64 {
    x.div_ceil(4) * 4
}

fn usize_of(x: u64) -> usize {
    usize::try_from(x).unwrap_or(usize::MAX)
}

/// Little-endian bytes of `words`. WGSL host-shareable memory is little-endian.
pub(crate) fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// The u32 words of little-endian bytes. A tail of fewer than 4 bytes is ignored.
pub(crate) fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect()
}
