//! The CUDA driver API, loaded at run time from `libcuda` (ADR 0008).
//!
//! This is the only module with `unsafe` code. It loads the driver library with `libloading`,
//! keeps the function pointers, and wraps each call that the crate needs in a safe method. No
//! CUDA toolkit is necessary: the driver comes with the GPU (on WSL2 it is
//! `/usr/lib/wsl/lib/libcuda.so.1`).
#![allow(unsafe_code)]

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::rc::Rc;

use crate::error::{Error, Result};

type CuResult = c_int;
type CuDevice = c_int;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuDevicePtr = u64;
type CuStream = *mut c_void;
type CuGraph = *mut c_void;
type CuGraphExec = *mut c_void;

/// `CU_STREAM_NON_BLOCKING`.
const STREAM_NON_BLOCKING: c_uint = 1;
/// `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`.
const CAPTURE_THREAD_LOCAL: c_int = 1;

/// The names of the driver library, in the order of the search.
const LIBRARY_NAMES: [&str; 3] = ["libcuda.so.1", "libcuda.so", "nvcuda.dll"];

const ATTR_CC_MAJOR: c_int = 75;
const ATTR_CC_MINOR: c_int = 76;

/// The function pointers of the driver API that this crate uses.
struct Api {
    init: unsafe extern "C" fn(c_uint) -> CuResult,
    device_get_count: unsafe extern "C" fn(*mut c_int) -> CuResult,
    device_get: unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult,
    device_get_name: unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult,
    device_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, CuDevice) -> CuResult,
    primary_ctx_retain: unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult,
    primary_ctx_release: unsafe extern "C" fn(CuDevice) -> CuResult,
    ctx_set_current: unsafe extern "C" fn(CuContext) -> CuResult,
    module_load_data: unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult,
    module_unload: unsafe extern "C" fn(CuModule) -> CuResult,
    module_get_function: unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult,
    mem_alloc: unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult,
    mem_free: unsafe extern "C" fn(CuDevicePtr) -> CuResult,
    memset_d8_async: unsafe extern "C" fn(CuDevicePtr, u8, usize, CuStream) -> CuResult,
    mem_get_info: unsafe extern "C" fn(*mut usize, *mut usize) -> CuResult,
    get_error_name: unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult,
    stream_create: unsafe extern "C" fn(*mut CuStream, c_uint) -> CuResult,
    stream_destroy: unsafe extern "C" fn(CuStream) -> CuResult,
    stream_synchronize: unsafe extern "C" fn(CuStream) -> CuResult,
    memcpy_htod_async:
        unsafe extern "C" fn(CuDevicePtr, *const c_void, usize, CuStream) -> CuResult,
    memcpy_dtoh_async: unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize, CuStream) -> CuResult,
    stream_begin_capture: unsafe extern "C" fn(CuStream, c_int) -> CuResult,
    stream_end_capture: unsafe extern "C" fn(CuStream, *mut CuGraph) -> CuResult,
    graph_instantiate: unsafe extern "C" fn(*mut CuGraphExec, CuGraph, u64) -> CuResult,
    graph_launch: unsafe extern "C" fn(CuGraphExec, CuStream) -> CuResult,
    graph_exec_destroy: unsafe extern "C" fn(CuGraphExec) -> CuResult,
    graph_destroy: unsafe extern "C" fn(CuGraph) -> CuResult,
    #[allow(clippy::type_complexity)]
    launch_kernel: unsafe extern "C" fn(
        CuFunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CuResult,
    /// Keeps the library loaded while the pointers live.
    _lib: libloading::Library,
}

/// Loads one symbol as a function pointer of type `T`.
///
/// # Safety
/// `T` must be the exact C signature of the symbol `name` in the CUDA driver API.
unsafe fn sym<T: Copy>(lib: &libloading::Library, name: &str) -> Result<T> {
    // SAFETY: the caller gives the type of the symbol; the pointer stays valid while `lib`
    // lives, and `Api` keeps `lib` next to the pointers.
    unsafe { lib.get::<T>(name.as_bytes()).map(|s| *s) }
        .map_err(|_| Error::NoDevice { reason: format!("the driver has no symbol {name}") })
}

impl Api {
    fn load() -> Result<Self> {
        let mut last = String::new();
        for name in LIBRARY_NAMES {
            // SAFETY: loading the NVIDIA driver library runs its initialisers, which is the
            // documented way to use the driver API. No other code of ours runs in them.
            match unsafe { libloading::Library::new(name) } {
                Ok(lib) => return Self::from_library(lib),
                Err(e) => last = e.to_string(),
            }
        }
        Err(Error::NoDevice { reason: format!("no CUDA driver library ({last})") })
    }

    fn from_library(lib: libloading::Library) -> Result<Self> {
        // SAFETY: each type below is the C signature of the named function in the CUDA driver
        // API (cuda.h, CUDA 12); the `_v2` names are the ones that cuda.h maps the plain names
        // to on 64-bit platforms.
        unsafe {
            Ok(Self {
                init: sym(&lib, "cuInit")?,
                device_get_count: sym(&lib, "cuDeviceGetCount")?,
                device_get: sym(&lib, "cuDeviceGet")?,
                device_get_name: sym(&lib, "cuDeviceGetName")?,
                device_get_attribute: sym(&lib, "cuDeviceGetAttribute")?,
                primary_ctx_retain: sym(&lib, "cuDevicePrimaryCtxRetain")?,
                primary_ctx_release: sym(&lib, "cuDevicePrimaryCtxRelease_v2")?,
                ctx_set_current: sym(&lib, "cuCtxSetCurrent")?,
                module_load_data: sym(&lib, "cuModuleLoadData")?,
                module_unload: sym(&lib, "cuModuleUnload")?,
                module_get_function: sym(&lib, "cuModuleGetFunction")?,
                mem_alloc: sym(&lib, "cuMemAlloc_v2")?,
                mem_free: sym(&lib, "cuMemFree_v2")?,
                memset_d8_async: sym(&lib, "cuMemsetD8Async")?,
                mem_get_info: sym(&lib, "cuMemGetInfo_v2")?,
                get_error_name: sym(&lib, "cuGetErrorName")?,
                stream_create: sym(&lib, "cuStreamCreate")?,
                stream_destroy: sym(&lib, "cuStreamDestroy_v2")?,
                stream_synchronize: sym(&lib, "cuStreamSynchronize")?,
                memcpy_htod_async: sym(&lib, "cuMemcpyHtoDAsync_v2")?,
                memcpy_dtoh_async: sym(&lib, "cuMemcpyDtoHAsync_v2")?,
                stream_begin_capture: sym(&lib, "cuStreamBeginCapture_v2")?,
                stream_end_capture: sym(&lib, "cuStreamEndCapture")?,
                graph_instantiate: sym(&lib, "cuGraphInstantiateWithFlags")?,
                graph_launch: sym(&lib, "cuGraphLaunch")?,
                graph_exec_destroy: sym(&lib, "cuGraphExecDestroy")?,
                graph_destroy: sym(&lib, "cuGraphDestroy")?,
                launch_kernel: sym(&lib, "cuLaunchKernel")?,
                _lib: lib,
            })
        }
    }

    fn check(&self, call: &'static str, r: CuResult) -> Result<()> {
        if r == 0 {
            return Ok(());
        }
        let mut p: *const c_char = std::ptr::null();
        // SAFETY: cuGetErrorName writes a pointer to a static string or leaves it null.
        let known = unsafe { (self.get_error_name)(r, &raw mut p) } == 0 && !p.is_null();
        let name = if known {
            // SAFETY: the driver gives a NUL-terminated static string.
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        } else {
            format!("error {r}")
        };
        Err(Error::Cuda { call, code: r, name })
    }
}

/// A kernel argument. The kernel signatures are in `kernels.rs`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Arg {
    /// A device pointer (8 bytes).
    Ptr(u64),
    /// A 32-bit value.
    U32(u32),
    /// A 32-bit signed value.
    I32(i32),
    /// A 64-bit value.
    U64(u64),
}

/// A CUDA context on one device, with the loaded module of the HardShell kernels.
pub(crate) struct Context {
    api: Rc<Api>,
    dev: CuDevice,
    ctx: CuContext,
    module: CuModule,
    /// The stream of all work of this context, in order.
    stream: CuStream,
    name: String,
    cc: (i32, i32),
}

/// An instantiated CUDA graph: a recorded sequence of kernel launches that one call replays.
pub(crate) struct Graph {
    api: Rc<Api>,
    graph: CuGraph,
    exec: CuGraphExec,
}

impl Drop for Graph {
    fn drop(&mut self) {
        // SAFETY: both handles were made by `Context::end_capture` and are destroyed once.
        unsafe {
            let _ = (self.api.graph_exec_destroy)(self.exec);
            let _ = (self.api.graph_destroy)(self.graph);
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: the module and the primary context were made by this value and are not used
        // after this point. Errors at shut-down cannot be acted on.
        unsafe {
            if (self.api.ctx_set_current)(self.ctx) == 0 {
                let _ = (self.api.stream_synchronize)(self.stream);
                let _ = (self.api.stream_destroy)(self.stream);
                let _ = (self.api.module_unload)(self.module);
            }
            let _ = (self.api.primary_ctx_release)(self.dev);
        }
    }
}

/// A device buffer. It frees its memory when it drops.
pub(crate) struct DevBuf {
    api: Rc<Api>,
    ptr: CuDevicePtr,
    bytes: usize,
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from cuMemAlloc and is freed once, here.
        unsafe {
            let _ = (self.api.mem_free)(self.ptr);
        }
    }
}

impl DevBuf {
    pub(crate) fn ptr(&self) -> u64 {
        self.ptr
    }
}

impl Context {
    /// Opens device `ordinal` with its primary context and loads the PTX module `ptx` (the
    /// driver compiles it for the GPU).
    pub(crate) fn open(ordinal: usize, ptx: &str) -> Result<Self> {
        let api = Rc::new(Api::load()?);
        // SAFETY: plain driver calls with valid out-pointers to locals.
        unsafe {
            api.check("cuInit", (api.init)(0))?;
            let mut count = 0;
            api.check("cuDeviceGetCount", (api.device_get_count)(&raw mut count))?;
            let ord = c_int::try_from(ordinal).unwrap_or(c_int::MAX);
            if ord >= count {
                return Err(Error::NoDevice { reason: format!("{count} CUDA devices") });
            }
            let mut dev = 0;
            api.check("cuDeviceGet", (api.device_get)(&raw mut dev, ord))?;
            let mut buf = [0 as c_char; 256];
            api.check("cuDeviceGetName", (api.device_get_name)(buf.as_mut_ptr(), 255, dev))?;
            let name = CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned();
            let (mut major, mut minor) = (0, 0);
            api.check(
                "cuDeviceGetAttribute",
                (api.device_get_attribute)(&raw mut major, ATTR_CC_MAJOR, dev),
            )?;
            api.check(
                "cuDeviceGetAttribute",
                (api.device_get_attribute)(&raw mut minor, ATTR_CC_MINOR, dev),
            )?;
            let mut ctx = std::ptr::null_mut();
            api.check("cuDevicePrimaryCtxRetain", (api.primary_ctx_retain)(&raw mut ctx, dev))?;
            api.check("cuCtxSetCurrent", (api.ctx_set_current)(ctx))?;
            let image = CString::new(ptx)
                .map_err(|_| Error::Unsupported { reason: "the PTX has a NUL byte" })?;
            let mut module = std::ptr::null_mut();
            let r = (api.module_load_data)(&raw mut module, image.as_ptr().cast());
            if let Err(e) = api.check("cuModuleLoadData", r) {
                let _ = (api.primary_ctx_release)(dev);
                return Err(e);
            }
            let mut stream = std::ptr::null_mut();
            let r = (api.stream_create)(&raw mut stream, STREAM_NON_BLOCKING);
            if let Err(e) = api.check("cuStreamCreate", r) {
                let _ = (api.module_unload)(module);
                let _ = (api.primary_ctx_release)(dev);
                return Err(e);
            }
            Ok(Self { api, dev, ctx, module, stream, name, cc: (major, minor) })
        }
    }

    /// The device name.
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    /// The compute capability (major, minor).
    pub(crate) fn compute_capability(&self) -> (i32, i32) {
        self.cc
    }

    fn current(&self) -> Result<()> {
        // SAFETY: `ctx` is the retained primary context of this value.
        self.api.check("cuCtxSetCurrent", unsafe { (self.api.ctx_set_current)(self.ctx) })
    }

    /// Free and total device memory, in bytes.
    pub(crate) fn mem_info(&self) -> Result<(usize, usize)> {
        self.current()?;
        let (mut free, mut total) = (0, 0);
        // SAFETY: valid out-pointers.
        self.api.check("cuMemGetInfo", unsafe {
            (self.api.mem_get_info)(&raw mut free, &raw mut total)
        })?;
        Ok((free, total))
    }

    /// A kernel of the module, by name.
    pub(crate) fn function(&self, name: &'static str) -> Result<Kernel> {
        self.current()?;
        let c = CString::new(name)
            .map_err(|_| Error::Unsupported { reason: "a kernel name has a NUL byte" })?;
        let mut f = std::ptr::null_mut();
        // SAFETY: `module` is loaded; `c` is NUL-terminated.
        let r = unsafe { (self.api.module_get_function)(&raw mut f, self.module, c.as_ptr()) };
        self.api.check("cuModuleGetFunction", r)?;
        Ok(Kernel { f, name })
    }

    /// A device buffer of `bytes` zero bytes (at least 256).
    pub(crate) fn zeroed(&self, bytes: usize) -> Result<DevBuf> {
        self.current()?;
        let n = bytes.max(256);
        let mut ptr = 0;
        // SAFETY: valid out-pointer; the size is not zero.
        self.api.check("cuMemAlloc", unsafe { (self.api.mem_alloc)(&raw mut ptr, n) })?;
        let buf = DevBuf { api: Rc::clone(&self.api), ptr, bytes: n };
        // NOTE: the memset runs on the stream of the context, so every later use of the buffer
        // sees zero bytes. A context-wide wait here would break a graph capture of another
        // thread that shares the primary context.
        // SAFETY: `ptr` holds `n` bytes; `stream` belongs to this context.
        self.api.check("cuMemsetD8Async", unsafe {
            (self.api.memset_d8_async)(ptr, 0, n, self.stream)
        })?;
        Ok(buf)
    }

    /// A device buffer with these bytes, then zero bytes up to `capacity`.
    pub(crate) fn upload(&self, bytes: &[u8], capacity: usize) -> Result<DevBuf> {
        let buf = self.zeroed(capacity.max(bytes.len()))?;
        self.write(&buf, 0, bytes)?;
        Ok(buf)
    }

    /// Copies `bytes` to `buf` at `offset`, in stream order.
    pub(crate) fn write(&self, buf: &DevBuf, offset: usize, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        if offset.checked_add(bytes.len()).is_none_or(|e| e > buf.bytes) {
            return Err(Error::LengthMismatch { expected: buf.bytes, found: offset + bytes.len() });
        }
        self.current()?;
        // SAFETY: the range is inside the buffer (checked above); the source is a valid slice.
        // The source is pageable memory: the driver copies it to its staging memory before the
        // call returns, so the slice can drop after the call.
        let r = unsafe {
            (self.api.memcpy_htod_async)(
                buf.ptr + offset as u64,
                bytes.as_ptr().cast(),
                bytes.len(),
                self.stream,
            )
        };
        self.api.check("cuMemcpyHtoDAsync", r)
    }

    /// Copies `out.len()` bytes from `buf` at offset 0, in stream order, and waits.
    pub(crate) fn read(&self, buf: &DevBuf, out: &mut [u8]) -> Result<()> {
        if out.is_empty() {
            return Ok(());
        }
        if out.len() > buf.bytes {
            return Err(Error::LengthMismatch { expected: buf.bytes, found: out.len() });
        }
        self.current()?;
        // SAFETY: the range is inside the buffer; `out` is a valid, writable slice, and the
        // stream synchronisation below ends the copy before `out` can be used or dropped.
        let r = unsafe {
            (self.api.memcpy_dtoh_async)(out.as_mut_ptr().cast(), buf.ptr, out.len(), self.stream)
        };
        self.api.check("cuMemcpyDtoHAsync", r)?;
        self.synchronize()
    }

    /// Waits for all work of the stream.
    pub(crate) fn synchronize(&self) -> Result<()> {
        self.current()?;
        // SAFETY: `stream` belongs to this context.
        self.api.check("cuStreamSynchronize", unsafe { (self.api.stream_synchronize)(self.stream) })
    }

    /// Starts to record the launches of this thread on the stream into a graph.
    pub(crate) fn begin_capture(&self) -> Result<()> {
        self.current()?;
        // SAFETY: `stream` belongs to this context and is not capturing (the caller pairs each
        // begin with one end).
        let r = unsafe { (self.api.stream_begin_capture)(self.stream, CAPTURE_THREAD_LOCAL) };
        self.api.check("cuStreamBeginCapture", r)
    }

    /// Ends the recording and makes an executable graph.
    pub(crate) fn end_capture(&self) -> Result<Graph> {
        let mut graph = std::ptr::null_mut();
        // SAFETY: the stream is capturing (begin_capture); valid out-pointer.
        let r = unsafe { (self.api.stream_end_capture)(self.stream, &raw mut graph) };
        self.api.check("cuStreamEndCapture", r)?;
        let mut exec = std::ptr::null_mut();
        // SAFETY: `graph` is a valid graph from the capture.
        let r = unsafe { (self.api.graph_instantiate)(&raw mut exec, graph, 0) };
        if let Err(e) = self.api.check("cuGraphInstantiate", r) {
            // SAFETY: `graph` is not used again.
            unsafe {
                let _ = (self.api.graph_destroy)(graph);
            }
            return Err(e);
        }
        Ok(Graph { api: Rc::clone(&self.api), graph, exec })
    }

    /// Replays `g` on the stream.
    pub(crate) fn launch_graph(&self, g: &Graph) -> Result<()> {
        self.current()?;
        // SAFETY: `g.exec` is a live executable graph of this context.
        self.api.check("cuGraphLaunch", unsafe { (self.api.graph_launch)(g.exec, self.stream) })
    }

    /// Launches `k` on the stream of the context.
    ///
    /// The caller gives the arguments in the order and with the sizes of the kernel signature
    /// in `kernels/hardshell.cu`; `kernels.rs` keeps one launch function for each kernel, and
    /// the tests run every one of them.
    pub(crate) fn launch(
        &self,
        k: &Kernel,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[Arg],
    ) -> Result<()> {
        let mut values: Vec<u64> = args
            .iter()
            .map(|a| match *a {
                Arg::Ptr(p) | Arg::U64(p) => p,
                Arg::U32(v) => u64::from(v),
                Arg::I32(v) => u64::from(v.cast_unsigned()),
            })
            .collect();
        // NOTE: the driver reads the size of each parameter from the kernel; a 4-byte value
        // is the low half of its little-endian u64 slot.
        let mut ptrs: Vec<*mut c_void> =
            values.iter_mut().map(|v| std::ptr::from_mut(v).cast()).collect();
        // SAFETY: `k` is a function of the loaded module; `ptrs` point to live values whose
        // sizes and order match the kernel signature (see the doc comment); the grid and block
        // sizes are checked by the driver, which reports an error and does not launch.
        let r = unsafe {
            (self.api.launch_kernel)(
                k.f,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                0,
                self.stream,
                ptrs.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        self.api.check(k.name, r)
    }
}

/// A kernel function of the module.
pub(crate) struct Kernel {
    f: CuFunction,
    name: &'static str,
}
