//! Streaming fold and unfold: inputs of any size with bounded memory, from and to files and
//! pipes (ADR 0018).
//!
//! The container format does not change. A version 2 container cannot be written in one pass
//! (ADR 0018), so [`StreamFold`] has two output methods:
//!
//! - [`StreamFold::to_seekable`]: the input length is known and the output can seek. The
//!   writer reserves the header and the table, writes the stored blocks, and writes the table
//!   (in groups of 4,096 entries) and the header with a seek.
//! - [`StreamFold::to_writer`]: any output, any input. The stored blocks and the table go to
//!   two temporary files in a directory of the caller. At the end the header, the table and
//!   the blocks go to the output.
//!
//! [`StreamUnfold`] reads in one pass. With [`StreamUnfold::from_seekable`] it first checks the
//! whole table and the input length, so a damaged file is refused before any output. With
//! [`StreamUnfold::from_reader`] (a pipe) it keeps the table in memory (5 bytes for each block)
//! and finds a missing or an extra byte where it occurs.
//!
//! The bytes are the bytes of [`crate::fold`] and [`crate::unfold`]: each block is coded and
//! decoded alone, and the work moves in segments of blocks. A test compares them.
//!
//! # Memory
//!
//! A segment has at most `clamp(64 MiB / block size, 1, 256)` blocks ([`segment_blocks`]).
//! fold holds the input of one segment, its tokens (4 bytes for each token), its zstd frames
//! and its payloads. unfold holds the stored bytes and the restored bytes of one segment. Both
//! hold the model, one session or batch for each thread, and a table buffer of 4,096 entries.
//! Only [`StreamUnfold::from_reader`] holds memory that grows with the file: 5 bytes for each
//! block. The block size is at most [`MAX_BLOCK_SIZE`] (a limit of the stream, not of the
//! format).

use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use hardshell::gpt2::Model;
use taco_format::{
    BlockSize, ChecksumBuilder, ContainerParams, FormatError, HEADER_LEN, Header, StreamWriter,
    TABLE_ENTRY_LEN, TableEntry, TableReader,
};

use crate::backend::Engine;
use crate::container_hash;
use crate::error::TacoError;
use crate::fold::{FoldOptions, FoldPlan, FoldStats, code_segment};
use crate::tokenizer::Tokenizer;
use crate::unfold::{UnfoldOptions, checked_model, checksum_error, decode_blocks};

/// The largest block size: 64 MiB, the maximum of the format (`taco_format::MAX_BLOCK_SIZE`,
/// ADR 0011 addendum §4). It is the default limit of a stream. A caller can give a smaller
/// limit ([`StreamOptions::with_max_block_size`]); a larger one has no effect, because the
/// format refuses a larger block size.
pub const MAX_BLOCK_SIZE: BlockSize = taco_format::MAX_BLOCK_SIZE;

/// The most input bytes of one segment (the work of one call to the coder or the decoder).
const SEGMENT_BYTES: u64 = 64 * 1024 * 1024;

/// The most blocks of one segment.
// UNVERIFIED: 256 blocks fill the 224 slots of the CUDA unfold and give each of 16 CPU threads
// 16 blocks. The speed against `taco::fold` on one input is in ADR 0018, measurements.
const SEGMENT_BLOCKS: u64 = 256;

/// The number of table entries that the writer and the reader move with one seek.
const TABLE_GROUP: usize = 4096;

/// The size of the buffers of the reader and of the temporary files: 1 MiB.
const IO_BUFFER: usize = 1024 * 1024;

/// The number of blocks of one segment for `block_size`: `clamp(64 MiB / block size, 1, 256)`.
/// The output does not depend on it.
#[must_use]
pub fn segment_blocks(block_size: BlockSize) -> usize {
    let blocks = (SEGMENT_BYTES / u64::from(block_size.get())).clamp(1, SEGMENT_BLOCKS);
    usize::try_from(blocks).unwrap_or(1)
}

/// A fault of a stream fold or a stream unfold.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum StreamError {
    /// A fault of the library: the model, the coder or the container.
    #[error(transparent)]
    Taco(#[from] TacoError),
    /// A read or a write failed.
    #[error("cannot {what}: {source}")]
    Io {
        /// What the stream did.
        what: &'static str,
        /// The error of the system.
        #[source]
        source: io::Error,
    },
    /// The block size is more than the limit of the stream ([`MAX_BLOCK_SIZE`]).
    #[error("the block size {block_size} is more than the limit {limit} of a stream")]
    BlockSizeLimit {
        /// The block size of the options or of the header.
        block_size: u32,
        /// The limit.
        limit: u32,
    },
}

impl From<FormatError> for StreamError {
    fn from(e: FormatError) -> Self {
        Self::Taco(TacoError::Format(e))
    }
}

/// The result type of this module.
pub type Result<T> = core::result::Result<T, StreamError>;

/// A function for an [`StreamError::Io`] error of `what`.
fn io_err(what: &'static str) -> impl Fn(io::Error) -> StreamError {
    move |source| StreamError::Io { what, source }
}

/// `n` as a `u64`. INVARIANT: `usize` has at most 64 bits on every supported target.
fn len64(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

/// The limits of a stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct StreamOptions {
    /// The largest block size. The default is [`MAX_BLOCK_SIZE`].
    pub max_block_size: BlockSize,
}

impl StreamOptions {
    /// The default limits.
    #[must_use]
    pub const fn new() -> Self {
        Self { max_block_size: MAX_BLOCK_SIZE }
    }

    /// The same limits with another largest block size.
    #[must_use]
    pub const fn with_max_block_size(mut self, max: BlockSize) -> Self {
        self.max_block_size = max;
        self
    }

    fn check(&self, block_size: BlockSize) -> Result<()> {
        if block_size > self.max_block_size {
            return Err(StreamError::BlockSizeLimit {
                block_size: block_size.get(),
                limit: self.max_block_size.get(),
            });
        }
        Ok(())
    }
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// How far a stream is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Progress {
    /// fold: the input bytes that are coded. unfold: the container bytes that are read.
    pub input_bytes: u64,
    /// fold: the stored bytes of the blocks that are written. unfold: the restored bytes.
    pub output_bytes: u64,
    /// The length of the input, if it is known.
    pub input_len: Option<u64>,
}

/// A function that receives the progress. fold calls it after each block, from the thread
/// that coded the block. unfold calls it after each segment.
pub type ProgressFn<'p> = &'p (dyn Fn(Progress) + Sync);

/// Numbers about one stream unfold.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct UnfoldStats {
    /// The length of the restored data, in bytes.
    pub original_len: u64,
    /// The length of the container, in bytes.
    pub container_len: u64,
    /// The number of blocks.
    pub blocks: u64,
    /// The format version of the container.
    pub version: u16,
}

/// Reads until `buf` is full or the input ends. Gives the number of bytes read.
fn read_full<R: Read>(input: &mut R, buf: &mut [u8]) -> io::Result<usize> {
    let mut done = 0;
    while let Some(rest) = buf.get_mut(done..).filter(|r| !r.is_empty()) {
        match input.read(rest) {
            Ok(0) => break,
            Ok(n) => done += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(done)
}

/// Reads at most `limit` bytes into `buf` (cleared first). The buffer grows only with bytes
/// that arrive. Gives the number of bytes read.
fn read_up_to<R: Read>(input: &mut R, limit: u64, buf: &mut Vec<u8>) -> io::Result<u64> {
    buf.clear();
    let n = input.take(limit).read_to_end(buf)?;
    Ok(len64(n))
}

// ---------------------------------------------------------------------------------------
// fold

/// Where the parts of the container go.
trait Layout {
    /// The stored bytes of the next block.
    fn block(&mut self, stored: &[u8]) -> Result<()>;
    /// The table entry of the next block.
    fn entry(&mut self, entry: &TableEntry) -> Result<()>;
    /// The header. The container is then complete.
    fn finish(&mut self, header: &[u8]) -> Result<()>;
}

/// The method "seek back": header and table reserved, then written with a seek.
struct SeekLayout<'w, W: Write + Seek> {
    out: &'w mut W,
    base: u64,
    table_pos: u64,
    end: u64,
    pending: Vec<u8>,
}

impl<'w, W: Write + Seek> SeekLayout<'w, W> {
    /// Writes `data_offset` zero bytes at the position of `out`.
    fn new(out: &'w mut W, data_offset: u64) -> Result<Self> {
        let base = out.stream_position().map_err(io_err("seek the output"))?;
        let zeros = [0u8; 64 * 1024];
        let mut left = data_offset;
        while left > 0 {
            let n = left.min(len64(zeros.len()));
            let part = zeros.get(..usize::try_from(n).unwrap_or(0)).unwrap_or(&[]);
            out.write_all(part).map_err(io_err("write the output"))?;
            left -= n;
        }
        Ok(Self {
            out,
            base,
            table_pos: base + len64(HEADER_LEN),
            end: base + data_offset,
            pending: Vec::with_capacity(TABLE_GROUP * TABLE_ENTRY_LEN),
        })
    }

    fn flush_table(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let err = io_err("write the block table");
        self.out.seek(SeekFrom::Start(self.table_pos)).map_err(&err)?;
        self.out.write_all(&self.pending).map_err(&err)?;
        self.table_pos += len64(self.pending.len());
        self.pending.clear();
        self.out.seek(SeekFrom::Start(self.end)).map_err(&err)?;
        Ok(())
    }
}

impl<W: Write + Seek> Layout for SeekLayout<'_, W> {
    fn block(&mut self, stored: &[u8]) -> Result<()> {
        self.out.write_all(stored).map_err(io_err("write the output"))?;
        self.end += len64(stored.len());
        Ok(())
    }

    fn entry(&mut self, entry: &TableEntry) -> Result<()> {
        self.pending.extend_from_slice(&entry.to_bytes());
        if self.pending.len() >= TABLE_GROUP * TABLE_ENTRY_LEN {
            self.flush_table()?;
        }
        Ok(())
    }

    fn finish(&mut self, header: &[u8]) -> Result<()> {
        self.flush_table()?;
        let err = io_err("write the header");
        self.out.seek(SeekFrom::Start(self.base)).map_err(&err)?;
        self.out.write_all(header).map_err(&err)?;
        self.out.seek(SeekFrom::Start(self.end)).map_err(&err)?;
        self.out.flush().map_err(io_err("write the output"))
    }
}

/// A temporary file. The file is deleted when the value is dropped, also after an error.
struct TempFile {
    path: PathBuf,
    file: BufWriter<File>,
}

impl TempFile {
    /// A new file in `dir` with a name that no other file has.
    fn create(dir: &Path, tag: &str) -> Result<Self> {
        let mut last = None;
        for n in 0..1000u32 {
            let path = dir.join(format!(".taco-spool-{}-{tag}-{n}", std::process::id()));
            match OpenOptions::new().read(true).write(true).create_new(true).open(&path) {
                Ok(f) => return Ok(Self { path, file: BufWriter::with_capacity(IO_BUFFER, f) }),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => last = Some(e),
                Err(e) => return Err(StreamError::Io { what: "make a temporary file", source: e }),
            }
        }
        Err(StreamError::Io {
            what: "make a temporary file",
            source: last.unwrap_or_else(|| io::Error::from(io::ErrorKind::AlreadyExists)),
        })
    }

    /// Copies all bytes of the file to `out`.
    fn copy_to<W: Write + ?Sized>(&mut self, out: &mut W) -> Result<()> {
        let err = io_err("read a temporary file");
        self.file.flush().map_err(&err)?;
        let file = self.file.get_mut();
        file.seek(SeekFrom::Start(0)).map_err(&err)?;
        let mut reader = BufReader::with_capacity(IO_BUFFER, file);
        io::copy(&mut reader, out).map_err(io_err("write the output"))?;
        Ok(())
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        // NOTE: a temporary file that cannot be deleted is not a fault of the fold. The name
        // starts with ".taco-spool-", so a person can find it.
        if std::fs::remove_file(&self.path).is_err() {}
    }
}

/// The method "spool": the stored blocks and the table go to temporary files.
struct SpoolLayout<'w, W: Write + ?Sized> {
    out: &'w mut W,
    blocks: TempFile,
    table: TempFile,
}

impl<W: Write + ?Sized> Layout for SpoolLayout<'_, W> {
    fn block(&mut self, stored: &[u8]) -> Result<()> {
        self.blocks.file.write_all(stored).map_err(io_err("write a temporary file"))
    }

    fn entry(&mut self, entry: &TableEntry) -> Result<()> {
        self.table.file.write_all(&entry.to_bytes()).map_err(io_err("write a temporary file"))
    }

    fn finish(&mut self, header: &[u8]) -> Result<()> {
        self.out.write_all(header).map_err(io_err("write the output"))?;
        self.table.copy_to(self.out)?;
        self.blocks.copy_to(self.out)?;
        self.out.flush().map_err(io_err("write the output"))
    }
}

/// A fold that streams: the same container as [`crate::fold_with`], with bounded memory.
pub struct StreamFold<'a, 'e> {
    engine: &'a Engine<'e>,
    model: &'a Model,
    tokenizer: &'a Tokenizer,
    options: FoldOptions,
    limits: StreamOptions,
    progress: Option<ProgressFn<'a>>,
}

impl<'a, 'e> StreamFold<'a, 'e> {
    /// A stream fold with `options` on the backend of `engine`, with the default limits and no
    /// progress function.
    #[must_use]
    pub fn new(
        engine: &'a Engine<'e>,
        model: &'a Model,
        tokenizer: &'a Tokenizer,
        options: &FoldOptions,
    ) -> Self {
        Self {
            engine,
            model,
            tokenizer,
            options: *options,
            limits: StreamOptions::new(),
            progress: None,
        }
    }

    /// The same fold with other limits.
    #[must_use]
    pub const fn with_limits(mut self, limits: StreamOptions) -> Self {
        self.limits = limits;
        self
    }

    /// The same fold with a progress function.
    #[must_use]
    pub fn with_progress(mut self, progress: ProgressFn<'a>) -> Self {
        self.progress = Some(progress);
        self
    }

    /// Folds `input` of exactly `input_len` bytes into `out`, from the position of `out`, with
    /// the method "seek back" (ADR 0018 §1.1). The output needs no temporary file.
    ///
    /// # Errors
    ///
    /// - The errors of [`crate::fold_with`].
    /// - [`StreamError::BlockSizeLimit`] if the block size is more than the limit.
    /// - [`FormatError::InputLengthMismatch`] (in [`StreamError::Taco`]) if the input does not
    ///   have `input_len` bytes, and [`FormatError::TooManyBlocks`] if it needs more than
    ///   `u32::MAX` blocks.
    /// - [`StreamError::Io`] if a read, a write or a seek fails.
    pub fn to_seekable<R: Read, W: Write + Seek>(
        &self,
        input: R,
        input_len: u64,
        out: &mut W,
    ) -> Result<FoldStats> {
        let (plan, writer) = self.start(Some(input_len))?;
        let mut layout = SeekLayout::new(out, writer.data_offset(input_len)?)?;
        self.run(input, Some(input_len), &plan, writer, &mut layout)
    }

    /// Folds `input` into `out` with the method "spool" (ADR 0018 §1.2). `input_len` is the
    /// input length if it is known, else `None`. The temporary files go into `spool_dir` and
    /// are deleted at the end. They hold at most `5 × block count + input length` bytes. The
    /// output starts after the input ends.
    ///
    /// # Errors
    ///
    /// The errors of [`StreamFold::to_seekable`].
    pub fn to_writer<R: Read, W: Write + ?Sized>(
        &self,
        input: R,
        input_len: Option<u64>,
        out: &mut W,
        spool_dir: &Path,
    ) -> Result<FoldStats> {
        let (plan, writer) = self.start(input_len)?;
        let mut layout = SpoolLayout {
            out,
            blocks: TempFile::create(spool_dir, "blocks")?,
            table: TempFile::create(spool_dir, "table")?,
        };
        self.run(input, input_len, &plan, writer, &mut layout)
    }

    fn start(&self, input_len: Option<u64>) -> Result<(FoldPlan, StreamWriter)> {
        let plan = FoldPlan::new(self.model, self.tokenizer, &self.options)?;
        self.limits.check(plan.block_size)?;
        let (profile, coding) = plan.block_parts();
        let params = ContainerParams::new(container_hash(self.model), plan.block_size)
            .with_profile(profile)
            .with_coding(coding);
        Ok((plan, StreamWriter::new(params, input_len)?))
    }

    fn run<R: Read, L: Layout>(
        &self,
        mut input: R,
        input_len: Option<u64>,
        plan: &FoldPlan,
        mut writer: StreamWriter,
        layout: &mut L,
    ) -> Result<FoldStats> {
        let segment = len64(segment_blocks(plan.block_size)) * u64::from(plan.block_size.get());
        let done_in = AtomicU64::new(0);
        let mut done_out = 0u64;
        let mut stats = FoldStats::default();
        let mut buf = Vec::new();
        loop {
            let n = read_up_to(&mut input, segment, &mut buf).map_err(io_err("read the input"))?;
            if n == 0 {
                break;
            }
            let chunks: Vec<&[u8]> = plan.block_size.split(&buf).collect();
            let out_now = done_out;
            let progress = self.progress;
            let on_block = |len: usize| {
                let d = done_in.fetch_add(len64(len), Ordering::Relaxed) + len64(len);
                if let Some(p) = progress {
                    p(Progress { input_bytes: d, output_bytes: out_now, input_len });
                }
            };
            let first = u64::from(writer.blocks());
            let coded = code_segment(
                self.engine,
                self.model,
                self.tokenizer,
                plan,
                (first, &chunks),
                &on_block,
            )?;
            for (chunk, (block, text_stored, tokens)) in chunks.iter().zip(coded) {
                let (entry, stored) = writer.push(chunk, &block)?;
                layout.block(stored)?;
                layout.entry(&entry)?;
                done_out += len64(stored.len());
                stats.count_block(&block, text_stored, tokens)?;
            }
            if n < segment {
                break;
            }
        }
        let original_len = writer.original_len();
        let (profile, coding) = plan.container_parts(stats.mixed_blocks);
        let (_, header, container_len) = writer.finish(profile, coding)?;
        layout.finish(&header)?;
        stats.finish(plan, self.engine, profile, original_len, container_len)?;
        if let Some(p) = self.progress {
            p(Progress { input_bytes: original_len, output_bytes: container_len, input_len });
        }
        Ok(stats)
    }
}

// ---------------------------------------------------------------------------------------
// unfold

/// Where the table entries and the stored bytes come from.
trait Source {
    /// Puts the next entries of the table (at most `max`) into `out`.
    fn next_entries(
        &mut self,
        table: &mut TableReader,
        max: usize,
        out: &mut Vec<TableEntry>,
    ) -> Result<()>;
    /// Reads the `len` stored bytes of the next block into `buf`.
    fn read_stored(&mut self, entry: &TableEntry, buf: &mut Vec<u8>) -> Result<()>;
    /// Checks that the input ends after the last block.
    fn finish(&mut self) -> Result<()>;
    /// The container bytes read so far.
    fn bytes_read(&self) -> u64;
}

/// Reads `count` table entries from `bytes` with `table` into `out`.
fn entries_of(bytes: &[u8], table: &mut TableReader, out: &mut Vec<TableEntry>) -> Result<()> {
    let (chunks, _) = bytes.as_chunks::<TABLE_ENTRY_LEN>();
    for e in chunks {
        out.push(table.next_entry(e)?);
    }
    Ok(())
}

/// A pipe: the table in memory, the blocks in order.
struct PipeSource<R: Read> {
    input: R,
    table: Vec<u8>,
    next: usize,
    read: u64,
    /// The length of a valid container with this table.
    expected: u64,
}

impl<R: Read> PipeSource<R> {
    fn new(mut input: R, header: &Header, header_len: u64) -> Result<Self> {
        let mut check = TableReader::new(header)?;
        let data_offset = check.data_offset()?;
        let table_len = data_offset - header_len;
        let mut table = Vec::new();
        // NOTE: the table grows only with bytes that arrive, so a false block count costs no
        // memory before the bytes exist (ADR 0018 §3.3).
        let got =
            read_up_to(&mut input, table_len, &mut table).map_err(io_err("read the input"))?;
        if got < table_len {
            return Err(FormatError::Truncated {
                part: "block table",
                needed: data_offset,
                found: header_len + got,
            }
            .into());
        }
        // Check all entries before the first block, as `Container::parse` does.
        let (chunks, _) = table.as_chunks::<TABLE_ENTRY_LEN>();
        for e in chunks {
            check.next_entry(e)?;
        }
        Ok(Self { input, table, next: 0, read: data_offset, expected: check.container_len() })
    }
}

impl<R: Read> Source for PipeSource<R> {
    fn next_entries(
        &mut self,
        table: &mut TableReader,
        max: usize,
        out: &mut Vec<TableEntry>,
    ) -> Result<()> {
        let end = self.next.saturating_add(max.saturating_mul(TABLE_ENTRY_LEN));
        let bytes = self.table.get(self.next..end.min(self.table.len())).unwrap_or(&[]);
        self.next += bytes.len();
        entries_of(bytes, table, out)
    }

    fn read_stored(&mut self, entry: &TableEntry, buf: &mut Vec<u8>) -> Result<()> {
        let want = u64::from(entry.stored_len());
        let got = read_up_to(&mut self.input, want, buf).map_err(io_err("read the input"))?;
        self.read += got;
        if got < want {
            return Err(FormatError::WrongTotalLength {
                expected: self.expected,
                found: self.read,
            }
            .into());
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        let mut one = [0u8; 1];
        let n = read_full(&mut self.input, &mut one).map_err(io_err("read the input"))?;
        if n > 0 {
            return Err(FormatError::WrongTotalLength {
                expected: self.expected,
                found: self.read + len64(n),
            }
            .into());
        }
        Ok(())
    }

    fn bytes_read(&self) -> u64 {
        self.read
    }
}

/// A file: the table in groups with a seek, the blocks in order.
struct SeekSource<R: Read + Seek> {
    input: BufReader<R>,
    base: u64,
    table_pos: u64,
    block_pos: u64,
    at_blocks: bool,
    group: Vec<u8>,
}

impl<R: Read + Seek> SeekSource<R> {
    /// Checks the whole table and the input length before the first block (ADR 0018 §3.2).
    /// `base` is the position of the container in the input, `input_len` the length after it.
    fn new(input: R, header: &Header, base: u64, input_len: u64) -> Result<Self> {
        let header_len = len64(header.byte_len());
        let mut check = TableReader::new(header)?;
        let data_offset = check.data_offset()?;
        if data_offset > input_len {
            return Err(FormatError::Truncated {
                part: "block table",
                needed: data_offset,
                found: input_len,
            }
            .into());
        }
        let mut s = Self {
            input: BufReader::with_capacity(IO_BUFFER, input),
            base,
            table_pos: header_len,
            block_pos: data_offset,
            at_blocks: false,
            group: Vec::new(),
        };
        while check.remaining() > 0 {
            s.read_group(check.remaining())?;
            let (chunks, _) = s.group.as_chunks::<TABLE_ENTRY_LEN>();
            for e in chunks {
                check.next_entry(e)?;
            }
        }
        if check.container_len() != input_len {
            return Err(FormatError::WrongTotalLength {
                expected: check.container_len(),
                found: input_len,
            }
            .into());
        }
        s.table_pos = header_len;
        Ok(s)
    }

    /// Reads at most `max` (and at most 4,096) entries at the table position into the group.
    fn read_group(&mut self, max: u32) -> Result<()> {
        let n = usize::try_from(max).unwrap_or(usize::MAX).min(TABLE_GROUP);
        let len = n * TABLE_ENTRY_LEN;
        let err = io_err("read the input");
        self.input.seek(SeekFrom::Start(self.base + self.table_pos)).map_err(&err)?;
        self.at_blocks = false;
        self.group.resize(len, 0);
        let got = read_full(&mut self.input, &mut self.group).map_err(&err)?;
        if got < len {
            return Err(StreamError::Io {
                what: "read the input",
                source: io::Error::from(io::ErrorKind::UnexpectedEof),
            });
        }
        self.table_pos += len64(len);
        Ok(())
    }
}

impl<R: Read + Seek> Source for SeekSource<R> {
    fn next_entries(
        &mut self,
        table: &mut TableReader,
        max: usize,
        out: &mut Vec<TableEntry>,
    ) -> Result<()> {
        let mut left = max.min(usize::try_from(table.remaining()).unwrap_or(usize::MAX));
        while left > 0 {
            self.read_group(u32::try_from(left).unwrap_or(u32::MAX))?;
            let group = core::mem::take(&mut self.group);
            entries_of(&group, table, out)?;
            left -= group.len() / TABLE_ENTRY_LEN;
            self.group = group;
        }
        Ok(())
    }

    fn read_stored(&mut self, entry: &TableEntry, buf: &mut Vec<u8>) -> Result<()> {
        let err = io_err("read the input");
        if !self.at_blocks {
            self.input.seek(SeekFrom::Start(self.base + self.block_pos)).map_err(&err)?;
            self.at_blocks = true;
        }
        let want = u64::from(entry.stored_len());
        let got = read_up_to(&mut self.input, want, buf).map_err(&err)?;
        if got < want {
            // NOTE: the length was checked before the start, so the file changed.
            return Err(StreamError::Io {
                what: "read the input",
                source: io::Error::from(io::ErrorKind::UnexpectedEof),
            });
        }
        self.block_pos += want;
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        Ok(())
    }

    fn bytes_read(&self) -> u64 {
        self.block_pos
    }
}

/// An unfold that streams: the same output as [`crate::unfold_with`], with bounded memory.
pub struct StreamUnfold<'a, 'e> {
    engine: &'a Engine<'e>,
    model: Option<&'a Model>,
    options: UnfoldOptions,
    limits: StreamOptions,
    progress: Option<ProgressFn<'a>>,
}

impl<'a, 'e> StreamUnfold<'a, 'e> {
    /// A stream unfold with `options` on the backend of `engine`. `model` is `None` for a
    /// container in the raw-only mode.
    #[must_use]
    pub fn new(engine: &'a Engine<'e>, model: Option<&'a Model>, options: &UnfoldOptions) -> Self {
        Self { engine, model, options: *options, limits: StreamOptions::new(), progress: None }
    }

    /// The same unfold with other limits.
    #[must_use]
    pub const fn with_limits(mut self, limits: StreamOptions) -> Self {
        self.limits = limits;
        self
    }

    /// The same unfold with a progress function.
    #[must_use]
    pub fn with_progress(mut self, progress: ProgressFn<'a>) -> Self {
        self.progress = Some(progress);
        self
    }

    /// Unfolds the container that starts at the position of `input` and ends at its end, into
    /// `out`. Before any output, it checks the header, the model, the whole table and the
    /// input length (ADR 0018 §3.2). The output is complete and correct only if the function
    /// returns `Ok`: the checksum is known at the end.
    ///
    /// # Errors
    ///
    /// - The errors of [`crate::unfold_with`].
    /// - [`StreamError::BlockSizeLimit`] if the block size of the header is more than the
    ///   limit.
    /// - [`StreamError::Io`] if a read or a write fails.
    pub fn from_seekable<R: Read + Seek, W: Write + ?Sized>(
        &self,
        mut input: R,
        out: &mut W,
    ) -> Result<UnfoldStats> {
        let err = io_err("read the input");
        let base = input.stream_position().map_err(&err)?;
        let end = input.seek(SeekFrom::End(0)).map_err(&err)?;
        input.seek(SeekFrom::Start(base)).map_err(&err)?;
        let input_len = end.saturating_sub(base);
        let header = read_header(&mut input, Some(input_len))?;
        self.check(&header)?;
        let mut source = SeekSource::new(input, &header, base, input_len)?;
        self.run(&header, &mut source, out, Some(input_len))
    }

    /// Unfolds the container of `input` (a pipe) into `out` in one pass. A missing byte or an
    /// extra byte is an error where the reader finds it, so `out` can hold restored bytes
    /// before the error. The output is complete and correct only if the function returns
    /// `Ok`.
    ///
    /// # Errors
    ///
    /// The errors of [`StreamUnfold::from_seekable`].
    pub fn from_reader<R: Read, W: Write + ?Sized>(
        &self,
        input: R,
        out: &mut W,
    ) -> Result<UnfoldStats> {
        let mut input = BufReader::with_capacity(IO_BUFFER, input);
        let header = read_header(&mut input, None)?;
        self.check(&header)?;
        let mut source = PipeSource::new(input, &header, len64(header.byte_len()))?;
        self.run(&header, &mut source, out, None)
    }

    fn check(&self, header: &Header) -> Result<()> {
        self.limits.check(header.block_size())?;
        checked_model(header, self.model, &self.options)?;
        Ok(())
    }

    fn run<S: Source, W: Write + ?Sized>(
        &self,
        header: &Header,
        source: &mut S,
        out: &mut W,
        input_len: Option<u64>,
    ) -> Result<UnfoldStats> {
        let model = checked_model(header, self.model, &self.options)?;
        let segment = segment_blocks(header.block_size());
        let mut table = TableReader::new(header)?;
        let mut checksum = ChecksumBuilder::new();
        let mut restored = 0u64;
        let mut entries = Vec::with_capacity(segment);
        while table.remaining() > 0 {
            entries.clear();
            source.next_entries(&mut table, segment, &mut entries)?;
            let mut stored = Vec::with_capacity(entries.len());
            for e in &entries {
                let mut buf = Vec::new();
                source.read_stored(e, &mut buf)?;
                stored.push(buf);
            }
            let refs = entries
                .iter()
                .zip(&stored)
                .map(|(e, s)| e.block_ref(s))
                .collect::<core::result::Result<Vec<_>, _>>()?;
            let parts = decode_blocks(self.engine, model, header, &refs, self.options.threads)?;
            for p in &parts {
                checksum.update(p);
                out.write_all(p).map_err(io_err("write the output"))?;
                restored += len64(p.len());
            }
            if let Some(f) = self.progress {
                f(Progress { input_bytes: source.bytes_read(), output_bytes: restored, input_len });
            }
        }
        source.finish()?;
        out.flush().map_err(io_err("write the output"))?;
        if restored != header.original_len() {
            return Err(FormatError::RestoredLengthMismatch {
                expected: header.original_len(),
                found: restored,
            }
            .into());
        }
        let found = checksum.finish();
        if found != header.checksum() {
            return Err(checksum_error(FormatError::ChecksumMismatch {
                expected: header.checksum(),
                found,
            })
            .into());
        }
        Ok(UnfoldStats {
            original_len: restored,
            container_len: table.container_len(),
            blocks: u64::from(header.block_count()),
            version: header.version(),
        })
    }
}

/// Reads and checks the header at the position of `input`: 88 bytes (version 1) or 104 bytes
/// (version 2). `input_len` is the input length if it is known, for the error.
fn read_header<R: Read>(input: &mut R, input_len: Option<u64>) -> Result<Header> {
    let err = io_err("read the input");
    let mut bytes = [0u8; HEADER_LEN];
    let (prefix, _) = bytes.split_at_mut(6);
    let n = read_full(input, prefix).map_err(&err)?;
    let len = Header::len_from_prefix(prefix.get(..n).unwrap_or(&[]))?;
    let rest = bytes.get_mut(6..len).unwrap_or(&mut []);
    let m = read_full(input, rest).map_err(&err)?;
    let got = bytes.get(..6 + m).unwrap_or(&[]);
    match Header::from_bytes(got) {
        Err(FormatError::Truncated { part, needed, found }) => {
            Err(FormatError::Truncated { part, needed, found: input_len.unwrap_or(found) }.into())
        }
        other => Ok(other?),
    }
}
