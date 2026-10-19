//! Output files that get their name only when they are complete (ADR 0018 §2 and §3.4).

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::CliResult;

/// The size of the write buffer of an output file: 1 MiB.
const BUFFER: usize = 1024 * 1024;

/// The directory of `path`: its parent, or `.` for a bare name.
pub(crate) fn dir_of(path: &Path) -> &Path {
    match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    }
}

/// An output file under a temporary name in the directory of the output. [`PendingFile::commit`]
/// gives it the output name. If the value is dropped before, the file is deleted. Thus an
/// error never leaves a partial or a wrong file with the output name.
pub(crate) struct PendingFile {
    tmp: PathBuf,
    dest: PathBuf,
    file: BufWriter<File>,
    done: bool,
}

impl PendingFile {
    /// Makes the temporary file for the output `dest`. Without `force`, refuses an output
    /// that exists.
    pub(crate) fn create(dest: &Path, force: bool) -> CliResult<Self> {
        if !force && dest.exists() {
            return Err(format!("{} exists: use -f to replace it", dest.display()).into());
        }
        let name = dest.file_name().ok_or(format!("{} is not a file name", dest.display()))?;
        let dir = dir_of(dest);
        let mut last = None;
        for n in 0..1000u32 {
            let mut tmp_name = std::ffi::OsString::from(".");
            tmp_name.push(name);
            tmp_name.push(format!(".partial-{}-{n}", std::process::id()));
            let tmp = dir.join(tmp_name);
            match OpenOptions::new().read(true).write(true).create_new(true).open(&tmp) {
                Ok(f) => {
                    return Ok(Self {
                        tmp,
                        dest: dest.to_path_buf(),
                        file: BufWriter::with_capacity(BUFFER, f),
                        done: false,
                    });
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => last = Some(e),
                Err(e) => return Err(format!("cannot write {}: {e}", tmp.display()).into()),
            }
        }
        Err(format!("cannot write in {}: {:?}", dir.display(), last).into())
    }

    /// The writer of the temporary file.
    pub(crate) fn writer(&mut self) -> &mut BufWriter<File> {
        &mut self.file
    }

    /// The directory of the output, for the temporary files of a spool.
    pub(crate) fn dir(&self) -> PathBuf {
        dir_of(&self.dest).to_path_buf()
    }

    /// Writes the buffer, and gives the file the output name.
    pub(crate) fn commit(mut self) -> CliResult<()> {
        let fail = |e: io::Error| format!("cannot write {}: {e}", self.dest.display());
        self.file.flush().map_err(fail)?;
        self.file.get_ref().sync_all().map_err(fail)?;
        std::fs::rename(&self.tmp, &self.dest).map_err(fail)?;
        self.done = true;
        Ok(())
    }
}

impl Drop for PendingFile {
    fn drop(&mut self) {
        // NOTE: a temporary file that cannot be deleted does not change the result. Its name
        // starts with "." and ends with ".partial-<pid>-<n>", so a person can find it.
        if !self.done && std::fs::remove_file(&self.tmp).is_err() {}
    }
}
