//! Runs one job for each block on several threads, with results in block order.
//!
//! DETERMINISM: each block is independent (plan §9), and each result goes to the position of
//! its block. Thus the output does not depend on the number of threads or on the schedule. If
//! several blocks fail, the error of the block with the lowest index is the result, for every
//! number of threads: a worker never skips a block below the lowest failed index that it
//! knows, and each worker takes its blocks in increasing order.

use core::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::error::{Result, TacoError};

/// The number of worker threads: at least 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Threads(NonZeroUsize);

impl Threads {
    /// One thread: the work runs on the thread of the caller.
    pub const ONE: Self = Self(NonZeroUsize::MIN);

    /// `n` threads, or `None` if `n` is 0.
    #[must_use]
    pub const fn new(n: usize) -> Option<Self> {
        match NonZeroUsize::new(n) {
            Some(n) => Some(Self(n)),
            None => None,
        }
    }

    /// The number of threads that the machine can run at the same time, or 1 if the system
    /// does not say. The output of Taco does not depend on this value.
    #[must_use]
    pub fn available() -> Self {
        Self(std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN))
    }

    /// The number of threads.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0.get()
    }
}

impl Default for Threads {
    fn default() -> Self {
        Self::ONE
    }
}

/// Runs `job(state, i)` for each `i` in `0..count`, and returns the results in index order.
///
/// Each worker makes its own state with `make_state` (for example a HardShell session) and
/// takes the indices `w, w + n, w + 2n, ...`. With one thread, or with fewer than two jobs, the
/// work runs on the thread of the caller.
///
/// # Errors
///
/// The error of the job with the lowest index that failed, or [`TacoError::WorkerFailed`] if a
/// worker thread stopped without a result.
pub(crate) fn map_blocks<S, T, M, J>(
    count: usize,
    threads: Threads,
    make_state: M,
    job: J,
) -> Result<Vec<T>>
where
    M: Fn() -> S + Sync,
    J: Fn(&mut S, usize) -> Result<T> + Sync,
    T: Send,
{
    let workers = threads.get().min(count);
    if workers <= 1 {
        let mut state = make_state();
        return (0..count).map(|i| job(&mut state, i)).collect();
    }
    let lowest_failed = AtomicUsize::new(usize::MAX);
    let per_worker: Vec<Vec<(usize, Result<T>)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|w| {
                let (make_state, job, lowest_failed) = (&make_state, &job, &lowest_failed);
                scope.spawn(move || {
                    let mut state = make_state();
                    let mut done = Vec::new();
                    for i in (w..count).step_by(workers) {
                        if i > lowest_failed.load(Ordering::Relaxed) {
                            break;
                        }
                        let r = job(&mut state, i);
                        let failed = r.is_err();
                        done.push((i, r));
                        if failed {
                            lowest_failed.fetch_min(i, Ordering::Relaxed);
                            break;
                        }
                    }
                    done
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().map_err(|_| TacoError::WorkerFailed))
            .collect::<Result<Vec<_>>>()
    })?;
    let mut slots: Vec<Option<T>> = (0..count).map(|_| None).collect();
    let mut first_error: Option<(usize, TacoError)> = None;
    for (i, r) in per_worker.into_iter().flatten() {
        match r {
            Ok(v) => {
                if let Some(slot) = slots.get_mut(i) {
                    *slot = Some(v);
                }
            }
            Err(e) => {
                if first_error.as_ref().is_none_or(|(j, _)| i < *j) {
                    first_error = Some((i, e));
                }
            }
        }
    }
    if let Some((_, e)) = first_error {
        return Err(e);
    }
    slots.into_iter().map(|s| s.ok_or(TacoError::WorkerFailed)).collect()
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use super::*;

    #[test]
    fn results_are_in_index_order_for_every_thread_count() {
        for n in 1..6 {
            let out = map_blocks(17, Threads::new(n).unwrap(), || 0u32, |_, i| Ok(i * 3)).unwrap();
            assert_eq!(out, (0..17).map(|i| i * 3).collect::<Vec<_>>());
        }
    }

    #[test]
    fn the_lowest_failed_index_is_the_error() {
        for n in 1..6 {
            let r: Result<Vec<usize>> = map_blocks(
                20,
                Threads::new(n).unwrap(),
                || (),
                |_, i| {
                    if i == 7 || i == 13 {
                        Err(TacoError::CorruptBlock { block: i as u32, reason: "test" })
                    } else {
                        Ok(i)
                    }
                },
            );
            assert_eq!(r, Err(TacoError::CorruptBlock { block: 7, reason: "test" }));
        }
    }
}
