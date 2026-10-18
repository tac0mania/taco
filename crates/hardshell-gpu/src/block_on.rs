//! A minimal executor: wait on one future in the current thread.
//!
//! wgpu gives futures for the adapter, the device and buffer reads. HardShell is synchronous,
//! so this module waits for them. It uses only the standard library and no `unsafe` code.

use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::thread::{self, Thread};

/// Wakes the thread that waits.
struct ThreadWaker(Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

/// Runs `fut` to completion on the current thread and gives its output.
pub(crate) fn block_on<F: Future>(fut: F) -> F::Output {
    let mut fut = pin!(fut);
    let waker = Waker::from(Arc::new(ThreadWaker(thread::current())));
    let mut cx = Context::from_waker(&waker);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            // NOTE: a spurious unpark only causes one more poll.
            Poll::Pending => thread::park(),
        }
    }
}
