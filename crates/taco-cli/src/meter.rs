//! The progress line on stderr: bytes done, the total if it is known, and the speed.
//!
//! The line is printed only when stderr is a terminal and `--quiet` is not given, at most 4
//! times in each second. The arithmetic uses integers only.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use taco::stream::Progress;

/// The shortest time between two lines.
const PERIOD: Duration = Duration::from_millis(250);

/// `bytes` as megabytes with one decimal: "12.3 MB" (1 MB = 1,000,000 bytes).
fn mb(bytes: u64) -> String {
    let tenths = bytes / 100_000;
    format!("{}.{} MB", tenths / 10, tenths % 10)
}

/// The progress line of one fold or unfold.
pub(crate) struct Meter {
    on: bool,
    verb: &'static str,
    total: Option<u64>,
    start: Instant,
    /// The time of the last line, and the length of the last line (0: no line yet).
    last: Mutex<(Instant, usize)>,
}

impl Meter {
    /// A meter that prints only if `on`. `total` is the input length, if it is known.
    pub(crate) fn new(on: bool, verb: &'static str, total: Option<u64>) -> Self {
        let start = Instant::now();
        Self { on, verb, total, start, last: Mutex::new((start, 0)) }
    }

    /// Prints the line for `p` if the last line is older than 250 ms.
    pub(crate) fn update(&self, p: Progress) {
        if !self.on {
            return;
        }
        let Ok(mut last) = self.last.lock() else { return };
        let now = Instant::now();
        if last.1 > 0 && now.duration_since(last.0) < PERIOD {
            return;
        }
        let ms = u64::try_from(now.duration_since(self.start).as_millis()).unwrap_or(u64::MAX);
        let speed = p.input_bytes.saturating_mul(1000) / ms.max(1);
        let done = match self.total.or(p.input_len) {
            Some(t) if t > 0 => format!(
                "{} of {} ({} %)",
                mb(p.input_bytes),
                mb(t),
                u128::from(p.input_bytes) * 100 / u128::from(t)
            ),
            _ => mb(p.input_bytes),
        };
        let line =
            format!("taco: {} {done}, {}/s, {} out", self.verb, mb(speed), mb(p.output_bytes));
        let pad = last.1.saturating_sub(line.len());
        eprint!("\r{line}{}", " ".repeat(pad));
        *last = (now, line.len());
    }

    /// Removes the line, so the summary starts at the left.
    pub(crate) fn clear(&self) {
        if let Ok(last) = self.last.lock()
            && last.1 > 0
        {
            eprint!("\r{}\r", " ".repeat(last.1));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::mb;

    #[test]
    fn megabytes_have_one_decimal() {
        assert_eq!(mb(0), "0.0 MB");
        assert_eq!(mb(99_999), "0.0 MB");
        assert_eq!(mb(12_345_678), "12.3 MB");
        assert_eq!(mb(200_000_000), "200.0 MB");
    }
}
