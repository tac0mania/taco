//! The fixed inputs of the conformance run.
//!
//! DETERMINISM: every input comes from splitmix64 with a fixed seed and from text in this
//! file. No input depends on a file of the checkout, so line ends (Windows) and edits of other
//! files do not change them.

/// splitmix64: the generator of the Rust tests and of `tools/convert/tiny.py`.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A value in `0..n`. `n` is not 0.
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }

    /// One item of `items`, which is not empty.
    fn pick<T: Copy + Default>(&mut self, items: &[T]) -> T {
        let n = u64::try_from(items.len()).unwrap_or(u64::MAX);
        let i = usize::try_from(self.below(n)).unwrap_or(0);
        items.get(i).copied().unwrap_or_default()
    }
}

const SENTENCES: &[&str] = &[
    "The compressor reads the file in blocks of a fixed size.",
    "A language model predicts the next token from the tokens before it.",
    "The range coder keeps only the part that the model did not predict.",
    "Every machine must calculate the same predictions, bit for bit.",
    "If a block does not become smaller, the container stores it raw.",
    "The decoder verifies the checksum before it writes the output.",
    "Integer arithmetic gives one defined result on every processor.",
    "Do not use floating-point arithmetic in the prediction path.",
    "The model file is the model, and the container stores its hash.",
    "A test that fails is better than a silent wrong output.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "On Monday the team measured the speed again and wrote the numbers down.",
    "Open the valve slowly, then close it when the pressure is correct.",
    "She said that the weather in the mountains changes very quickly.",
    "Prices rose by 3.5 percent in the third quarter, the report said.",
];

const UTF8_WORDS: &[&str] = &[
    "naïve",
    "café",
    "façade",
    "Zürich",
    "smörgåsbord",
    "jalapeño",
    "crème brûlée",
    "東京",
    "Привет",
    "γειά",
    "שלום",
    "مرحبا",
    "🍮",
    "🌮",
    "—",
    "…",
    "€100",
    "½",
];

const LEVELS: &[&str] = &["INFO", "INFO", "INFO", "INFO", "DEBUG", "WARN", "ERROR"];
const PATHS: &[&str] =
    &["/api/v1/items", "/api/v1/users", "/health", "/static/app.js", "/login", "/search"];
const METHODS: &[&str] = &["GET", "GET", "GET", "POST", "PUT", "DELETE"];
const STATUS: &[u32] = &[200, 200, 200, 201, 204, 304, 404];

fn cut(mut data: Vec<u8>, n: usize) -> Vec<u8> {
    data.truncate(n);
    data
}

fn english(n: usize) -> Vec<u8> {
    let mut rng = SplitMix64(0x7ac0_0001);
    let mut out = String::new();
    let mut count = 0u32;
    while out.len() < n {
        out.push_str(rng.pick(SENTENCES));
        count += 1;
        out.push(if count.is_multiple_of(5) { '\n' } else { ' ' });
    }
    cut(out.into_bytes(), n)
}

fn utf8_text(n: usize) -> Vec<u8> {
    let mut rng = SplitMix64(0x7ac0_0002);
    let mut out = String::new();
    while out.len() < n {
        if rng.below(3) == 0 {
            out.push_str(rng.pick(UTF8_WORDS));
        } else {
            let words: Vec<&str> = rng.pick(SENTENCES).split(' ').collect();
            out.push_str(rng.pick(&words));
        }
        out.push(' ');
    }
    // NOTE: a cut in a character is intended: the tokenizer takes any bytes (plan §10).
    cut(out.into_bytes(), n)
}

fn server_log(n: usize) -> Vec<u8> {
    let mut rng = SplitMix64(0x7ac0_0003);
    let mut out = String::new();
    let mut ms = 0u64;
    while out.len() < n {
        ms += 1 + rng.below(900);
        let (sec, milli) = (ms / 1000, ms % 1000);
        let (minute, sec) = ((sec / 60) % 60, sec % 60);
        let level = rng.pick(LEVELS);
        let worker = rng.below(8);
        let path = rng.pick(PATHS);
        let text = match level {
            "ERROR" => {
                format!("request failed path={path} err=timeout after {}ms", 1000 + rng.below(4000))
            }
            "WARN" => format!("slow request path={path} took={}ms", 200 + rng.below(800)),
            _ => format!(
                "{} {path} status={} bytes={} ms={}",
                rng.pick(METHODS),
                rng.pick(STATUS),
                rng.below(20_000),
                rng.below(150)
            ),
        };
        let id = rng.next();
        out.push_str(&format!(
            "2026-10-19T08:{minute:02}:{sec:02}.{milli:03}Z {level:5} [worker-{worker}] id={id:016x} {text}\n"
        ));
    }
    cut(out.into_bytes(), n)
}

fn nginx_jsonl(n: usize) -> Vec<u8> {
    let mut rng = SplitMix64(0x7ac0_0004);
    let mut out = String::new();
    let mut t = 1_760_860_800u64;
    while out.len() < n {
        t += rng.below(3);
        let (a, b) = (rng.below(4), rng.below(256));
        let (method, uri) = (rng.pick(METHODS), rng.pick(PATHS));
        let status = rng.pick(&[200u32, 200, 200, 404, 500]);
        let (bytes, rt, ua) = (rng.below(65_536), rng.below(1000), rng.below(12));
        out.push_str(&format!(
            "{{\"time\":{t},\"remote_addr\":\"10.0.{a}.{b}\",\"method\":\"{method}\",\"uri\":\"{uri}\",\
             \"status\":{status},\"bytes_sent\":{bytes},\"request_time\":\"0.{rt:03}\",\
             \"user_agent\":\"curl/8.{ua}.0\"}}\n"
        ));
    }
    cut(out.into_bytes(), n)
}

fn random_bytes(n: usize) -> Vec<u8> {
    let mut rng = SplitMix64(20_261_018);
    (0..n).map(|_| rng.next().to_le_bytes()[0]).collect()
}

/// Fixed-width little-endian records: binary data that zstd compresses well.
fn records(n: usize) -> Vec<u8> {
    let mut rng = SplitMix64(0x7ac0_0005);
    let mut out = Vec::with_capacity(n + 16);
    let mut i = 0u32;
    while out.len() < n {
        out.extend_from_slice(&i.to_le_bytes());
        out.extend_from_slice(&(1000 + rng.below(50)).to_le_bytes()[..4]);
        out.extend_from_slice(&[rng.next().to_le_bytes()[0] & 3, 0, 0, 0]);
        out.extend_from_slice(b"ITEM");
        i += 1;
    }
    cut(out, n)
}

/// The inputs: (file name, bytes).
pub(crate) fn all() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("empty.bin", Vec::new()),
        ("dasdas.txt", b"dasdasdasdada".to_vec()),
        ("byte_cycle_1k.bin", (0..1024u32).map(|i| i.to_le_bytes()[0]).collect()),
        ("random_3000.bin", random_bytes(3000)),
        ("english_12k.txt", english(12 * 1024)),
        ("utf8_4k.txt", utf8_text(4 * 1024)),
        ("server_24k.log", server_log(24 * 1024)),
        ("nginx_8k.jsonl", nginx_jsonl(8 * 1024)),
        ("records_8k.bin", records(8 * 1024)),
    ]
}
