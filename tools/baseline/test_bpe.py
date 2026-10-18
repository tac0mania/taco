"""Round-trip tests for the BPE tokenizer (plan §10, §11.2).

Run: uv run --project tools python -m baseline.test_bpe
The tests use a fixed random seed. They need the model files in tools/data/model.
"""

from __future__ import annotations

import random

from baseline.bpe import EOT_ID, Tokenizer

MULTILINGUAL = [
    "Hello world",
    "Grüße aus München. Straße, Bäcker, Öl.",
    "Le cœur a ses raisons que la raison ne connaît point.",
    "हिन्दी एक भारतीय भाषा है।",
    "中文是世界上使用人数最多的语言之一。",
    "اللغة العربية لغة سامية.",
    "日本語のテキスト、カタカナとひらがな。",
    "Emoji: 🌮🌮 and a ZWJ family 👨‍👩‍👧",
    "dasdasdasdada " * 20,
    "  leading spaces\n\n\ttabs\r\nwindows lines   ",
    "fn main() { println!(\"{}\", 1 + 2); }",
    '{"key": [1, 2.5, null, true], "s": "\\u00e9"}',
    "<|endoftext|> is only text here",
]


def check(tok: Tokenizer, data: bytes) -> list[int]:
    ids = tok.encode(data)
    back = tok.decode(ids)
    assert back == data, f"round trip failed for {data[:40]!r}"
    assert EOT_ID not in ids, "the encoder emitted <|endoftext|>"
    assert all(0 <= i < tok.vocab_size for i in ids)
    return ids


def main() -> None:
    tok = Tokenizer()
    rng = random.Random(20261018)
    count = 0

    # Known GPT-2 token ids (published tokenizer output).
    assert tok.encode(b"Hello world") == [15496, 995], tok.encode(b"Hello world")
    assert tok.encode(b" Paris") == [6342], tok.encode(b" Paris")
    assert tok.encode(b"<|endoftext|>") != [EOT_ID]

    assert check(tok, b"") == []
    count += 1
    for b in range(256):
        assert check(tok, bytes([b])) == [tok.single_byte[b]]
        count += 1
    check(tok, bytes(range(256)))
    check(tok, bytes(range(255, -1, -1)) * 3)
    count += 2
    for text in MULTILINGUAL:
        check(tok, text.encode("utf-8"))
        count += 1
    # Random bytes, random letters, and valid UTF-8 cut at random places.
    for _ in range(2000):
        n = rng.randrange(0, 300)
        check(tok, bytes(rng.randrange(256) for _ in range(n)))
        check(tok, "".join(rng.choice("ab cd\n") for _ in range(n)).encode())
        text = rng.choice(MULTILINGUAL).encode("utf-8")
        i, j = sorted(rng.randrange(len(text) + 1) for _ in range(2))
        check(tok, text[i:j] + bytes([rng.randrange(256)]) + text[j:])
        count += 3
    # Invalid UTF-8 forms: surrogates, overlong, truncated, beyond U+10FFFF.
    for bad in [b"\xed\xa0\x80", b"\xc0\xaf", b"\xe2\x82", b"\xf4\x90\x80\x80", b"a\xffb", b"\x80"]:
        check(tok, bad)
        check(tok, b"text " + bad + b" more text")
        count += 2
    print(f"bpe: {count} round trips passed")


if __name__ == "__main__":
    main()
