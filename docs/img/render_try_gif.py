"""Render the README GIF (docs/img/try-3os.gif) from real ./try output captured on macOS, Linux
and Windows (docs/img/try-output/, 2026-10-20, release 0.1.0-alpha.1).

    python3 -I docs/img/render_try_gif.py docs/img/try-3os.gif   # macOS: needs Pillow and Menlo

Long taco lines are shortened with "…" (paths and statistics); every value shown is from the
captured output files next to this script.
"""
import sys
from PIL import Image, ImageDraw, ImageFont

OUT = sys.argv[1]
H = "2fab1ef7a1c70cd4e39ddfd88a8d66389ae07a403111db131326ae425460652e"

PANELS = [
    ("macOS  ·  Apple M2  ·  backend: CPU", "~/taco-demo-0.1.0 $ ", "./try", [
        "taco: fold sample.txt … 3672 -> 648 bytes … backend cpu (CPU), 3.2s",
        "taco: unfold … 648 -> 3672 bytes, checksum verified, backend cpu (CPU), 6.8s",
        "PASS: the file came back byte for byte.",
        "SHA-256 of the .taco file: " + H,
    ]),
    ("Linux  ·  NVIDIA RTX 3060  ·  backend: CUDA", "~/taco-demo-0.1.0 $ ", "./try", [
        "taco: fold sample.txt … 3672 -> 648 bytes … backend cuda (NVIDIA GeForce RTX 3060 Laptop GPU), 2.4s",
        "taco: unfold … 648 -> 3672 bytes, checksum verified, backend cuda (…), 2.3s",
        "PASS: the file came back byte for byte.",
        "SHA-256 of the .taco file: " + H,
    ]),
    ("Windows  ·  NVIDIA RTX 3060  ·  backend: CUDA", "C:\\taco-demo-0.1.0> ", "try.cmd", [
        "taco: fold sample.txt … 3672 -> 648 bytes … backend cuda (NVIDIA GeForce RTX 3060 Laptop GPU), 821.6ms",
        "taco: unfold … 648 -> 3672 bytes, checksum verified, backend cuda (…), 1.9s",
        "PASS: the file came back byte for byte.",
        "SHA-256 of the .taco file: " + H,
    ]),
]

W, PAD, LINE = 1060, 14, 19
PANEL_H = 30 + LINE * 5 + 16
TOP = 44
HEIGHT = TOP + len(PANELS) * (PANEL_H + PAD) + 40
BG, PANEL_BG, BAR = (246, 246, 243), (24, 26, 30), (44, 47, 54)
FG, DIM, GREEN, YELLOW, ORANGE = (220, 223, 228), (140, 146, 156), (110, 214, 120), (240, 200, 90), (255, 102, 0)

mono = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 12)
bold = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 13, index=1)
title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)


def frame(state, done):
    """state[i] = (typed chars of the command, lines shown) for panel i."""
    im = Image.new("RGB", (W, HEIGHT), BG)
    d = ImageDraw.Draw(im)
    d.text((PAD, 12), "Taco: the same compressed bytes on every machine", font=title, fill=(30, 30, 30))
    y = TOP
    for (head, prompt, cmd, lines), (typed, shown) in zip(PANELS, state):
        d.rounded_rectangle((PAD, y, W - PAD, y + PANEL_H), 8, fill=PANEL_BG)
        d.rounded_rectangle((PAD, y, W - PAD, y + 24), 8, fill=BAR)
        d.rectangle((PAD, y + 16, W - PAD, y + 24), fill=BAR)
        for k, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
            d.ellipse((PAD + 10 + k * 16, y + 7, PAD + 20 + k * 16, y + 17), fill=c)
        d.text((PAD + 70, y + 5), head, font=bold, fill=FG)
        ty = y + 30
        d.text((PAD + 10, ty), prompt, font=mono, fill=GREEN)
        px = PAD + 10 + d.textlength(prompt, font=mono)
        d.text((px, ty), cmd[:typed], font=mono, fill=FG)
        for j, line in enumerate(lines[:shown]):
            ty += LINE
            if line.startswith("SHA-256"):
                lab = "SHA-256 of the .taco file: "
                d.text((PAD + 10, ty), lab, font=mono, fill=FG)
                hx = PAD + 10 + d.textlength(lab, font=mono)
                col = YELLOW if done else FG
                d.text((hx, ty), H, font=bold if done else mono, fill=col)
            else:
                d.text((PAD + 10, ty), line, font=mono, fill=GREEN if line.startswith("PASS") else DIM)
        y += PANEL_H + PAD
    if done:
        d.text((PAD, y + 4), "Result: macOS CPU, Linux CUDA and Windows CUDA folded the same file, with an identical SHA-256.",
               font=small, fill=(20, 120, 40))
    else:
        d.text((PAD, y + 4), "Real ./try output from the 0.1.0 release (long lines shortened with …).",
               font=small, fill=(110, 110, 110))
    return im


frames, durations = [], []
state = [(0, 0)] * 3
frames.append(frame(state, False)); durations.append(700)
for i, (_, _, cmd, lines) in enumerate(PANELS):
    for t in range(1, len(cmd) + 1):
        state[i] = (t, 0); frames.append(frame(state, False)); durations.append(70)
    for s in range(1, len(lines) + 1):
        state[i] = (len(cmd), s); frames.append(frame(state, False)); durations.append(450 if s < len(lines) else 700)
frames.append(frame(state, True)); durations.append(5000)

pal = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
pal[0].save(OUT, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True)
print(OUT, len(frames), "frames")
