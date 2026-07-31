#!/usr/bin/env python3
"""Build docs/demo.gif from frames captured out of the real TUI.

The frames are not a screen recording and nothing here is hand-drawn: `capture_demo_frames`
(see crates/cli/src/demo.rs) drives the actual `App` with actual keystrokes against a real
index, renders each step through the same code the terminal uses, and writes the resulting
cells out as ANSI. This script only paints them.

That means the animation cannot drift from the product — change the layout and the next
regeneration shows the change.

Full recipe:

    cargo run --release -- --cache /tmp/ds-demo.bin index .
    DEMO_CACHE=/tmp/ds-demo.bin \\
        cargo test --release --bin deepsearch capture_demo_frames -- --ignored
    python3 docs/make-demo.py

Requires ImageMagick with the Pango delegate (`magick -list format | grep PANGO`).
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# A neutral dark terminal. The TUI itself only ever emits ANSI slots — it borrows whatever
# palette the terminal has — so this file is where those slots get concrete colours.
BACKGROUND = "#0d1117"
FOREGROUND = "#c9d1d9"
FONT = "JetBrainsMono Nerd Font Mono"
POINTSIZE = 13

PALETTE = {
    30: "#484f58", 31: "#ff7b72", 32: "#7ee787", 33: "#e3b341",
    34: "#79c0ff", 35: "#d2a8ff", 36: "#56d4dd", 37: "#b1bac4",
    90: "#6e7681", 91: "#ffa198", 92: "#56d364", 93: "#f2cc60",
    94: "#a5d6ff", 95: "#e2c5ff", 96: "#b3f0ff", 97: "#f0f6fc",
}

SGR = re.compile(r"\x1b\[([0-9;]*)m")


class Style:
    """The subset of SGR the TUI emits."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.bold = self.dim = self.italic = self.underline = self.reverse = False
        self.fg: str | None = None
        self.bg: str | None = None

    def apply(self, codes: list[int]) -> None:
        for code in codes or [0]:
            if code == 0:
                self.reset()
            elif code == 1:
                self.bold = True
            elif code == 2:
                self.dim = True
            elif code == 3:
                self.italic = True
            elif code == 4:
                self.underline = True
            elif code == 7:
                self.reverse = True
            elif code == 39:
                self.fg = None
            elif code == 49:
                self.bg = None
            elif code in PALETTE:
                self.fg = PALETTE[code]
            elif code - 10 in PALETTE:
                self.bg = PALETTE[code - 10]

    def wrap(self, text: str) -> str:
        if not text:
            return ""
        escaped = html.escape(text, quote=False)

        fg = self.fg or FOREGROUND
        bg = self.bg or BACKGROUND
        if self.reverse:
            fg, bg = bg, fg
        # `dim` has no Pango equivalent; fade the foreground towards the background instead,
        # which is what a terminal does anyway.
        if self.dim and not self.reverse:
            fg = blend(fg, bg, 0.45)

        attrs = f'foreground="{fg}"'
        if bg != BACKGROUND:
            attrs += f' background="{bg}"'
        out = f"<span {attrs}>{escaped}</span>"
        if self.bold:
            out = f"<b>{out}</b>"
        if self.italic:
            out = f"<i>{out}</i>"
        if self.underline:
            out = f"<u>{out}</u>"
        return out


def blend(fg: str, bg: str, amount: float) -> str:
    """Move `fg` `amount` of the way towards `bg`."""

    def channels(color: str) -> tuple[int, int, int]:
        c = color.lstrip("#")
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)

    f, b = channels(fg), channels(bg)
    mixed = tuple(round(f[i] + (b[i] - f[i]) * amount) for i in range(3))
    return "#%02x%02x%02x" % mixed


def ansi_to_pango(text: str) -> str:
    out: list[str] = []
    style = Style()
    position = 0
    for match in SGR.finditer(text):
        out.append(style.wrap(text[position : match.start()]))
        codes = [int(c) for c in match.group(1).split(";") if c.isdigit()] or [0]
        style.apply(codes)
        position = match.end()
    out.append(style.wrap(text[position:]))
    return "".join(out)


def render(markup: str, path: Path, scratch: Path) -> None:
    # The markup goes through a file rather than the command line: ImageMagick runs an argument
    # through its own property interpreter first, which mangles the escaped ampersands that any
    # source-code preview is full of.
    scratch.write_text(markup)
    subprocess.run(
        [
            "magick",
            "-background", BACKGROUND,
            "-font", FONT,
            "-pointsize", str(POINTSIZE),
            f"pango:@{scratch}",
            "-bordercolor", BACKGROUND,
            "-border", "16x12",
            str(path),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, default=Path("docs/frames"))
    parser.add_argument("--out", type=Path, default=Path("docs/demo.gif"))
    parser.add_argument("--colors", type=int, default=128)
    args = parser.parse_args()

    if not shutil.which("magick"):
        print("error: ImageMagick (`magick`) is required", file=sys.stderr)
        return 1
    manifest = args.frames / "manifest.txt"
    if not manifest.exists():
        print(f"error: {manifest} not found — capture the frames first", file=sys.stderr)
        return 1

    entries = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        name, delay = line.rsplit(" ", 1)
        entries.append((args.frames / name, int(delay)))

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        pngs = []
        for index, (frame, _) in enumerate(entries):
            markup = ansi_to_pango(frame.read_text().rstrip("\n"))
            png = tmpdir / f"f{index:04d}.png"
            render(markup, png, tmpdir / "markup.txt")
            pngs.append(png)

        command = ["magick", "-loop", "0"]
        for png, (_, delay) in zip(pngs, entries):
            command += ["-delay", str(delay), str(png)]
        command += ["-layers", "OptimizeTransparency", "-colors", str(args.colors), str(args.out)]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)

    seconds = sum(delay for _, delay in entries) / 100
    size_kb = args.out.stat().st_size / 1024
    print(f"{args.out}: {len(entries)} frames, {seconds:.1f}s, {size_kb:.0f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
