#!/usr/bin/env python3
"""Create HNNX application and web icon assets from a square source image."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw


def resize(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.Resampling.LANCZOS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()

    root = args.root.resolve()
    image = Image.open(args.source).convert("RGBA")
    edge = min(image.size)
    left = (image.width - edge) // 2
    top = (image.height - edge) // 2
    image = image.crop((left, top, left + edge, top + edge))
    image = resize(image, 1024)

    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, 1023, 1023), radius=205, fill=255)
    image.putalpha(mask)

    master = root / "publish" / "hnnx-icon.png"
    image.save(master, optimize=True)
    resize(image, 512).save(root / "publish" / "icon.png", optimize=True)
    resize(image, 256).save(root / "source" / "icon.png", optimize=True)

    ico_sizes = [
        (16, 16),
        (24, 24),
        (32, 32),
        (48, 48),
        (64, 64),
        (128, 128),
        (256, 256),
    ]
    image.save(root / "publish" / "icon.ico", sizes=ico_sizes)
    image.save(root / "source" / "favicon.ico", sizes=ico_sizes[:5])

    with tempfile.TemporaryDirectory(prefix="hnnx-icon-") as directory:
        iconset = Path(directory) / "icon.iconset"
        iconset.mkdir()
        for size in (16, 32, 128, 256, 512):
            resize(image, size).save(iconset / f"icon_{size}x{size}.png")
            resize(image, size * 2).save(iconset / f"icon_{size}x{size}@2x.png")
        subprocess.run(
            [
                "iconutil",
                "-c",
                "icns",
                str(iconset),
                "-o",
                str(root / "publish" / "icon.icns"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
