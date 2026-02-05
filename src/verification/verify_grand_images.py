from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class ImageInfo:
    path: Path
    width: int
    height: int


def _iter_pngs(images_dir: Path) -> list[Path]:
    return sorted([p for p in images_dir.glob("*.png") if p.is_file()])


def _read_size(path: Path) -> ImageInfo:
    with Image.open(path) as im:
        width, height = im.size
    return ImageInfo(path=path, width=width, height=height)


def _is_grand_intended(path: Path) -> bool:
    # Convention in this repo: filenames that contain "3160x2820" are meant to be "Grand".
    return "3160x2820" in path.name


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify plot resolution compliance for repo artifacts.")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/artifacts/images"),
        help="Directory containing PNG artifacts (default: data/artifacts/images).",
    )
    parser.add_argument("--width", type=int, default=3160, help="Expected width in pixels.")
    parser.add_argument("--height", type=int, default=2820, help="Expected height in pixels.")
    parser.add_argument(
        "--enforce-all",
        action="store_true",
        help="Fail if any PNG is not the expected resolution (default: only enforce 'grand-intended' files).",
    )
    args = parser.parse_args()

    images_dir: Path = args.images_dir
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    pngs = _iter_pngs(images_dir)
    infos = [_read_size(p) for p in pngs]

    expected = (args.width, args.height)
    grand_intended = [info for info in infos if _is_grand_intended(info.path)]
    grand_ok = [info for info in infos if (info.width, info.height) == expected]
    non_grand = [info for info in infos if (info.width, info.height) != expected]

    print(f"Checked: {len(infos)} PNGs in {images_dir}")
    print(f"Grand resolution: {expected[0]}x{expected[1]}")
    print(f"Grand-intended (name contains '3160x2820'): {len(grand_intended)}")
    print(f"Grand-compliant (any filename): {len(grand_ok)}")
    print(f"Non-grand: {len(non_grand)}")

    failures: list[ImageInfo] = []
    if args.enforce_all:
        failures = non_grand
    else:
        failures = [info for info in grand_intended if (info.width, info.height) != expected]

    if failures:
        print("\nFAIL: Non-compliant images:")
        for info in failures:
            print(f"- {info.path}: {info.width}x{info.height}")
        return 2

    # If not enforcing all, still provide a short list for visibility.
    if non_grand and not args.enforce_all:
        print("\nNOTE: Non-grand images exist (not enforced unless --enforce-all):")
        for info in non_grand[:20]:
            print(f"- {info.path.name}: {info.width}x{info.height}")
        if len(non_grand) > 20:
            print(f"... ({len(non_grand) - 20} more)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

