"""Extract the Traditional Chinese Handwriting Dataset on Windows.

The full release is published as ``all_data.zip.001`` ... ``all_data.zip.012``.
Filenames inside the archive are Big5, so Explorer and a default Python unzip
turn the character folders into mojibake. This script joins the parts in order
and decodes the names as Big5 (Python 3.11+).

Examples:

    python scripts/extract_dataset.py --parts-dir D:\\handwritting_data_all --output D:\\data
    python scripts/extract_dataset.py --zip D:\\cleaned_data.zip --output D:\\data --organize
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
_FLAT_NAME = re.compile(r"^(.*)_(\d+)$", re.IGNORECASE)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract the handwriting dataset with Big5 filenames.")
    parser.add_argument("--parts-dir", type=Path, help="Directory that contains all_data.zip.001, .002, ...")
    parser.add_argument("--zip", dest="zips", type=Path, action="append", default=[], help="A zip file to extract. Repeatable.")
    parser.add_argument("--output", type=Path, required=True, help="Directory that will contain cleaned_data.")
    parser.add_argument("--encoding", default="big5", help="Zip filename encoding when the UTF-8 flag is not set.")
    parser.add_argument(
        "--organize",
        action="store_true",
        help="Move loose <character>_<number>.png files into one folder per character.",
    )
    args = parser.parse_args(argv)

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    archives = list(args.zips)
    if args.parts_dir:
        archives.append(combine_parts(args.parts_dir.expanduser().resolve(), output))
    if not archives:
        raise SystemExit("Pass --parts-dir or at least one --zip.")

    for archive in archives:
        extract_zip(archive, output, args.encoding)
    if args.organize:
        moved = organize_flat_images(output)
        print(f"Moved {moved} loose images into character folders.")
    print(f"Dataset root: {output}")
    print("Point training at the cleaned_data folder inside it if that folder was created.")


def combine_parts(parts_dir: Path, output: Path) -> Path:
    parts = sorted(parts_dir.glob("all_data.zip.*"))
    if not parts:
        raise FileNotFoundError(f"No all_data.zip.* parts in {parts_dir}")
    destination = output / "all_data.zip"
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Using existing combined archive {destination}")
        return destination
    print(f"Joining {len(parts)} parts into {destination}")
    with destination.open("wb") as combined:
        for part in parts:
            print(f"  {part.name}")
            with part.open("rb") as source:
                while True:
                    chunk = source.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    combined.write(chunk)
    return destination


def extract_zip(archive: Path, output: Path, encoding: str) -> None:
    print(f"Extracting {archive}")
    try:
        zipped = zipfile.ZipFile(archive, metadata_encoding=encoding)
    except TypeError:
        zipped = zipfile.ZipFile(archive)
    count = 0
    with zipped:
        for info in zipped.infolist():
            name = info.filename.replace("\\", "/").lstrip("/")
            if not name or name.endswith("/"):
                continue
            destination = _safe_destination(output, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(info) as source, destination.open("wb") as target:
                while True:
                    chunk = source.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
            count += 1
            if count % 5000 == 0:
                print(f"  {count} files")
    print(f"  extracted {count} files")


def organize_flat_images(root: Path) -> int:
    """Group ``character_12.png`` files that sit directly inside a directory."""

    directories = [root, *[path for path in root.rglob("*") if path.is_dir()]]
    moved = 0
    for directory in directories:
        loose = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        for image in loose:
            match = _FLAT_NAME.match(image.stem)
            if not match or not match.group(1):
                continue
            label = match.group(1)
            if image.parent.name == label:
                continue
            target_dir = directory / label
            target_dir.mkdir(exist_ok=True)
            target = target_dir / image.name
            if target.exists():
                continue
            image.rename(target)
            moved += 1
    return moved


def _safe_destination(root: Path, name: str) -> Path:
    parts = Path(name).parts
    if Path(name).is_absolute() or ".." in parts:
        raise ValueError(f"Unsafe path inside zip: {name}")
    destination = (root / name).resolve()
    destination.relative_to(root.resolve())
    return destination


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
