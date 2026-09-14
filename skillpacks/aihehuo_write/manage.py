"""Install the write package independently of the six public read packages."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from skillpacks.aihehuo_write.adapter import MANIFEST, NAME

ROOT = Path(__file__).resolve().parent


def install(*, target: Path, dry_run: bool = False) -> Path:
    """Publish a complete write package without replacing existing content.

    Args:
        target: Write skill source on the host filesystem.
        dry_run: Validate conflicts without writing files.

    Returns:
        Installed or already identical package directory.

    Raises:
        FileExistsError: Existing content differs or uses symlinks.
        OSError: Files cannot be read or published.
    """
    files = {
        "SKILL.md": (ROOT / "packs" / NAME / "SKILL.md").read_bytes(),
        "adapter.py": (ROOT / "adapter.py").read_bytes(),
        "package.json": (json.dumps(MANIFEST, indent=2) + "\n").encode(),
    }
    destination = target / NAME
    if destination.is_symlink() or destination.exists():
        identical = destination.is_dir() and not destination.is_symlink()
        if identical:
            names = {path.name for path in destination.iterdir() if path.name != "__pycache__"}
            identical = names == files.keys() and all(
                (destination / name).is_file() and not (destination / name).is_symlink() and (destination / name).read_bytes() == data
                for name, data in files.items()
            )
        if not identical:
            msg = f"Refusing to overwrite {destination}; install to a new source directory"
            raise FileExistsError(msg)
        return destination
    if dry_run:
        return destination
    target.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".aihehuo-write-", dir=target))
    try:
        for name, data in files.items():
            (staging / name).write_bytes(data)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    """Install or preview the independently selectable write skill.

    Args:
        argv: Explicit arguments or the process command line.

    Returns:
        Zero on success, one on installation failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        path = install(target=args.target, dry_run=args.dry_run)
    except (ValueError, OSError) as error:
        print(f"Write skill installation failed: {error}")  # noqa: T201  # Installer CLI prints its explicit command result.
        return 1
    print(path)  # noqa: T201  # Installer CLI prints its explicit command result.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
