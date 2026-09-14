"""Install any selection of independent AIHEHUO skills into a backend source."""

import argparse
import json
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _catalog() -> dict[str, dict[str, str | list[str]]]:
    return json.loads((ROOT / "catalog.json").read_text(encoding="utf-8"))


def _files(name: str) -> dict[str, bytes]:
    source = ROOT / "packs" / name
    files = {"SKILL.md": (source / "SKILL.md").read_bytes(), "client.py": (ROOT / "client.py").read_bytes()}
    resources = _catalog()[name]["resources"]
    files["run.py"] = (
        '"""Standalone entrypoint for this installed AIHEHUO skill."""\n\n'
        "from client import main\n\n"
        f"if __name__ == '__main__':\n    raise SystemExit(main({resources!r}))\n"
    ).encode()
    files["package.json"] = json.dumps({"name": name, "version": "1.0.0", "resources": resources}, indent=2).encode() + b"\n"
    return files


def install(names: Sequence[str], *, target: Path, dry_run: bool = False) -> list[Path]:
    """Copy selected self-contained packages without overwriting existing skills.

    Args:
        names: One or more catalog skill identifiers.
        target: Skill source directory in an agent's filesystem backend.
        dry_run: Validate selection and conflicts without writing files.

    Returns:
        Installed or already identical skill directories.

    Raises:
        ValueError: A selection is empty or contains an unknown skill.
        FileExistsError: A destination differs or is a symlink.
        OSError: A filesystem operation fails.
    """
    catalog = _catalog()
    selection = list(dict.fromkeys(names))
    if not selection or set(selection) - catalog.keys():
        msg = "Select one or more known catalog skills"
        raise ValueError(msg)
    packages = {name: _files(name) for name in selection}
    # Preflight all known conflicts before installing any selected package.
    for name, files in packages.items():
        destination = target / name
        if destination.is_symlink() or (destination.exists() and not _identical(destination, files)):
            msg = f"Refusing to overwrite {destination}; choose an empty source directory"
            raise FileExistsError(msg)
    paths = [target / name for name in selection]
    if dry_run:
        return paths
    target.mkdir(parents=True, exist_ok=True)
    for name, files in packages.items():
        destination = target / name
        if destination.exists():
            continue
        staging = Path(tempfile.mkdtemp(prefix=".aihehuo-", dir=target))
        try:
            for filename, content in files.items():
                (staging / filename).write_bytes(content)
            # Rename publishes only a complete package; no partially copied skill.
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return paths


def _identical(destination: Path, files: dict[str, bytes]) -> bool:
    if not destination.is_dir():
        return False
    actual = {item.name for item in destination.iterdir() if item.name != "__pycache__"}
    return actual == files.keys() and all(
        (destination / name).is_file() and not (destination / name).is_symlink() and (destination / name).read_bytes() == content
        for name, content in files.items()
    )


def main(argv: Sequence[str] | None = None) -> int:
    """List packages or install a selected combination.

    Args:
        argv: Explicit command arguments; otherwise process arguments.

    Returns:
        Zero on success or one on installation failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    installer = commands.add_parser("install")
    installer.add_argument("names", nargs="*")
    installer.add_argument("--all", action="store_true")
    installer.add_argument("--target", type=Path, required=True)
    installer.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    catalog = _catalog()
    if args.command == "list":
        sys.stdout.write(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
        return 0
    if args.all and args.names:
        parser.error("Choose --all or explicit names, not both")
    try:
        paths = install(list(catalog) if args.all else args.names, target=args.target, dry_run=args.dry_run)
    except (ValueError, OSError) as error:
        sys.stderr.write(f"Installation failed: {error}\n")
        return 1
    sys.stdout.write(json.dumps([str(path) for path in paths], ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
