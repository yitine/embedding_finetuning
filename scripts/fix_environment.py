"""Check and repair the NumPy and HuggingFace datasets environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from packaging.version import Version
except ImportError:  # pragma: no cover - packaging is installed with pip
    Version = None


NUMPY_SPEC = "numpy>=1.24.0,<2.0.0"
DATASETS_SPEC = "datasets>=2.14.0"
MANUAL_FIX = (
    "pip uninstall numpy datasets -y && "
    'pip install "numpy>=1.24.0,<2.0.0" "datasets>=2.14.0"'
)


def installed_version(package_name: str) -> str | None:
    """Read an installed distribution version without importing the package."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def version_in_range(version: str | None, minimum: str, maximum: str | None = None) -> bool:
    if version is None:
        return False
    if Version is None:
        raise RuntimeError("The 'packaging' package is required to compare versions")
    parsed = Version(version)
    if parsed < Version(minimum):
        return False
    return maximum is None or parsed < Version(maximum)


def run_pip(*arguments: str) -> None:
    command = [sys.executable, "-m", "pip", *arguments]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def repair_numpy() -> None:
    version = installed_version("numpy")
    print(f"NumPy version: {version or 'not installed'}")
    if not version_in_range(version, "1.24.0", "2.0.0"):
        print(f"Installing compatible NumPy ({NUMPY_SPEC})")
        if version is not None:
            run_pip("uninstall", "numpy", "-y")
        run_pip("install", NUMPY_SPEC)


def repair_datasets() -> None:
    version = installed_version("datasets")
    print(f"datasets version: {version or 'not installed'}")
    if not version_in_range(version, "2.14.0"):
        print(f"Installing compatible datasets ({DATASETS_SPEC})")
        run_pip("install", "--upgrade", DATASETS_SPEC)


def clear_huggingface_cache() -> None:
    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
    if not cache_dir.exists():
        print(f"HuggingFace datasets cache does not exist: {cache_dir}")
        return

    removed = 0
    for child in cache_dir.iterdir():
        if child.is_symlink():
            child.unlink()
            removed += 1
        elif child.is_dir():
            shutil.rmtree(child)
            removed += 1
        elif child.is_file():
            child.unlink()
            removed += 1
    print(f"Removed {removed} HuggingFace dataset cache entries from {cache_dir}")


def verify_imports() -> bool:
    """Verify imports in a fresh interpreter after any pip operation."""
    print("Verifying imports in a fresh Python process...")
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import numpy; import datasets; "
            "print(f'numpy={numpy.__version__} datasets={datasets.__version__}')",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if check.returncode == 0:
        print(f"  OK: {check.stdout.strip()}")
        return True

    print("  FAILED:")
    print((check.stderr or check.stdout).strip())
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Remove entries under ~/.cache/huggingface/datasets",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        repair_numpy()
        repair_datasets()
        if args.clear_cache:
            clear_huggingface_cache()

        if verify_imports():
            print("Environment check passed.")
            return 0

        print("Import verification failed; reinstalling both packages once.")
        run_pip("uninstall", "numpy", "datasets", "-y")
        run_pip("install", NUMPY_SPEC, DATASETS_SPEC)
        if verify_imports():
            print("Environment repaired successfully.")
            return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Environment repair failed: {error}")

    print("Manual fallback:")
    print(MANUAL_FIX)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
