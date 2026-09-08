"""Download and install FFmpeg for Windows into the project tools/ directory.

Adds it to the current session PATH so subsequent scripts can find ffmpeg/ffprobe.
"""
from __future__ import annotations

import hashlib
import io
import os
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "tools"
FFMPEG_DIR = TOOLS_DIR / "ffmpeg"

# BtbN release: static GPL build for Windows x64
FFMPEG_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"


def download_with_progress(url: str) -> bytes:
    """Download a URL with simple progress reporting."""
    print(f"Downloading from:\n  {url}")
    req = Request(url, headers={"User-Agent": "Pramaan-X/1.0"})
    response = urlopen(req, timeout=300)
    total = int(response.headers.get("Content-Length", 0))
    downloaded = 0
    chunks = []
    while True:
        chunk = response.read(1024 * 1024)  # 1 MB chunks
        if not chunk:
            break
        chunks.append(chunk)
        downloaded += len(chunk)
        if total > 0:
            pct = downloaded / total * 100
            print(f"\r  {downloaded / 1024 / 1024:.1f} MB / {total / 1024 / 1024:.1f} MB ({pct:.0f}%)", end="", flush=True)
        else:
            print(f"\r  {downloaded / 1024 / 1024:.1f} MB downloaded", end="", flush=True)
    print()
    return b"".join(chunks)


def main() -> None:
    print("=" * 60)
    print("PRAMAAN-X: FFmpeg Installer for Windows")
    print("=" * 60)

    # Check if already installed
    existing = shutil.which("ffmpeg")
    if existing:
        print(f"\nFFmpeg already found on PATH: {existing}")
        print("Skipping download. If you want to re-download, remove it from PATH first.")
        return

    ffmpeg_bin = FFMPEG_DIR / "bin"
    ffmpeg_exe = ffmpeg_bin / "ffmpeg.exe"
    ffprobe_exe = ffmpeg_bin / "ffprobe.exe"

    if ffmpeg_exe.is_file() and ffprobe_exe.is_file():
        print(f"\nFFmpeg already installed at: {ffmpeg_bin}")
    else:
        # Download
        data = download_with_progress(FFMPEG_URL)
        print(f"  Download complete: {len(data) / 1024 / 1024:.1f} MB")

        # Extract
        print("Extracting...")
        TOOLS_DIR.mkdir(parents=True, exist_ok=True)

        # Clean previous install
        if FFMPEG_DIR.exists():
            shutil.rmtree(FFMPEG_DIR)

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            # Find the top-level directory name inside the zip
            top_dirs = {name.split("/")[0] for name in zf.namelist() if "/" in name}
            if len(top_dirs) != 1:
                raise RuntimeError(f"Unexpected zip structure: {top_dirs}")
            top_dir = top_dirs.pop()

            # Extract all
            zf.extractall(TOOLS_DIR)

            # Rename to our standard name
            extracted = TOOLS_DIR / top_dir
            extracted.rename(FFMPEG_DIR)

        if not ffmpeg_exe.is_file():
            raise RuntimeError(f"ffmpeg.exe not found after extraction at {ffmpeg_exe}")
        if not ffprobe_exe.is_file():
            raise RuntimeError(f"ffprobe.exe not found after extraction at {ffprobe_exe}")

        print(f"  Installed to: {FFMPEG_DIR}")

    # Add to current process PATH
    bin_str = str(ffmpeg_bin.resolve())
    current_path = os.environ.get("PATH", "")
    if bin_str not in current_path:
        os.environ["PATH"] = bin_str + os.pathsep + current_path

    # Verify
    ffmpeg_check = shutil.which("ffmpeg")
    ffprobe_check = shutil.which("ffprobe")
    if ffmpeg_check and ffprobe_check:
        print(f"\n✓ ffmpeg:  {ffmpeg_check}")
        print(f"✓ ffprobe: {ffprobe_check}")
    else:
        print(f"\n✗ ffmpeg not found on PATH after setup!")
        sys.exit(1)

    # Print instructions for permanent PATH
    print("\n" + "=" * 60)
    print("IMPORTANT: To make FFmpeg available in ALL PowerShell sessions,")
    print("run this command ONCE (as administrator is NOT required):")
    print("=" * 60)
    print(f'\n$ffmpegBin = "{bin_str}"')
    print('$currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")')
    print('if ($currentPath -notlike "*$ffmpegBin*") {')
    print('    [Environment]::SetEnvironmentVariable("PATH", "$ffmpegBin;$currentPath", "User")')
    print('    Write-Host "FFmpeg added to user PATH. Restart PowerShell to take effect."')
    print("}")
    print(f"\nOR simply run in every new session:")
    print(f'  $env:PATH = "{bin_str};$env:PATH"')


if __name__ == "__main__":
    main()
