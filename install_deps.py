"""
Offline-friendly installer for environments where DNS fails for
files.pythonhosted.org (common on some Windows networks).

Uses the PyPI JSON API (pypi.org) + curl --resolve against a known
Fastly CDN IP, then installs wheels with pip --no-index.
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WHEELS = ROOT / "wheels"
REQUIREMENTS = ROOT / "requirements.txt"

# Fastly CDN IPs used by files.pythonhosted.org / pypi.org
CDN_IPS = ["151.101.0.223", "151.101.64.223", "151.101.128.223", "151.101.192.223"]

# Core packages from requirements.txt (+ common hard deps)
PACKAGES = [
    "pandas",
    "numpy",
    "scikit-learn",
    "xgboost",
    "matplotlib",
    "seaborn",
    "streamlit",
    "joblib",
    "jupyter",
    # hard dependencies often needed offline
    "scipy",
    "pillow",
    "contourpy",
    "cycler",
    "fonttools",
    "kiwisolver",
    "pyparsing",
    "altair",
    "blinker",
    "cachetools",
    "click",
    "gitpython",
    "protobuf",
    "pyarrow",
    "pydeck",
    "tenacity",
    "toml",
    "tornado",
    "watchdog",
    "narwhals",
    "gitdb",
    "smmap",
    "itsdangerous",
    "jinja2",
    "markupsafe",
    "rich",
    "markdown-it-py",
    "mdurl",
    "pygments",
    "referencing",
    "rpds-py",
    "jsonschema",
    "jsonschema-specifications",
    "attrs",
    "threadpoolctl",
]


def pick_wheel(pkg: str) -> tuple[str, str] | None:
    """Return (url, filename) for the best Windows wheel, if any."""
    with urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/json", timeout=60) as resp:
        data = json.load(resp)

    cands: list[tuple[int, str, str]] = []
    for u in data["urls"]:
        if u["packagetype"] != "bdist_wheel":
            continue
        fn = u["filename"]
        score = 0
        if "win_amd64" in fn and "cp314" in fn and "cp314t" not in fn:
            score = 5
        elif "win_amd64" in fn and "abi3" in fn:
            score = 4
        elif "win_amd64" in fn and "py3-none" in fn:
            score = 3
        elif "py3-none-any" in fn or "py2.py3-none-any" in fn:
            score = 2
        if score:
            cands.append((score, u["url"], fn))

    if not cands:
        return None
    cands.sort(reverse=True)
    _, url, fn = cands[0]
    return url, fn


def download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  exists: {dest.name}")
        return True
    for ip in CDN_IPS:
        cmd = [
            "curl.exe",
            "-fsSL",
            "--resolve",
            f"files.pythonhosted.org:443:{ip}",
            "-o",
            str(dest),
            url,
        ]
        if subprocess.run(cmd).returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            print(f"  downloaded: {dest.name}")
            return True
        if dest.exists():
            dest.unlink(missing_ok=True)
    print(f"  FAILED: {dest.name}")
    return False


def main() -> int:
    WHEELS.mkdir(exist_ok=True)
    print("Downloading wheels (DNS workaround for files.pythonhosted.org)...")
    ok = 0
    for pkg in PACKAGES:
        print(f"* {pkg}")
        picked = pick_wheel(pkg)
        if not picked:
            print("  no suitable wheel found")
            continue
        url, fn = picked
        if download(url, WHEELS / fn):
            ok += 1

    print(f"\nDownloaded/ready: {ok} wheels")
    print("Installing into current Python environment...")
    # Install core packages individually so one optional failure (e.g. jupyter)
    # does not block the rest.
    core = [
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "matplotlib",
        "seaborn",
        "streamlit",
        "joblib",
    ]
    failed: list[str] = []
    for pkg in core + ["jupyter"]:
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                f"--find-links={WHEELS}",
                pkg,
                "--disable-pip-version-check",
            ]
        )
        if r.returncode != 0:
            failed.append(pkg)
            print(f"WARN: could not fully install {pkg}")

    if failed:
        print(
            f"\nInstalled with warnings for: {', '.join(failed)}\n"
            "Core ML app packages may still be ready.\n"
            "Retry:  python install_deps.py\n"
            "Or fix DNS permanently (Admin PowerShell):\n"
            '  Add-Content $env:SystemRoot\\System32\\drivers\\etc\\hosts '
            '"`n151.101.0.223 files.pythonhosted.org"'
        )
        # Non-zero only if a core package failed
        return 1 if any(p in failed for p in core) else 0

    print("\nAll requested packages installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
