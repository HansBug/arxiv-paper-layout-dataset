#!/usr/bin/env python3
"""Download arxiv LaTeX sources for testing algorithm / listing handling.

Picks a static list of arXiv IDs that are known to contain either
``\\begin{algorithm}`` (algorithm or algorithm2e) or ``\\begin{lstlisting}``
or ``\\begin{listing}`` and extracts them into the format expected by the
pipeline:

    <dest-root>/<paper_id>_<slug>/
      src/...

The caller can then run ``scripts/build_dataset.py --source-root <dest>``.
"""

from __future__ import annotations

import argparse
import gzip
import io
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path


# Curated list. Each entry: (arxiv_id, slug, expected_kinds).
# Kinds is a hint for what the paper should exercise; not enforced.
CANDIDATE_PAPERS = [
    # confirmed to carry algorithm / lstlisting
    ("2107.03374", "codex",                  {"listing", "algorithm"}),
    ("2310.10631", "llava_15",               {"lstlisting"}),
    # classical RL / DL papers with algorithm2e or algorithm package
    ("1509.02971", "ddpg",                   {"algorithm"}),
    ("1707.06347", "ppo",                    {"algorithm"}),
    ("1602.01783", "a3c",                    {"algorithm"}),
    ("1805.11604", "batchnorm_analysis",     {"algorithm"}),
    ("1703.03400", "maml",                   {"algorithm"}),
    ("1611.05431", "resnext",                {"algorithm"}),
    ("2006.04768", "linformer",              {"algorithm"}),
    ("2004.10964", "don_t_stop_pretraining", {"algorithm"}),
    ("2006.11239", "ddpm",                   {"algorithm"}),  # minipage-wraps-algorithm pattern
    ("1706.10295", "prioritized_replay",     {"algorithm"}),
    ("1911.12889", "deepmind_data_augmentation", {"algorithm"}),
    ("1812.06162", "empirical_ntk",          {"algorithm"}),
    ("2006.09011", "score_sde",              {"algorithm"}),
    ("1802.09477", "td3",                    {"algorithm"}),
    # additional edge-case coverage (added later)
    ("2103.00020", "clip",                   {"algorithm"}),           # multiple algorithm2e floats
    ("2305.10601", "tot",                    {"algorithm"}),           # tree-of-thoughts
    ("1703.10593", "cyclegan",               {"algorithm"}),
    ("2007.02500", "swav",                   {"algorithm"}),
    ("2006.07733", "byol",                   {"algorithm"}),
    ("2010.02193", "perceiver",              {"algorithm"}),
    ("2302.13971", "llama",                  {"algorithm"}),
    ("1904.08779", "specaugment",            {"algorithm"}),
]


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "arxiv-layout-dataset/1.0"})
    last_error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                payload = response.read()
            if not payload:
                raise RuntimeError(f"Empty response from {url}")
            dest.write_bytes(payload)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def extract(blob: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    data = blob.read_bytes()
    for attempt in (data, None):
        try:
            if attempt is None:
                attempt = gzip.decompress(data)
            with tarfile.open(fileobj=io.BytesIO(attempt), mode="r:*") as tar:
                for member in tar.getmembers():
                    if member.issym() or member.islnk():
                        raise RuntimeError(f"symlink in tar: {member.name}")
                    path = (dest / member.name).resolve()
                    if not str(path).startswith(str(dest.resolve())):
                        raise RuntimeError(f"suspicious path: {member.name}")
                tar.extractall(dest)
            return
        except tarfile.TarError:
            continue
        except OSError:
            continue
    # Single-file .tex source fallback
    text = data.decode("utf-8", errors="ignore")
    if "\\documentclass" in text:
        (dest / "main.tex").write_text(text, encoding="utf-8")
        return
    raise RuntimeError("Cannot unpack arXiv source")


def paper_has_target_env(src: Path) -> set[str]:
    found: set[str] = set()
    for tex in src.rglob("*.tex"):
        try:
            content = tex.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if re.search(r"\\begin\{algorithm\b", content):
            found.add("algorithm")
        if re.search(r"\\begin\{listing\b", content):
            found.add("listing")
        if re.search(r"\\begin\{lstlisting\b", content):
            found.add("lstlisting")
        if re.search(r"\\usepackage\{algorithm2e\}|\\usepackage\[[^]]*\]\{algorithm2e\}", content):
            found.add("algorithm2e")
    return found


def fetch_paper(arxiv_id: str, slug: str, dest_root: Path) -> Path | None:
    work = dest_root / f"{arxiv_id}_{slug}"
    src = work / "src"
    downloads = work / "downloads"
    if src.exists():
        shutil.rmtree(src)
    if downloads.exists():
        shutil.rmtree(downloads)
    src.mkdir(parents=True, exist_ok=True)
    downloads.mkdir(parents=True, exist_ok=True)

    blob = downloads / "source.blob"
    try:
        download(f"https://arxiv.org/e-print/{arxiv_id}", blob)
        extract(blob, src)
    except Exception as exc:  # noqa: BLE001
        print(f"[fail] {arxiv_id} {slug}: {exc}", flush=True)
        return None

    found = paper_has_target_env(src)
    print(f"[ok] {arxiv_id} {slug}: {sorted(found)}", flush=True)
    return work


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "runs" / "test_papers_src",
    )
    parser.add_argument("--only", nargs="*", help="restrict to these arxiv ids")
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    for arxiv_id, slug, _ in CANDIDATE_PAPERS:
        if args.only and arxiv_id not in args.only:
            continue
        fetch_paper(arxiv_id, slug, args.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
