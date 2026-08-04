"""Shared sweep driver: crash-safe row writing and resume across both backends.

Why this exists
---------------

The steady and unsteady drivers both had the same two properties, and at panel
speed neither mattered. At URANS speed both are fatal.

**Rows were written once, after the whole sweep finished.** A twelve-case URANS
sweep is a day of compute; a crash on the last case, a full disk, or a Ctrl-C
threw away every solved case with it, because nothing had touched the chunk file
yet. Rows are now appended and flushed as each case completes, so whatever
finished is on disk.

**Every job was re-solved on a re-run.** With no record of what was already
done, restarting a partially completed sweep re-ran the cases that had
succeeded. The sweep now reads the rows directory first and skips any
(foil, alpha) that already carries an ``ok`` row at the same or higher fidelity.
Steady will not clobber an unsteady row; unsteady *will* re-solve a case that
only exists at steady or panel fidelity, which is the intent -- that is an
upgrade.

Resume is on by default and ``resume=False`` forces the full sweep.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

# Ranked low to high; a sweep skips a case whose existing row is at least as
# good as what the sweep would produce.
FIDELITY_RANK = {"panel": 0, "rans": 1, "urans": 2}


def foil_name(dat_path: str | Path) -> str:
    """The foil key a row will carry. Must match ``run_one`` in both drivers."""
    return Path(dat_path).stem.replace(".dat", "")


def _key(foil: str, alpha: float) -> tuple:
    # Angles are written as floats and read back as floats; round so that 12.0
    # and 12.000000001 are the same case.
    return (str(foil), round(float(alpha), 6))


def solved_keys(rows_dir: str | Path, min_rank: int = 0) -> set:
    """(foil, alpha) pairs that already have an ``ok`` row of rank >= min_rank."""
    rows_dir = Path(rows_dir)
    found: set = set()
    if not rows_dir.exists():
        return found
    for chunk in sorted(rows_dir.glob("*.jsonl")):
        for line in chunk.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                # A row half-written by a killed process. Ignore it; the case
                # will simply be re-solved.
                continue
            if not r.get("ok"):
                continue
            if FIDELITY_RANK.get(r.get("fidelity", "panel"), 0) < min_rank:
                continue
            if r.get("foil") is None or r.get("alpha_deg") is None:
                continue
            found.add(_key(r["foil"], r["alpha_deg"]))
    return found


def run_sweep(
    jobs: Sequence[tuple],
    worker: Callable[..., dict],
    rows_dir: str | Path,
    prefix: str,
    fidelity: str,
    max_workers: int = 1,
    resume: bool = True,
    progress: bool = True,
    log: Optional[Callable[[dict, int, int, float], None]] = None,
) -> tuple:
    """Run ``worker(path, alpha)`` over ``jobs``, streaming rows to a chunk file.

    Returns ``(rows, chunk_path, skipped)``. ``rows`` are only the cases this
    call solved; the skipped ones are already on disk in an earlier chunk and
    :func:`sensor.generate.finalize_manifest` merges all chunks anyway.
    """
    rows_dir = Path(rows_dir)
    rows_dir.mkdir(parents=True, exist_ok=True)

    todo = list(jobs)
    skipped: list = []
    if resume:
        done = solved_keys(rows_dir, min_rank=FIDELITY_RANK.get(fidelity, 0))
        keep = []
        for p, a in todo:
            (skipped if _key(foil_name(p), a) in done else keep).append((p, a))
        todo = keep

    t0 = time.time()
    chunk = rows_dir / f"{prefix}_{int(t0)}.jsonl"
    rows: list = []
    if not todo:
        return rows, chunk, [(str(p), a) for p, a in skipped]

    with chunk.open("w") as fh:

        def _emit(row: dict) -> None:
            rows.append(row)
            fh.write(json.dumps(
                {k: v for k, v in row.items() if k != "traceback"}) + "\n")
            # A sweep that runs for a day must survive being killed at any
            # point in it, so the row is on the platter before the next case
            # starts rather than in a buffer that dies with the process.
            fh.flush()
            os.fsync(fh.fileno())
            if progress and log is not None:
                log(row, len(rows), len(todo), t0)

        if max_workers == 1:
            for p, a in todo:
                _emit(worker(p, a))
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futs = {pool.submit(worker, p, a): (p, a) for p, a in todo}
                for fut in as_completed(futs):
                    _emit(fut.result())

    rows.sort(key=lambda r: (r.get("foil") or "", r.get("alpha_deg") or 0.0))
    return rows, chunk, [(str(p), a) for p, a in skipped]
