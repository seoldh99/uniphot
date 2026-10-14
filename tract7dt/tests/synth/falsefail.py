"""False-failure device: how often would a correct (ideal) implementation fail a test?

Design 6 sample-size rule: a tolerance is set so that the ideal implementation fails it less than
1% of the time, and "the test code measures the false-failure rate once with the ideal implementation and records it".
``measure`` runs a check (``check(rng) -> CheckResult``) ``n_repeats`` times with independent
seeds (``numpy.random.SeedSequence(seed0).spawn``), counts failures, and also estimates the rate
from the spread of each statistic (normal tail outside its limits, summed over statistics),
which resolves rates far below 1 / n_repeats.  ``save`` writes one JSON file per test id under
``false_failure/``.  ``accepts`` is the design rule read strictly: the normal tail AND the one-sided
95% Clopper-Pearson upper bound of the counted rate are both below 1%; when the bound is not, the
remedy is more repeats, never a looser limit.  A record stores the hash of the generator source and
of its check's module (docstrings and comment-only lines left out); a test fails on a stale record.

Command line (from ``tract7dt/tests``):
  python -m synth.falsefail [ids ...] [--workers N] [--write]          # checks in synth.checks.REGISTRY
  python -m synth.falsefail --check MODULE:FUNC --id ID --repeats N [--workers N] [--write]
The second form measures a check defined anywhere importable (e.g. a later unit's test module), so a
unit can record its own test's rate in a new file false_failure/ID.json without editing this package.
After editing generator code, measure again (the record test fails until then).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import stats as _stats

RECORD_DIR = Path(__file__).resolve().parent / "false_failure"
DEFAULT_SEED0 = 20261010


@dataclass
class Record:
    """Result of one false-failure measurement.

    rate = n_fail / n_repeats; rate_upper95 = one-sided 95% Clopper-Pearson upper bound;
    normal_tail_rate = sum over statistics of the normal tail outside [lo, hi] given the
    statistic's mean and SD over the repeats (capped at 1).
    stats: name -> {"mean", "sd", "lo", "hi", "tail"}.
    """
    test_id: str
    n_repeats: int
    n_fail: int
    rate: float
    rate_upper95: float
    normal_tail_rate: float
    stats: dict = field(default_factory=dict)
    seed0: int = DEFAULT_SEED0
    generator_sha256: str = ""
    check_sha256: str = ""
    created: str = ""
    seconds: float = 0.0
    note: str = ""


def rngs_for(seed0: int, n: int) -> list:
    """The n independent generators measure() uses, in order."""
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed0).spawn(n)]


def clopper_pearson_upper(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided upper confidence bound of a binomial rate with k failures in n trials."""
    if k >= n:
        return 1.0
    return float(_stats.beta.ppf(conf, k + 1, n - k))


def _code_text(text: str) -> str:
    """Source lines without docstrings, comment-only lines, blank lines and trailing spaces.

    Docstring lines come from the AST (line numbers are stable across Python versions); every other
    line is kept verbatim, so any code change -- including an inline comment -- changes the text.
    """
    drop = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                drop.update(range(body[0].lineno, body[0].end_lineno + 1))
    keep = [line.rstrip() for i, line in enumerate(text.splitlines(), 1)
            if i not in drop and line.strip() and not line.strip().startswith("#")]
    return "\n".join(keep)


def source_sha256(path) -> str:
    """sha256 of one Python source with docstrings and comment-only lines left out."""
    return hashlib.sha256(_code_text(Path(path).read_text()).encode()).hexdigest()


def generator_sha256(directory=None) -> str:
    """sha256 over the synth package's .py sources (sorted by name), docstrings and comments left out."""
    h = hashlib.sha256()
    for p in sorted(Path(directory or Path(__file__).resolve().parent).glob("*.py")):
        h.update(p.name.encode() + b"\0" + _code_text(p.read_text()).encode() + b"\0")
    return h.hexdigest()


def _check_sha256(check: Callable) -> str:
    path = getattr(sys.modules.get(getattr(check, "__module__", ""), None), "__file__", None)
    return source_sha256(path) if path else ""


def accepts(record: Record, limit: float = 0.01) -> tuple:
    """Design 6 rule: the false-failure rate of the ideal implementation is below ``limit``.

    Both estimates must be below it: the normal tail and the one-sided 95% Clopper-Pearson upper
    bound of the counted rate.  Returns (ok, list of reasons it is not).
    """
    why = []
    if not record.normal_tail_rate < limit:
        why.append(f"normal tail {record.normal_tail_rate:.3%} is not below {limit:.0%}")
    if not record.rate_upper95 < limit:
        why.append(f"95% upper bound {record.rate_upper95:.3%} of the counted rate "
                   f"({record.n_fail} of {record.n_repeats}) is not below {limit:.0%}")
    return not why, why


def _run_one(check: Callable, seq) -> tuple:
    res = check(np.random.default_rng(seq))
    return bool(res.passed), {k: (float(v), float(lo), float(hi)) for k, (v, lo, hi) in res.stats.items()}


def _summarise(results: list) -> tuple:
    out, total = {}, 0.0
    for name in results[0][1]:
        vals = np.array([r[1][name][0] for r in results])
        lo, hi = results[0][1][name][1], results[0][1][name][2]
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        if np.all(np.isfinite(vals)) and sd > 0.0:
            tail = float(_stats.norm.cdf((lo - mean) / sd) + _stats.norm.sf((hi - mean) / sd))
        else:
            tail = 0.0 if np.all(np.isfinite(vals)) and lo <= mean <= hi else 1.0
        out[name] = {"mean": mean, "sd": sd, "lo": lo, "hi": hi, "tail": tail}
        total += tail
    return out, min(1.0, total)


def measure(test_id: str, check: Callable, n_repeats: int, seed0: int = DEFAULT_SEED0,
            workers: int = 1, note: str = "") -> Record:
    """Run ``check`` n_repeats times and summarise how often it fails."""
    t0 = time.time()
    seqs = np.random.SeedSequence(seed0).spawn(n_repeats)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_run_one, [check] * n_repeats, seqs,
                                  chunksize=max(1, n_repeats // (4 * workers))))
    else:
        results = [_run_one(check, s) for s in seqs]
    n_fail = sum(not ok for ok, _ in results)
    summary, tail = _summarise(results)
    return Record(test_id=test_id, n_repeats=n_repeats, n_fail=n_fail, rate=n_fail / n_repeats,
                  rate_upper95=clopper_pearson_upper(n_fail, n_repeats), normal_tail_rate=tail,
                  stats=summary, seed0=seed0, generator_sha256=generator_sha256(),
                  check_sha256=_check_sha256(check),
                  created=datetime.now().isoformat(timespec="seconds"),
                  seconds=round(time.time() - t0, 3), note=note)


def save(record: Record, directory: Path = RECORD_DIR) -> Path:
    """Write record as <directory>/<test_id>.json and return the path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record.test_id}.json"
    path.write_text(json.dumps(asdict(record), indent=1, sort_keys=True) + "\n")
    return path


def load(test_id: str, directory: Path = RECORD_DIR) -> Record:
    """Read the record of one test id."""
    return Record(**json.loads((Path(directory) / f"{test_id}.json").read_text()))


def main(argv=None) -> int:
    from .checks import REGISTRY

    ap = argparse.ArgumentParser(description="Measure false-failure rates of registered checks.")
    ap.add_argument("ids", nargs="*", help="test ids (default: all registered)")
    ap.add_argument("--repeats", type=int, default=None, help="override the registered repeat count")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--write", action="store_true", help="save the records under false_failure/")
    ap.add_argument("--note", default="")
    ap.add_argument("--check", default=None, help="MODULE:FUNC of a check outside the registry")
    ap.add_argument("--id", default=None, help="test id of the --check record")
    a = ap.parse_args(argv)
    if a.check:
        if not (a.id and a.repeats) or a.ids:
            ap.error("--check needs --id and --repeats and no positional ids")
        mod, _, fn = a.check.partition(":")
        jobs = [(a.id, (getattr(importlib.import_module(mod), fn), a.repeats))]
    else:
        jobs = [(tid, REGISTRY[tid]) for tid in (a.ids or sorted(REGISTRY))]
    for tid, (check, reps) in jobs:
        rec = measure(tid, check, a.repeats or reps, seed0=a.seed, workers=a.workers, note=a.note)
        print(f"{tid}: repeats {rec.n_repeats}, failed {rec.n_fail} ({rec.rate:.4%}, 95% upper "
              f"{rec.rate_upper95:.4%}), normal tail {rec.normal_tail_rate:.3e}, {rec.seconds:.1f} s",
              flush=True)
        if a.write:
            print(f"  -> {save(rec)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
