"""Time the PR check against the base branch: the same cases at the same
settings, solved by this environment's package and by the base's.

    python -m validation.timing record out/head-1.json   # one round
    python -m validation.timing compare out              # base-*.json against head-*.json

The rounds alternate between the two environments, and each case's time is the
least over its rounds, the one a shared runner disturbed least. ``compare``
prints Markdown for a PR comment and never fails: a time only blocks once the
spread between rounds is known.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import magpylib_material_response

from .cases import CASES
from .methods import METHODS
from .run import PR_CHECK, run, versions

#: the comment's first line, by which .github/workflows/timing.yml finds it again
MARKER = "<!-- validation timing -->"
#: slower than this, a case is flagged
SLOWER = 1.2
#: cases faster than this are not flagged: their time is mostly noise
FLOOR = 0.05


def record(path):
    """Solve the PR check once and write each case's phase times."""
    times = {}
    for name, by_method in json.loads(PR_CHECK.read_text()).items():
        for method_name, entry in by_method.items():
            method = METHODS[method_name]
            if not method.can_solve(CASES[name]):
                continue
            key = f"{name}, {method_name}"
            try:
                times[key] = run(CASES[name], method, entry["setting"])["seconds"]
            except Exception as error:  # noqa: BLE001  # the base may lack what a case needs
                times[key] = {"failed": f"{type(error).__name__}: {error}"[:200]}
    data = {
        "version": magpylib_material_response.__version__,
        "versions": versions(),
        "times": times,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=1) + "\n")


def _rounds(folder, side):
    return [json.loads(p.read_text()) for p in sorted(folder.glob(f"{side}-*.json"))]


def _least(rounds, key):
    """The least total time of `key` over the rounds, and the spread between
    rounds as a fraction of it; None if a round failed."""
    totals = []
    for data in rounds:
        seconds = data["times"].get(key)
        if seconds is None or "failed" in seconds:
            return None, None
        totals.append(sum(seconds.values()))
    least = min(totals)
    return least, (max(totals) - least) / least if least > 0 else 0.0


def _environment(data):
    """The package's version and the magpylib it ran on, for the comment's end."""
    version, threads = (
        data["versions"]["magpylib"],
        data["versions"]["magpylib_threads"],
    )
    return (
        f"{data['version']} with magpylib {version} "
        f"on {threads} thread{'s' if threads != 1 else ''}"
    )


def _time(seconds):
    return f"{seconds * 1e3:.0f} ms" if seconds < 1 else f"{seconds:.2f} s"


def compare(folder):
    """Markdown: each case's time on the base and here, slowest change first."""
    base, head = _rounds(folder, "base"), _rounds(folder, "head")
    title = f"{MARKER}\n### Timing against the base branch\n\n"
    if not base:
        return f"{title}The base could not run."
    rows, flagged, total = [], 0, [0.0, 0.0]
    for key in head[0]["times"]:
        then, then_spread = _least(base, key)
        now, now_spread = _least(head, key)
        if now is None:
            rows.append((float("inf"), f"| {key} | | failed here | | |"))
            continue
        if then is None:
            rows.append((0.0, f"| {key} | not on the base | {_time(now)} | | |"))
            continue
        total[0] += then
        total[1] += now
        ratio = now / then
        flag = ratio > SLOWER and then > FLOOR
        flagged += flag
        spread = max(then_spread, now_spread)
        bold = "**" if flag else ""
        cells = [key, _time(then), _time(now), f"{bold}{ratio:.2f}x{bold}"]
        rows.append((ratio, f"| {' | '.join(cells)} | {100 * spread:.0f} % |"))
    rows.sort(key=lambda row: -row[0])
    summary = (
        f"PR check solved by this PR in {_time(total[1])} against "
        f"{_time(total[0])} on the base ({total[1] / total[0]:.2f}x), "
        f"least of {len(head)} rounds each."
        if total[0] > 0
        else "No case ran on both."
    )
    if flagged:
        summary += f" {flagged} case(s) more than {SLOWER}x slower, in bold."
    table = "\n".join(
        [
            "| case | base | this PR | ratio | spread between rounds |",
            "|---|---|---|---|---|",
            *(row for _, row in rows),
        ]
    )
    environments = (
        f"Base {_environment(base[0])}; this PR {_environment(head[0])}; "
        f"{head[0]['versions']['cpus']} CPUs."
    )
    return (
        f"{title}{summary}\n\n{table}\n\n{environments} "
        "For information only: nothing fails on time yet."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("record").add_argument("path", type=Path)
    sub.add_parser("compare").add_argument("folder", type=Path)
    args = parser.parse_args(argv)
    if args.command == "record":
        record(args.path)
    else:
        print(compare(args.folder))


if __name__ == "__main__":
    main(sys.argv[1:])
