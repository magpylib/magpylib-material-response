"""The nightly report: time to accuracy per case and method, and what got worse
since the night before.

    python -m validation.report records.jsonl                        # the table
    python -m validation.report records.jsonl --previous last.jsonl  # and what got worse

Both print Markdown. With ``--previous`` the exit status is 1 when an error grew
or a problem got larger at a setting both nights ran. Times are reported, never
failed on: a shared runner's noise is larger than most changes.

An error is that of the worst sensor group the reference resolves: a group
below its reference's own error says nothing. The time to an accuracy is that
of the first setting, coarse to fine, that reaches it, its prepare, solve and
getB together.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .cases import CASES
from .methods import METHODS

TARGETS = (0.01, 0.001)
#: relative growth of an error that still counts as the same: an iterative
#: solve's last digits change with the BLAS and the number of threads
ERROR_SLACK = 1e-3
#: a phase this many times slower than the night before is listed
SLOWER = 1.5


def measured(record):
    """The worst group error the reference resolves, and whether every group
    sat below its reference's own error (then the largest of those)."""
    limits = record["reference_errors"]
    seen = {g: e for g, e in record["errors"].items() if e >= limits[g]}
    if seen:
        return max(seen.values()), False
    return max(limits.values()), True


def read(path):
    """The records of a JSON Lines file, one per solve."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def _key(record):
    return record["case"], record["method"], record["setting"]


def _seconds(record):
    return sum(record["seconds"].values())


def _time(seconds):
    if seconds is None:
        return "–"
    return f"{seconds * 1e3:.0f} ms" if seconds < 1 else f"{seconds:.1f} s"


def _percent(value):
    return f"{100 * value:.2g} %"


def reach(records, case, method, target):
    """Seconds to `target` for `method` on `case`, or None."""
    for record in sorted(
        (r for r in records if r["case"] == case and r["method"] == method),
        key=lambda r: r["setting"],
    ):
        if measured(record)[0] <= target:
            return _seconds(record)
    return None


def _fastest(times):
    """The fastest method and how many times faster than the next."""
    reached = sorted((t, m) for m, t in times.items() if t is not None)
    if not reached:
        return "–"
    if len(reached) == 1:
        return f"{reached[0][1]} only"
    (first, method), (second, _) = reached[:2]
    return f"{method}, {second / first:.2g}×"


def table(records):
    """Markdown: per case, each method's time to each target, the fastest,
    and the best error each method reached (≤ when below the reference's own
    error)."""
    present = {r["method"] for r in records}
    methods = [m for m in METHODS if m in present]
    head = ["case"]
    for target in TARGETS:
        head += [f"{m} to {_percent(target)}" for m in methods] + ["fastest"]
    head += [f"best {m}" for m in methods]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name in CASES:
        rows = [r for r in records if r["case"] == name]
        if not rows:
            continue
        cells = [CASES[name].title]
        for target in TARGETS:
            times = {m: reach(records, name, m, target) for m in methods}
            cells += [_time(t) for t in times.values()] + [_fastest(times)]
        for method in methods:
            errors = [measured(r) for r in rows if r["method"] == method]
            if not errors:
                cells.append("–")
                continue
            error, below = min(errors)
            best = ("≤ " if below else "") + _percent(error)
            failure = CASES[name].known_failures.get(method)
            cells.append(f"{best} (known failure)" if failure else best)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def compare(records, previous):
    """What got worse at the settings both nights ran: errors, then problem
    sizes, then times. Returns the Markdown lines that fail and those that
    only inform."""
    before = {_key(r): r for r in previous}
    worse, slower = [], []
    for record in records:
        old = before.get(_key(record))
        if old is None:
            continue
        where = "{} {} at {}".format(*_key(record))
        for group, error in record["errors"].items():
            was = old["errors"].get(group)
            if was is not None and error > was * (1 + ERROR_SLACK) + 1e-15:
                worse.append(
                    f"- {where}, {group}: error {_percent(was)} → {_percent(error)}"
                )
        size, was = record["counts"]["unknowns"], old["counts"]["unknowns"]
        if size > was:
            worse.append(f"- {where}: {was:,} → {size:,} unknowns")
        for phase, seconds in record["seconds"].items():
            then = old["seconds"].get(phase, 0)
            if then > 0.05 and seconds > SLOWER * then:
                slower.append(f"- {where}, {phase}: {_time(then)} → {_time(seconds)}")
    return worse, slower


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", type=Path)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args(argv)

    records = read(args.records)
    print("## Time to accuracy\n")
    print(table(records))
    if args.previous is None:
        return 0
    previous = read(args.previous)
    worse, slower = compare(records, previous)
    print("\n## Since the night before\n")
    print("\n".join(worse) if worse else "No error grew and no problem got larger.")
    if records and previous:
        now, then = records[0]["versions"], previous[0]["versions"]
        changed = [f"{k} {then.get(k)} → {v}" for k, v in now.items() if then.get(k) != v]
        if changed:
            print("\nVersions changed: " + ", ".join(changed) + ".")
    if slower:
        print(f"\nMore than {SLOWER}× slower, for information:\n")
        print("\n".join(slower))
    return 1 if worse else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
