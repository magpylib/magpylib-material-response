"""Time the PR check and measure its peak memory against the base branch: the
same cases at the same settings, solved by this environment's package and by
the base's.

    python -m validation.timing record out/head-1.json       # one round
    python -m validation.timing memory out/memory-head.json  # peak memory, per case
    python -m validation.timing peak plate MoM               # one case's, in bytes
    python -m validation.timing compare out                  # base against head

The rounds alternate between the two environments, and each case's time is the
least over its rounds, the one a shared runner disturbed least. Peak memory
needs a process per case, since a process only knows its highest point so far;
it is measured once. ``compare`` prints Markdown for a PR comment and never
fails: a number only blocks once the spread between rounds is known.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import magpylib_material_response

from .cases import CASES
from .methods import METHODS
from .run import PR_CHECK, run

#: the comment's first line, by which .github/workflows/timing.yml finds it again
MARKER = "<!-- validation timing -->"
#: slower than this, a case is flagged
SLOWER = 1.2
#: cases faster than this are not flagged: their time is mostly noise
FLOOR = 0.05
#: more peak memory than this, in bytes, a case is flagged: the same case moved
#: by up to 66 MB between runs on a laptop and 27 MB on a CI runner
MORE_MEMORY = 100e6
#: a case's memory is given up on after this many seconds; it solves in seconds
PEAK_TIMEOUT = 600


def _pr_check():
    """Each PR-check case this package can solve: its key among the results, its
    name, method and setting."""
    for name, by_method in json.loads(PR_CHECK.read_text()).items():
        for method_name, entry in by_method.items():
            if METHODS[method_name].can_solve(CASES[name]):
                yield f"{name}, {method_name}", name, method_name, entry["setting"]


def record(path):
    """Solve the PR check once and write each case's phase times."""
    times = {}
    for key, name, method_name, setting in _pr_check():
        try:
            times[key] = run(CASES[name], METHODS[method_name], setting)["seconds"]
        except Exception as error:  # noqa: BLE001  # the base may lack what a case needs
            times[key] = {"failed": f"{type(error).__name__}: {error}"[:200]}
    data = {"version": magpylib_material_response.__version__, "times": times}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=1) + "\n")


def peak(name, method_name):
    """Solve one case and print the peak memory of this process, in bytes."""
    import resource  # noqa: PLC0415  # not on Windows; CI measures on Linux

    setting = json.loads(PR_CHECK.read_text())[name][method_name]["setting"]
    run(CASES[name], METHODS[method_name], setting)
    used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(used if sys.platform == "darwin" else used * 1024)


def _peak_elsewhere(name, method_name):
    """A case's peak memory, solved in a process of its own; None if that failed,
    with the reason in the log."""
    command = [sys.executable, "-m", "validation.timing", "peak", name, method_name]
    try:
        done = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parents[1],
            timeout=PEAK_TIMEOUT,
        )
        return int(done.stdout.split()[-1])
    except (subprocess.SubprocessError, ValueError, IndexError) as error:
        print(f"{name}, {method_name}: {error}", file=sys.stderr)
        print(getattr(error, "stderr", "") or "", file=sys.stderr)
        return None


def memory(path):
    """Write each case's peak memory, or None where it failed. Nothing is solved
    in this process: on Linux a child's peak starts from its parent's."""
    peaks = {
        key: _peak_elsewhere(name, method_name)
        for key, name, method_name, _ in _pr_check()
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(peaks, indent=1) + "\n")


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


def _time(seconds):
    return f"{seconds * 1e3:.0f} ms" if seconds < 1 else f"{seconds:.2f} s"


def _dot(worse, better):
    """The mark of a change: GitHub's tables keep no colours of their own."""
    return "🔴" if worse else "🟢" if better else "⚪"


def _bytes(size):
    return f"{size / 1e6:.0f} MB" if size < 1e9 else f"{size / 1e9:.2f} GB"


def _peaks(folder, side):
    path = folder / f"memory-{side}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _memory_cell(then, now):
    """The cell of a case's peak memory, and whether it is flagged. `then` is
    None where the base has no number, `now` where this PR's solve failed."""
    if now is None:
        return "🔴 **failed**", True
    if then is None:
        return _bytes(now), False
    flag = now - then > MORE_MEMORY
    bold = "**" if flag else ""
    dot = _dot(flag, then - now > MORE_MEMORY)
    return f"{dot} {_bytes(then)} → {bold}{_bytes(now)}{bold}", flag


def compare(folder):
    """Markdown: each case's time and peak memory on the base and here, slowest
    change first."""
    base, head = _rounds(folder, "base"), _rounds(folder, "head")
    peaks_then, peaks_now = _peaks(folder, "base"), _peaks(folder, "head")
    title = f"{MARKER}\n### Time and memory against the base branch\n\n"
    if not base:
        return f"{title}The base could not run."
    rows, flagged, more_memory, total = [], 0, [], [0.0, 0.0]
    for key in head[0]["times"]:
        cell, grew = ("", False)
        if key in peaks_now:
            cell, grew = _memory_cell(peaks_then.get(key), peaks_now[key])
        if grew:
            more_memory.append(key)
        then, then_spread = _least(base, key)
        now, now_spread = _least(head, key)
        if now is None:
            rows.append((float("inf"), f"| {key} | | failed here | | | {cell} |"))
            continue
        if then is None:
            rows.append(
                (0.0, f"| {key} | not on the base | {_time(now)} | | | {cell} |")
            )
            continue
        total[0] += then
        total[1] += now
        ratio = now / then
        flag = ratio > SLOWER and then > FLOOR
        dot = _dot(flag, ratio < 1 / SLOWER and then > FLOOR)
        flagged += flag
        spread = max(then_spread, now_spread)
        bold = "**" if flag else ""
        cells = [key, _time(then), _time(now), f"{dot} {bold}{ratio:.2f}x{bold}"]
        rows.append((ratio, f"| {' | '.join(cells)} | {100 * spread:.0f} % | {cell} |"))
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
    if more_memory:
        summary += (
            f" Over {MORE_MEMORY / 1e6:.0f} MB more memory, or failed: "
            f"{'; '.join(more_memory)}."
        )
    table = "\n".join(
        [
            "| case | base | this PR | ratio | spread between rounds | peak memory |",
            "|---|---|---|---|---|---|",
            *(row for _, row in rows),
        ]
    )
    legend = (
        f"🔴 worse, 🟢 better, ⚪ within {SLOWER}x of the time "
        f"(cases under {_time(FLOOR)} are not judged) or {MORE_MEMORY / 1e6:.0f} MB "
        "of the memory."
    )
    versions = f"Base {base[0]['version']}, this PR {head[0]['version']}."
    return (
        f"{title}{summary}\n\n{table}\n\n{legend}\n\n{versions} "
        "For information only: nothing fails on time or memory yet."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("record").add_argument("path", type=Path)
    sub.add_parser("memory").add_argument("path", type=Path)
    one = sub.add_parser("peak")
    one.add_argument("name")
    one.add_argument("method_name")
    sub.add_parser("compare").add_argument("folder", type=Path)
    args = parser.parse_args(argv)
    if args.command == "record":
        record(args.path)
    elif args.command == "memory":
        memory(args.path)
    elif args.command == "peak":
        peak(args.name, args.method_name)
    else:
        print(compare(args.folder))


if __name__ == "__main__":
    main(sys.argv[1:])
