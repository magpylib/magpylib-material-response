"""Check a docs build log against the time budget, so that the build stays well
inside Read the Docs' 15 minutes as the docs grow.

    sphinx-build ... 2>&1 | tee build.log
    python docs/check_budget.py build.log

Times are those of the CI docs job, a 4-CPU GitHub runner; Read the Docs, with
2 CPUs, takes up to 1.7 times as long. A page that needs longer moves its slow
solves to a study script and stores their results, as the examples do.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

#: seconds a page may take to execute. Two pages run side by side, and a
#: page's time depends on its neighbour: one page took 69 s on one run and
#: under 45 s on the next. The limits are 1.5 times the slowest seen, so they
#: catch a page that grows, not that spread; the total below is steadier.
PAGE = 60
#: pages over PAGE when the budget came in, each held under its own ceiling
#: until it is trimmed; lower a ceiling with each trim, never raise it
OVER = {
    "examples/hall_current_sensor": 200,
    "guide/reading": 130,
    "inside/cells": 105,
}
#: seconds the whole build may take, from its start to its last page written
TOTAL = 480


def main(log):
    text = Path(log).read_text(encoding="utf-8", errors="replace")
    pages = {
        match["page"]: float(match["seconds"])
        for match in re.finditer(
            r"docs/(?P<page>\S+)\.md: Executed notebook in (?P<seconds>[\d.]+) seconds",
            text,
        )
    }
    done = re.search(r"docs build: done after (\d+) s", text)
    if not pages or not done:
        sys.exit(f"{log}: no executed pages or no end of the build in this log")
    over = []
    for page, seconds in sorted(pages.items(), key=lambda item: -item[1]):
        ceiling = OVER.get(page, PAGE)
        flag = "  OVER" if seconds > ceiling else ""
        print(f"{seconds:7.1f} s  of {ceiling:4d} s  {page}{flag}")  # noqa: T201
        if flag:
            over.append(f"{page} executed in {seconds:.0f} s, over its {ceiling} s")
    total = int(done[1])
    print(f"the build: {total} s of {TOTAL} s")  # noqa: T201
    if total > TOTAL:
        over.append(f"the build took {total} s, over its {TOTAL} s")
    sys.exit("\n".join(over) or None)


if __name__ == "__main__":
    main(*sys.argv[1:])
