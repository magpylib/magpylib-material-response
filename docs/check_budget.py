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

#: seconds a page may take to execute, with room for the runners' spread
PAGE = 45
#: pages over PAGE when the budget came in, each held under its own ceiling
#: until it is trimmed; lower a ceiling with each trim, never raise it
OVER = {
    "examples/hall_current_sensor": 130,
    "guide/reading": 100,
    "inside/cells": 85,
    "examples/gear_tooth_sensor": 60,
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
    stale = [page for page in OVER if page in pages and pages[page] <= PAGE]
    for page in stale:
        print(f"{page} is back under {PAGE} s: remove it from OVER")  # noqa: T201
    sys.exit("\n".join(over) or None)


if __name__ == "__main__":
    main(*sys.argv[1:])
