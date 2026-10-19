#!/bin/bash
S=/private/tmp/claude-501/-Users-alexandreboisselet-Documents-GitRepos-nosync-magpylib/e2f29b63-0bf2-4985-9167-bfac6cef8364/scratchpad
PY=/Users/alexandreboisselet/Documents/GitRepos.nosync/magpylib-material-response/.venv/bin/python
export PYTHONPATH=$S/wt52/src:$S/adapt
cd $S/review/solver
for case in "cube 50" "ucore 50"; do
  echo "== $case, COLLOC=corner4"
  for f in 0.2 0.3 0.4 0.4472 0.5 0.55 0.6 0.7 0.8 0.9; do
    echo -n "f=$f  "; COLLOC=corner4 FRAC=$f $S/guard.sh 2 $PY $S/adapt/p1tet.py $case p1 2>&1 | grep -v "peak RSS" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(f\"cells {d['cells']}  err {d['err_pct']:+.3f} %  {d['s']} s\")"
  done
  echo "== $case, COLLOC=child8"; COLLOC=child8 $S/guard.sh 2 $PY $S/adapt/p1tet.py $case p1 2>&1 | grep -v "peak RSS"
  echo "== $case, COLLOC=corner4c f=0.5 (5 pts LSQ)"; COLLOC=corner4c FRAC=0.5 $S/guard.sh 2 $PY $S/adapt/p1tet.py $case p1 2>&1 | grep -v "peak RSS"
  echo "== $case, p0"; $S/guard.sh 2 $PY $S/adapt/p1tet.py $case p0 2>&1 | grep -v "peak RSS"
done
