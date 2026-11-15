#!/usr/bin/env bash
# DIAGNOSTIC, not to merge: see inside a Read the Docs build that stalls.
# Read the Docs keeps a command's log only when the command ends, so the build
# runs in the background and separate commands report on it.
#   rtd_probe.sh start    start the docs build in the background
#   rtd_probe.sh report   the log's tail, the busiest processes, memory, load
#   rtd_probe.sh wait     wait for the build, show its end, exit with its status
set -u
case "$1" in
start)
  rm -f build.log build.status
  # shellcheck disable=SC2016  # expanded by the inner bash, on purpose
  setsid nohup bash -c 'PYTHONUNBUFFERED=1 uv run --no-sync --no-dev sphinx-build -j 2 -T -b html -d _build/doctrees -D language=en docs "$READTHEDOCS_OUTPUT/html" > build.log 2>&1; echo $? > build.status' > /dev/null 2>&1 &
  echo "started at $(date +%T)"
  ;;
report)
  sleep "${2:-240}"
  echo "== $(date +%T), load $(cut -d' ' -f1-3 /proc/loadavg)"
  free -m
  echo "== the log's last lines"
  tail -n 25 build.log
  echo "== the busiest processes"
  ps -eo pid,ppid,etime,pcpu,rss,stat,args --sort=-pcpu | head -15 | cut -c1-200
  echo "== where the kernels are (if ptrace is allowed)"
  for pid in $(pgrep -f ipykernel_launcher); do
    timeout 60 uvx py-spy dump --pid "$pid" 2>&1 | head -40
  done
  ;;
wait)
  while [ ! -f build.status ]; do sleep 5; done
  tail -n 40 build.log
  exit "$(cat build.status)"
  ;;
esac
