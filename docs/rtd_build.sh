#!/usr/bin/env bash
# The docs build on Read the Docs, run in the background and waited for in
# steps of at most 4 minutes. From 2026-11-08 a build run as one command often
# stopped at the 15-minute limit with no log, a commit that stalled once passing
# on another try; the same build run this way passed every time (#207). The
# cause on Read the Docs' side is not known.
#   rtd_build.sh start           start the build in the background
#   rtd_build.sh wait SECONDS    wait up to SECONDS; if done, show the log's end
#                                and exit with the build's status
#   rtd_build.sh last SECONDS    the same, and fail if still not done
set -u
case "$1" in
start)
  rm -f build.log build.status
  # shellcheck disable=SC2016  # expanded by the inner bash, on purpose
  setsid nohup bash -c 'uv run --no-sync --no-dev sphinx-build -j 2 -T -b html -d _build/doctrees -D language=en docs "$READTHEDOCS_OUTPUT/html" > build.log 2>&1; echo $? > build.status' > /dev/null 2>&1 &
  echo "started at $(date +%T)"
  ;;
wait | last)
  for _ in $(seq "$2"); do
    if [ -f build.status ]; then
      grep -E "WARNING|ERROR|docs build:" build.log | tail -n 100
      tail -n 20 build.log
      exit "$(cat build.status)"
    fi
    sleep 1
  done
  echo "still building at $(date +%T): $(tail -n 1 build.log | cut -c1-150)"
  [ "$1" = wait ]
  ;;
esac
