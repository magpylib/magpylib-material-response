#!/bin/bash
# usage: guard.sh <limit_GB> cmd...   kills cmd when its RSS passes the limit
limit_kb=$(( $1 * 1024 * 1024 )); shift
"$@" &
pid=$!
peak=0
while kill -0 $pid 2>/dev/null; do
  rss=$(ps -o rss= -p $pid 2>/dev/null | tr -d ' ')
  [ -n "$rss" ] && [ "$rss" -gt "$peak" ] && peak=$rss
  if [ -n "$rss" ] && [ "$rss" -gt "$limit_kb" ]; then
    kill -9 $pid; echo "GUARD: killed at $((rss / 1024)) MB"; exit 137
  fi
  sleep 0.2
done
wait $pid; rc=$?
echo "   peak RSS $((peak / 1024)) MB"
exit $rc
