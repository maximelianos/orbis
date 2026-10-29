#!/bin/bash
# Run the commands of a queue file one after another; one log per command.
#   tools/tmux/queue.sh <queue file>
# Each non-empty, non-comment line is "<log file> <command...>". A line whose log
# already ends with "__EXIT 0" is skipped, so a queue can be restarted.
while read -r log cmd; do
  [[ -z "$log" || "$log" == \#* ]] && continue
  if [ -f "$log" ] && tail -1 "$log" | grep -q "__EXIT 0"; then echo "skip $log"; continue; fi
  mkdir -p "$(dirname "$log")"
  echo "=== $(date +%H:%M) $cmd"
  bash -c "$cmd" 2>&1 | tee "$log"
  echo "__EXIT ${PIPESTATUS[0]}" >> "$log"
done < "$1"
echo QUEUE_DONE
