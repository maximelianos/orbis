#!/bin/bash
# Run a command in a tmux window of the shared session and log its output.
#   tools/tmux/run.sh <window> <logfile> <command...>
# The window must already hold the navsim env with CUDA_VISIBLE_DEVICES=0.
# When the command ends, "__EXIT <code>" is appended to the log.
win=$1; log=$2; shift 2
mkdir -p "$(dirname "$log")"
: > "$log"
tmux -S /tmp/tmux-socket-12345 send-keys -t "mysession:$win" \
  "clear; ( $* ) 2>&1 | tee -a $log; echo __EXIT \${PIPESTATUS[0]} >> $log" Enter
