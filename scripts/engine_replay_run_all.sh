#!/bin/zsh
# Year-to-date engine replay in monthly chunks, N in parallel. Usage: scripts/engine_replay_run_all.sh [parallel] [scan-step]
P=${1:-6}; STEP=${2:-60}
cd "$(dirname "$0")/.."
LOGD=reports/backtest_cache/replay/logs; mkdir -p $LOGD
CH=("2026-01-01 2026-02-01 m01" "2026-02-01 2026-03-01 m02" "2026-03-01 2026-04-01 m03" "2026-04-01 2026-05-01 m04" \
    "2026-05-01 2026-06-01 m05" "2026-06-01 2026-07-01 m06" "2026-07-01 2026-08-01 m07" "2026-08-01 2026-09-01 m08" "2026-09-01 2026-09-16 m09")
printf '%s\n' "${CH[@]}" | xargs -P $P -L 1 sh -c 'venv/bin/python scripts/engine_replay.py --start $0 --end $1 --tag $2 --scan-step '"$STEP"' --log WARNING > '"$LOGD"'/$2.log 2>&1; echo "chunk $2 exit $?"'
