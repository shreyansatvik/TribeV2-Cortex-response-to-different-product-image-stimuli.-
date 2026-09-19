#!/bin/zsh
# Guardrail. Swap growth is the signal that actually precedes a Mac freezing.
LOG=viz/watchdog.log; : > $LOG
BASE_SWAP=3316.94      # MB at launch
MAX_SWAP_GROWTH=2600 # MB above baseline -> kill
MAX_GPU_GB=11.0
MIN_AVAIL_GB=0.6     # hard backstop
echo "$(date +%T) baseline swap=${BASE_SWAP}M growth_limit=${MAX_SWAP_GROWTH}M" >> $LOG
while true; do
  P=$(pgrep -f "run_analysis.py" | head -1)
  [[ -z "$P" ]] && { echo "$(date +%T) job ended, watchdog exiting" >> $LOG; exit 0; }
  AVAIL=$(vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3} END{printf "%.2f",(f+i)*16384/1073741824}')
  GPU=$(ioreg -r -d 1 -c IOAccelerator 2>/dev/null | grep -o '"In use system memory"=[0-9]*' | head -1 | cut -d= -f2)
  GPUGB=$(echo "scale=2; ${GPU:-0}/1073741824" | bc)
  SWAP=$(sysctl -n vm.swapusage | sed 's/.*used = \([0-9.]*\)M.*/\1/')
  GROWTH=$(echo "${SWAP:-0} - $BASE_SWAP" | bc)
  echo "$(date +%T) avail=${AVAIL}GB gpu=${GPUGB}GB swap=${SWAP}M (+${GROWTH}M)" >> $LOG
  if (( $(echo "$GROWTH > $MAX_SWAP_GROWTH" | bc -l) )) || \
     (( $(echo "$GPUGB > $MAX_GPU_GB" | bc -l) )) || \
     (( $(echo "$AVAIL < $MIN_AVAIL_GB" | bc -l) )); then
    echo "$(date +%T) !! BREACH (swap+${GROWTH}M gpu ${GPUGB}GB avail ${AVAIL}GB) - killing $P" >> $LOG
    kill -9 $P 2>/dev/null; exit 1
  fi
  sleep 5
done
