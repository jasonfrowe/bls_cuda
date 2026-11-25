#!/bin/bash
# (part of bls_cuda; see cudaFiles/README.md)
# Run a command with GPU temperature logging; kill it if the GPU reaches LIMIT C.
# Usage: gpurun.sh LOGFILE CMD...   (env LIMIT, default 83)
LIMIT=${LIMIT:-83}
LOG=$1; shift
# columns: timestamp, temp C, power W, SM MHz, fan %, util %, clock-limit reasons (older drivers name
# the last field clocks_throttle_reasons)
Q=timestamp,temperature.gpu,power.draw,clocks.sm,fan.speed,utilization.gpu,clocks_event_reasons.active
nvidia-smi --query-gpu=$Q --format=csv,noheader >/dev/null 2>&1 || Q=${Q/clocks_event_reasons/clocks_throttle_reasons}
nvidia-smi --query-gpu=$Q --format=csv,noheader,nounits -lms 1000 > "$LOG" &
SMI=$!
"$@" &
JOB=$!
while kill -0 $JOB 2>/dev/null; do
  # hottest visible GPU (several lines on a multi-GPU node); non-numeric (e.g. [N/A]) counts as 0
  T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
  T=${T%.*}; [[ "$T" =~ ^[0-9]+$ ]] || T=0
  if [ "$T" -ge "$LIMIT" ]; then
    echo "!!! GPU at ${T} C >= ${LIMIT} C: killing job" >&2
    kill $JOB; sleep 1; kill -9 $JOB 2>/dev/null
  fi
  sleep 1
done
wait $JOB; RC=$?
kill $SMI 2>/dev/null
awk -F', ' 'NR==1{t0=$2} {if($2>tm)tm=$2; if($3>pm)pm=$3; if($4>cm)cm=$4; n++} END{printf "[gpu] %d s, temp start %s max %s C, power max %.1f W, SM clock max %s MHz\n", n, t0, tm, pm, cm}' "$LOG" >&2
exit $RC
