#!/bin/bash
# (part of bls_cuda; see cudaFiles/README.md)
# Run a command with GPU temperature logging; kill it if the GPU reaches LIMIT C.
# Usage: gpurun.sh LOGFILE CMD...   (env LIMIT, default 83)
LIMIT=${LIMIT:-83}
LOG=$1; shift
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,fan.speed,utilization.gpu,clocks_event_reasons.active \
  --format=csv,noheader,nounits -lms 1000 > "$LOG" &
SMI=$!
"$@" &
JOB=$!
while kill -0 $JOB 2>/dev/null; do
  T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
  T=${T%.*}; if [ "${T:-0}" -ge "$LIMIT" ]; then
    echo "!!! GPU at ${T} C >= ${LIMIT} C: killing job" >&2
    kill $JOB; sleep 1; kill -9 $JOB 2>/dev/null
  fi
  sleep 1
done
wait $JOB; RC=$?
kill $SMI 2>/dev/null
awk -F', ' 'NR==1{t0=$2} {if($2>tm)tm=$2; if($3>pm)pm=$3; if($4>cm)cm=$4; n++} END{printf "[gpu] %d s, temp start %s max %s C, power max %.1f W, SM clock max %s MHz\n", n, t0, tm, pm, cm}' "$LOG" >&2
exit $RC
