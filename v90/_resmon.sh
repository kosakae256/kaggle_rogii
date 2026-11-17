#!/bin/bash
while true; do
  G=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1)
  R=$(wmic OS get FreePhysicalMemory /value 2>/dev/null | tr -d '\r' | grep = )
  echo "$(date +%T) GPU[$G] RAM_free_KB[$R]"
  sleep 120
done
