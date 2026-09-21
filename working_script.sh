#!/usr/bin/env bash
set -euo pipefail
MAP="/home/jeswes99/Desktop/Skole/Masteroppgave/name_taxid_map.tsv"
OUTDIR="/home/jeswes99/Desktop/Skole/Masteroppgave/ncbi_downloads"
LOG="/home/jeswes99/Desktop/Skole/Masteroppgave/download_log.txt"
mkdir -p "$OUTDIR"
: > "$LOG"

while IFS=$'\t' read -r name taxid || [ -n "$name" ]; do
  [ -z "$name" ] && continue
  safe=$(echo "$name" | tr ' /' '_' | tr -cd 'A-Za-z0-9_.' )
  zipfn="${OUTDIR}/${safe}.zip"
  echo "=== $name ===" | tee -a "$LOG"
  if [ -n "$taxid" ]; then
    echo "Using taxid $taxid" | tee -a "$LOG"
    if datasets download genome taxon-id "$taxid" --reference --filename "$zipfn" 2>>"$LOG"; then
      echo "OK: reference downloaded for $name" | tee -a "$LOG"
      unzip -o "$zipfn" -d "${OUTDIR}/${safe}" >>"$LOG" 2>&1 || true
      continue
    fi
    echo "No reference for taxid $taxid — trying best available" | tee -a "$LOG"
    if datasets download genome taxon-id "$taxid" --filename "$zipfn" 2>>"$LOG"; then
      echo "OK: best assembly downloaded for $name" | tee -a "$LOG"
      unzip -o "$zipfn" -d "${OUTDIR}/${safe}" >>"$LOG" 2>&1 || true
      continue
    fi
  else
    echo "No taxid for ${name}, attempting name-based download" | tee -a "$LOG"
    if datasets download genome taxon "${name}" --reference --filename "$zipfn" 2>>"$LOG"; then
      echo "OK: reference downloaded for $name" | tee -a "$LOG"
      unzip -o "$zipfn" -d "${OUTDIR}/${safe}" >>"$LOG" 2>&1 || true
      continue
    fi
    echo "No reference by name; trying best available" | tee -a "$LOG"
    if datasets download genome taxon "${name}" --filename "$zipfn" 2>>"$LOG"; then
      echo "OK: best assembly downloaded for $name" | tee -a "$LOG"
      unzip -o "$zipfn" -d "${OUTDIR}/${safe}" >>"$LOG" 2>&1 || true
      continue
    fi
  fi
  echo "ERROR: no assemblies found for ${name}" | tee -a "$LOG"
  sleep 0.3
done < "$MAP"

echo "Done. See $OUTDIR and $LOG"