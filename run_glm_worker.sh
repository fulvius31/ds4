#!/bin/sh
# Worker side of the two-Spark GLM 5.2 distributed run. Run on 10.0.0.2.
# Start this BEFORE the coordinator; it retries until the coordinator is up.
MODEL=gguf/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
cd "$(dirname "$0")"
# 94.8 GiB slice + graph does not fit under the default 32 GiB guard
# reserve on a 121 GiB Spark; 12 GiB still leaves the OS enough room.
# A slice starting at a selection-reuse layer (41 here) is promoted to a
# full-indexer layer by the engine: it recomputes its own token selection
# instead of depending on coordinator-side state.
export DS4_GLM_MEMORY_GUARD_RESERVE_GB=12
exec ./ds4 -m "$MODEL" --cuda \
  --role worker \
  --layers 40:output \
  -c "${GLM_CTX:-12288}" \
  --coordinator 10.0.0.1 9911
