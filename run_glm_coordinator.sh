#!/bin/sh
# Coordinator side of the two-Spark GLM 5.2 distributed run. Run on 10.0.0.1.
# Start the worker on 10.0.0.2 first. Extra args are passed through, e.g.:
#   ./run_glm_coordinator.sh -p "Hello" --temp 0
MODEL=gguf/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
cd "$(dirname "$0")"
# See run_glm_worker.sh: default 32 GiB guard reserve does not fit a 121 GiB Spark.
export DS4_GLM_MEMORY_GUARD_RESERVE_GB=12
# --dist-prefill-chunk 1: the CUDA GLM multi-token batch prefill path is
# numerically broken (single-token evaluation matches the CPU reference
# exactly; batched evaluation does not). Until that is fixed, prefill one
# token at a time. -c 2048: larger contexts push the 0:40 slice past the
# ~104 GiB planned-footprint ceiling of a 121 GiB Spark.
exec ./ds4 -m "$MODEL" --cuda \
  --role coordinator \
  --layers 0:40 \
  --listen 10.0.0.1 9911 \
  -c 2048 \
  --dist-prefill-chunk 1 \
  "$@"
