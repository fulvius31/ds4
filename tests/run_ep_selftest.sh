#!/bin/sh
# Compile and run the ds4 Expert Parallelism (EP) host-logic self-test.
# Requires only a C compiler (no GPU/CUDA/NCCL). See EP_IMPLEMENTATION_PLAN.md §10.
#
# Usage:   sh tests/run_ep_selftest.sh        (or set CC=clang, etc.)
set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
CC="${CC:-cc}"
OUT="$(mktemp -d)/ds4_ep_selftest"

"$CC" -DDS4_EP_SELFTEST -O2 -Wall -Wextra "$ROOT/ds4_ep.c" -o "$OUT"
"$OUT"
