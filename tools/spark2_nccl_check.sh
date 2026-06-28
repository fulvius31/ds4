#!/usr/bin/env bash
#
# spark2_nccl_check.sh — bring-up + end-to-end validation for two DGX Sparks
# wired together by the ConnectX-7 stacking cable.
#
# It (1) auto-detects each box's RDMA device / netdev / link IP straight from
# sysfs — handling the f0-vs-f1 PCI-enumeration asymmetry so you never have to
# guess which device to name; (2) installs the tooling (perftest, NCCL, OpenMPI,
# and builds NVIDIA nccl-tests); and (3) validates the link:
#     1. ib_write_bw     RDMA bandwidth     gate: >= 90 Gbps/rail (healthy link)
#     2. ib_write_lat    RDMA latency       gate: <= 3 us typical
#     3. all_reduce_perf NCCL/RoCE          gate: NET/IB + #wrong 0 + <~60 us @16-32K
#
# Run it ON THE MASTER Spark (it becomes rank 0); it drives the PEER over SSH.
# Requires passwordless SSH master -> peer (run: ssh-copy-id <peer> once).
#
#   ./spark2_nccl_check.sh <peer-ssh-host>
#   ./spark2_nccl_check.sh --detect-only                 # just print THIS box's link info
#   ./spark2_nccl_check.sh <peer> --skip-install         # tooling already installed
#   ./spark2_nccl_check.sh <peer> --master-ip 10.0.0.1 --peer-ip 10.0.0.2
#
# It does NOT change IPs. If a link interface has no IPv4, it tells you the exact
# `sudo ip addr add` to run on each box first.

set -uo pipefail

# ----- pretty output --------------------------------------------------------
RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; BLD=$'\033[1m'; RST=$'\033[0m'
say()  { printf '%s\n' "$*"; }
hdr()  { printf '\n%s== %s ==%s\n' "$BLD" "$*" "$RST"; }
ok()   { printf '  %s[ OK ]%s %s\n' "$GRN" "$RST" "$*"; }
warn() { printf '  %s[WARN]%s %s\n' "$YLW" "$RST" "$*"; }
bad()  { printf '  %s[FAIL]%s %s\n' "$RED" "$RST" "$*"; }
die()  { printf '%s[error]%s %s\n' "$RED" "$RST" "$*" >&2; exit 1; }

FAILS=0; WARNS=0
gate_ok()   { ok   "$*"; }
gate_warn() { warn "$*"; WARNS=$((WARNS+1)); }
gate_bad()  { bad  "$*"; FAILS=$((FAILS+1)); }

# ----- args -----------------------------------------------------------------
PEER=""; DETECT_ONLY=0; SKIP_INSTALL=0; MIP_OVR=""; PIP_OVR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --detect-only) DETECT_ONLY=1 ;;
    --skip-install) SKIP_INSTALL=1 ;;
    --master-ip) MIP_OVR="${2:?}"; shift ;;
    --peer-ip)   PIP_OVR="${2:?}"; shift ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) die "unknown flag: $1" ;;
    *) PEER="$1" ;;
  esac
  shift
done

# ----- detection (pure sysfs + iproute2 — works BEFORE any install) ---------
# Echoes three lines: DEV / NETDEV / IP  (IP may be empty).  Also prints a full
# inventory of every IB device + state to stderr so you can see "everything".
DETECT_SNIPPET='
  printf "  %-14s %-8s %-16s %s\n" DEVICE STATE NETDEV IPv4 >&2
  best_dev=""; best_net=""; best_ip=""
  for p in /sys/class/infiniband/*/ports/1/state; do
    [ -e "$p" ] || continue
    dev=$(basename "$(dirname "$(dirname "$(dirname "$p")")")")
    st=$(cat "$p" 2>/dev/null | sed "s/.*: //")
    net=$(ls "/sys/class/infiniband/$dev/device/net" 2>/dev/null | head -1)
    ip4=$(ip -4 -o addr show dev "$net" 2>/dev/null | awk "{print \$4}" | head -1)
    printf "  %-14s %-8s %-16s %s\n" "$dev" "$st" "${net:--}" "${ip4:--}" >&2
    if [ "$st" = "ACTIVE" ] && [ -z "$best_dev" ]; then best_dev="$dev"; best_net="$net"; best_ip="${ip4%/*}"; fi
    # prefer an ACTIVE device that already has an IPv4 (the live, configured rail)
    if [ "$st" = "ACTIVE" ] && [ -n "$ip4" ]; then best_dev="$dev"; best_net="$net"; best_ip="${ip4%/*}"; fi
  done
  printf "%s\n%s\n%s\n" "$best_dev" "$best_net" "$best_ip"
'
# Pipe the snippet to bash on stdin (local) or ssh-bash (peer): no nested-quote
# re-parsing, identical behavior both sides. stderr (the inventory) shows live;
# stdout (the 3 result lines) is captured by the caller.
detect_local() { printf '%s' "$DETECT_SNIPPET" | bash; }
detect_peer()  { printf '%s' "$DETECT_SNIPPET" | ssh -o BatchMode=yes "$PEER" bash; }

read_detect() {  # $1=local|peer  -> sets DEV NET IP (globals via name)
  local out; if [ "$1" = local ]; then out=$(detect_local); else out=$(detect_peer); fi
  DEV=$(printf '%s\n' "$out" | sed -n 1p)
  NET=$(printf '%s\n' "$out" | sed -n 2p)
  IP=$(printf  '%s\n' "$out" | sed -n 3p)
}

net30() { local IFS=.; read -r a b c d <<<"$1"; printf '%s.%s.%s.%s/30\n' "$a" "$b" "$c" "$((d & 0xFC))"; }

# ----- DETECT-ONLY mode -----------------------------------------------------
hdr "Detecting RDMA link on this box ($(hostname))"
read_detect local; MDEV="$DEV"; MNET="$NET"; MIP="${MIP_OVR:-$IP}"
[ -n "$MDEV" ] || die "no ACTIVE InfiniBand/RoCE port found on this box — is the cable seated and the link up? (check: ls /sys/class/infiniband, ip link)"
say "  -> using device ${BLD}$MDEV${RST} netdev ${BLD}$MNET${RST} ip ${BLD}${MIP:-<none>}${RST}"

if [ "$DETECT_ONLY" = 1 ]; then
  [ -n "$MIP" ] || warn "this rail has no IPv4 — assign one, e.g.:  sudo ip addr add 10.0.0.1/30 dev $MNET && sudo ip link set $MNET up"
  exit 0
fi

[ -n "$PEER" ] || die "give the peer's SSH host as the first arg (or use --detect-only). e.g. ./spark2_nccl_check.sh 10.0.0.2"
ssh -o BatchMode=yes -o ConnectTimeout=5 "$PEER" true 2>/dev/null \
  || die "passwordless SSH to '$PEER' failed. Run:  ssh-copy-id $PEER"

hdr "Detecting RDMA link on the peer ($PEER)"
read_detect peer;  PDEV="$DEV"; PNET="$NET"; PIP="${PIP_OVR:-$IP}"
[ -n "$PDEV" ] || die "no ACTIVE IB/RoCE port on peer $PEER"
say "  -> peer device ${BLD}$PDEV${RST} netdev ${BLD}$PNET${RST} ip ${BLD}${PIP:-<none>}${RST}"

# IPs are required for the link tests.
if [ -z "$MIP" ] || [ -z "$PIP" ]; then
  bad "the link rails need IPv4 addresses on the SAME /30 subnet. Assign them, then re-run:"
  say "    # on THIS box:   sudo ip addr add 10.0.0.1/30 dev $MNET && sudo ip link set $MNET up mtu 9000"
  say "    # on $PEER:      sudo ip addr add 10.0.0.2/30 dev $PNET && sudo ip link set $PNET up mtu 9000"
  exit 1
fi
SUBNET=$(net30 "$MIP")
say "  link subnet: ${BLD}$SUBNET${RST}   (rank0=$MIP via $MDEV, rank1=$PIP via $PDEV)"

# ----- INSTALL --------------------------------------------------------------
INSTALL_SNIPPET='
  set -e
  export DEBIAN_FRONTEND=noninteractive
  sudo apt-get update -qq
  sudo apt-get install -y -qq perftest ibverbs-utils infiniband-diags rdma-core \
       libnccl2 libnccl-dev libopenmpi-dev openmpi-bin build-essential git iproute2
  if [ ! -x "$HOME/nccl-tests/build/all_reduce_perf" ]; then
    [ -d "$HOME/nccl-tests" ] || git clone --depth 1 https://github.com/NVIDIA/nccl-tests "$HOME/nccl-tests"
    MPI_HOME=/usr/lib/$(uname -m)-linux-gnu/openmpi
    make -C "$HOME/nccl-tests" MPI=1 MPI_HOME="$MPI_HOME" NCCL_HOME=/usr -j"$(nproc)" >/tmp/ncclt_build.log 2>&1 \
      || { echo "nccl-tests build FAILED:"; tail -20 /tmp/ncclt_build.log; exit 1; }
  fi
  echo "install-ok $(hostname)"
'
if [ "$SKIP_INSTALL" = 0 ]; then
  hdr "Installing tooling + building nccl-tests (this box + peer)"
  printf '%s' "$INSTALL_SNIPPET" | bash 2>&1 | sed "s/^/  [local] /" || die "local install failed"
  printf '%s' "$INSTALL_SNIPPET" | ssh "$PEER" bash 2>&1 | sed "s/^/  [peer ] /" || die "peer install failed"
  ok "tooling ready on both boxes"
else
  warn "skipping install (--skip-install)"
fi

# ----- TEST 1: RDMA bandwidth (ib_write_bw) ---------------------------------
hdr "1/3  RDMA bandwidth  (ib_write_bw)"
pkill -f ib_write_bw 2>/dev/null; ssh "$PEER" "pkill -f ib_write_bw 2>/dev/null" || true
ib_write_bw -d "$MDEV" -i 1 -p 12000 -F --report_gbits >/tmp/bw_srv.log 2>&1 &
sleep 2
ssh "$PEER" "ib_write_bw -d $PDEV -i 1 -p 12000 -F --report_gbits $MIP" >/tmp/bw_cli.log 2>&1 || true
wait 2>/dev/null || true
say "  --- client result ---"; grep -E '^[[:space:]]*[0-9]' /tmp/bw_cli.log | tail -3 | sed 's/^/  /'
BW=$(grep -E '^[[:space:]]*[0-9]' /tmp/bw_cli.log | tail -1 | awk '{print $4}')
BW=${BW%%.*}
if   [ -z "$BW" ];            then gate_bad "no bandwidth result — see /tmp/bw_cli.log"
elif [ "$BW" -ge 90 ];        then gate_ok  "bandwidth ${BW} Gbps/rail — healthy"
elif [ "$BW" -ge 30 ];        then gate_warn "bandwidth ${BW} Gbps — below per-rail line rate (cable/PCIe?)"
else                               gate_bad "bandwidth ${BW} Gbps — FIRMWARE THROTTLE likely (update CX7 fw + full NIC power-drain)"; fi

# ----- TEST 2: RDMA latency (ib_write_lat) ----------------------------------
hdr "2/3  RDMA latency  (ib_write_lat)"
pkill -f ib_write_lat 2>/dev/null; ssh "$PEER" "pkill -f ib_write_lat 2>/dev/null" || true
ib_write_lat -d "$MDEV" -i 1 -p 12001 -F >/tmp/lat_srv.log 2>&1 &
sleep 2
ssh "$PEER" "ib_write_lat -d $PDEV -i 1 -p 12001 -F $MIP" >/tmp/lat_cli.log 2>&1 || true
wait 2>/dev/null || true
say "  --- client result ---"; grep -E '^[[:space:]]*[0-9]' /tmp/lat_cli.log | tail -2 | sed 's/^/  /'
LAT=$(grep -E '^[[:space:]]*[0-9]' /tmp/lat_cli.log | tail -1 | awk '{print $5}')   # t_typical
LATI=${LAT%%.*}
if   [ -z "$LATI" ];       then gate_bad "no latency result — see /tmp/lat_cli.log"
elif [ "$LATI" -le 3 ];    then gate_ok  "latency ${LAT} us typical — excellent"
elif [ "$LATI" -le 10 ];   then gate_warn "latency ${LAT} us — higher than expected"
else                            gate_bad "latency ${LAT} us — link unhealthy"; fi

# ----- TEST 3: NCCL all-reduce over RoCE (all_reduce_perf via mpirun) --------
hdr "3/3  NCCL all-reduce over RoCE  (all_reduce_perf, 8B..128M)"
MPIRUN=$(command -v mpirun); ARP="$HOME/nccl-tests/build/all_reduce_perf"
[ -x "$ARP" ] || die "all_reduce_perf not built ($ARP) — re-run without --skip-install"
"$MPIRUN" -np 2 -H "$MIP","$PIP" \
  --allow-run-as-root \
  --mca oob_tcp_if_include "$SUBNET" --mca btl_tcp_if_include "$SUBNET" \
  -x NCCL_NET_PLUGIN=none -x NCCL_IB_MERGE_NICS=1 \
  -x NCCL_IB_HCA="$MDEV,$PDEV" -x NCCL_SOCKET_IFNAME="$MNET,$PNET" \
  -x NCCL_DEBUG=INFO \
  "$ARP" -b 8 -e 128M -f 2 -g 1 >/tmp/nccl.log 2>&1 || true

say "  --- size / in-place time(us) / busbw(GB/s) / #wrong ---"
# data rows carry out-of-place then in-place results; in-place = last 4 cols:
#   ... time(NF-3) algbw(NF-2) busbw(NF-1) #wrong(NF)
grep -E '^[[:space:]]*[0-9]+[[:space:]]' /tmp/nccl.log \
  | awk '{printf "  %10s  %8s us  %8s GB/s  wrong=%s\n",$1,$(NF-3),$(NF-1),$NF}' | sed -n '1,12p'

# transport check
if grep -q 'NET/IB' /tmp/nccl.log; then gate_ok "NCCL on NET/IB (RoCE) — not a TCP fallback"
elif grep -q 'NET/Socket' /tmp/nccl.log; then gate_bad "NCCL fell back to NET/Socket (TCP) — RoCE not used; check NCCL_NET_PLUGIN/IB_HCA"
else gate_warn "could not confirm NCCL transport from log (/tmp/nccl.log)"; fi
# correctness
if grep -E '^[[:space:]]*[0-9]' /tmp/nccl.log | awk '{print $NF}' | grep -qvE '^(0|N/A)$'; then
  gate_bad "all-reduce produced WRONG results (nonzero #wrong) — see /tmp/nccl.log"
else gate_ok "all-reduce numerically correct (#wrong 0)"; fi
# small-message latency @ 16384 B
US=$(grep -E '^[[:space:]]*16384[[:space:]]' /tmp/nccl.log | head -1 | awk '{print $(NF-3)}')
USI=${US%%.*}
if   [ -z "$USI" ];        then gate_warn "no 16K row in NCCL output (see /tmp/nccl.log)"
elif [ "$USI" -le 60 ];    then gate_ok  "16K all-reduce ${US} us — within budget for a per-layer collective"
elif [ "$USI" -le 200 ];   then gate_warn "16K all-reduce ${US} us — usable but high"
else                            gate_bad "16K all-reduce ${US} us — too slow for latency-coupled work"; fi

# ----- SUMMARY --------------------------------------------------------------
hdr "Summary"
say "  master: $MDEV / $MNET / $MIP        peer: $PDEV / $PNET / $PIP"
say "  logs:   /tmp/bw_cli.log  /tmp/lat_cli.log  /tmp/nccl.log"
if [ "$FAILS" -gt 0 ]; then
  printf '%s  RESULT: %d FAIL, %d WARN — the link is NOT ready.%s\n' "$RED" "$FAILS" "$WARNS" "$RST"; exit 1
elif [ "$WARNS" -gt 0 ]; then
  printf '%s  RESULT: PASS with %d warning(s) — usable, review above.%s\n' "$YLW" "$WARNS" "$RST"; exit 0
else
  printf '%s  RESULT: ALL GREEN — the two Sparks are wired and collectives work.%s\n' "$GRN" "$RST"; exit 0
fi
