#!/bin/bash
# run_exp.sh — Full experiment pipeline
#
#   1. Generate config
#   2. Run Column Generation
#   3. Solve integer master with Gurobi
#   4. Plot resource-grid
#
# Usage:
#   ./run_exp.sh                    # run with default params below
#   ./run_exp.sh my_custom_name     # use a custom experiment folder name
#
# Each run creates a self-contained experiment folder:
#   experiments/YYYYMMDD_HHMMSS_bw5M_K3_ue3_mcs26/
#     config.json      — experiment parameters (copy)
#     experiment.log   — complete pipeline log (all 4 steps)
#     *.lp / *.mps        — master problem files
#     *_columns.json      — columns registry for plotting
#     solution.sol        — selected columns
#     grid.png            — resource-grid plot
#     shadow_prices.png   — LP dual price heatmap
#
# Edit the PARAMETERS section below to change the experiment.

set -euo pipefail
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

# ── Gurobi ────────────────────────────────────────────────────────────
export GUROBI_HOME="/opt/gurobi1101/linux64"
export PATH="${GUROBI_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${GUROBI_HOME}/lib:${LD_LIBRARY_PATH}"
export GRB_LICENSE_FILE="/home/prarabdha-srivastava/Desktop/gurobi.lic"

# ── Python environment ────────────────────────────────────────────────
source ~/Desktop/work/env/bin/activate

# ═══════════════════════════════════════════════════════════════════════
#  PARAMETERS — edit these for each experiment
# ═══════════════════════════════════════════════════════════════════════
BW=5000000          # bandwidth in Hz
TIME_MS=3.0         # scheduling window in ms
MCS=26              # MCS index (0-28)
K=5                 # max slots per VUE
UE_COUNT=3          # number of physical UEs
EMBB_MBPS=4.0       # eMBB throughput requirement (Mbps)
EMBB_LAT_MS=3.0     # eMBB latency deadline (ms)
URLLC_MBPS=1.0      # uRLLC throughput requirement (Mbps)
URLLC_LAT_MS=0.5    # uRLLC latency deadline (ms)
CG_MAX_ITER=200     # max CG iterations
GUROBI_GAP=0.01     # MIP gap tolerance
GUROBI_THREADS=2    # solver threads
GUROBI_TIMELIMIT=72000  # solver time limit (s)

# Optional: set this to skip config generation and use an existing config file
# Leave empty to use the parameters above; set to a path to use an existing config
EXISTING_CONFIG="${EXISTING_CONFIG:-}"  # can be overridden from env: EXISTING_CONFIG=path ./run_exp.sh
# ═══════════════════════════════════════════════════════════════════════

BW_MHZ=$(python -c "print(int($BW/1e6))")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_TAG="bw${BW_MHZ}M_K${K}_ue${UE_COUNT}_mcs${MCS}"

# Allow overriding experiment folder name via first argument
if [ -n "$EXISTING_CONFIG" ]; then
    # Derive tag from config filename when using an existing config
    CFG_BASE=$(basename "$EXISTING_CONFIG" | sed 's/\.[^.]*$//')
    EXP_DIR="${1:-experiments/${TIMESTAMP}_${CFG_BASE}}"
else
    EXP_DIR="${1:-experiments/${TIMESTAMP}_${EXP_TAG}}"
fi

mkdir -p "$EXP_DIR"

# ── Single log file: everything goes here AND to the terminal ─────────
LOG="$EXP_DIR/experiment.log"
exec > >(tee "$LOG") 2>&1

SEP="================================================================"
echo "$SEP"
echo "  Experiment : $EXP_TAG"
echo "  Output dir : $EXP_DIR"
echo "  Log        : $LOG"
echo "  Started    : $(date)"
echo "$SEP"

# ── Step 1: Generate (or copy) config ────────────────────────────────
echo ""
echo "[1/4] Generating config..."

CONFIG="$EXP_DIR/config.json"

if [ -n "$EXISTING_CONFIG" ]; then
    cp "$EXISTING_CONFIG" "$CONFIG"
    echo "  Copied existing config: $EXISTING_CONFIG → $CONFIG"
else
    python milp-5g-nr.py \
        --sample-config   "$CONFIG"      \
        --BW              $BW            \
        --time-horizon-ms $TIME_MS       \
        --mcs             $MCS           \
        --K               $K             \
        --ue-count        $UE_COUNT      \
        --embb-mbps       $EMBB_MBPS     \
        --embb-latency-ms $EMBB_LAT_MS   \
        --urllc-mbps      $URLLC_MBPS    \
        --urllc-latency-ms $URLLC_LAT_MS
fi

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found at $CONFIG"
    exit 1
fi
echo "  Config → $CONFIG"

# ── Step 2: Column Generation ─────────────────────────────────────────
echo ""
echo "[2/4] Running Column Generation..."
echo "------------------------------------------------------------"

python -u milp-5g-nr.py \
    --config      "$CONFIG"    \
    --cg                       \
    --cg-max-iter $CG_MAX_ITER \
    --cg-output   "$EXP_DIR"

echo "------------------------------------------------------------"

MPS_FILE=$(ls "$EXP_DIR"/*.mps 2>/dev/null | head -1)
if [ -z "$MPS_FILE" ]; then
    echo "ERROR: No .mps file found in $EXP_DIR after CG run."
    exit 1
fi
COLS_FILE="${MPS_FILE%.mps}_columns.json"
echo "  MPS  → $MPS_FILE"
echo "  Cols → $COLS_FILE"

# ── Step 3: Gurobi ────────────────────────────────────────────────────
echo ""
echo "[3/4] Solving with Gurobi..."

SOL_FILE="$EXP_DIR/solution.sol"

gurobi_cl                        \
    MIPGap=$GUROBI_GAP           \
    TimeLimit=$GUROBI_TIMELIMIT  \
    Threads=$GUROBI_THREADS      \
    ResultFile="$SOL_FILE"       \
    "$MPS_FILE"

if [ ! -f "$SOL_FILE" ]; then
    echo "ERROR: Gurobi did not produce $SOL_FILE"
    exit 1
fi
echo "  Solution → $SOL_FILE"

# ── Step 4: Plot ──────────────────────────────────────────────────────
echo ""
echo "[4/5] Plotting solution..."

PLOT_FILE="$EXP_DIR/grid.png"
OBJ=$(grep "Objective value" "$SOL_FILE" | head -1 | awk '{print $NF}')

python milp-5g-nr.py              \
    --config      "$CONFIG"       \
    --cg-solution "$SOL_FILE"     \
    --cg-columns  "$COLS_FILE"    \
    --plot-output "$PLOT_FILE"    \
    --plot-title  "${EXP_TAG} | obj=${OBJ}"

# ── Step 5: Statistics ────────────────────────────────────────────────
echo ""
echo "[5/5] Solution statistics..."

python milp-5g-nr.py          \
    --config      "$CONFIG"   \
    --cg-solution "$SOL_FILE" \
    --cg-columns  "$COLS_FILE"\
    --stats

echo ""
echo "$SEP"
echo "  Done!  $(date)"
echo ""
echo "  $EXP_DIR/"
echo "    config.json         experiment parameters"
echo "    experiment.log      complete pipeline log (all 5 steps)"
echo "    *.mps / *.lp        master problem files"
echo "    *_columns.json      column registry"
echo "    solution.sol        selected columns"
echo "    grid.png            resource-grid plot"
echo "    shadow_prices.png   LP dual price heatmap"
echo "$SEP"
