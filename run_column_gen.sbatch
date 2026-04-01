#!/bin/bash
# =============================================================================
# SLURM job script — Column Generation solver (MPI across 4 nodes)
#
# Submit from the login/master node:
#   sbatch run_column_gen.sbatch
#
# Full log (stdout + stderr merged) lands in:
#   logs/exp_<JOBID>.log
# =============================================================================

# ── SLURM resource directives ─────────────────────────────────────────────────

#SBATCH --job-name=col_gen

# 4 compute nodes, 1 MPI rank each  (rank 0 = master, ranks 1-3 = workers)
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1

# CPUs per rank:
#   - Rank 0 uses these for Phase 0 init MIPs (ProcessPoolExecutor) and for
#     the LP master + integer master (HiGHS auto-detects via os.cpu_count(),
#     which SLURM constrains to this value via cgroups).
#   - Worker ranks run pricing subproblems sequentially (1 CPU at a time),
#     so extra CPUs on workers are idle.  Set lower if cores are scarce.
#SBATCH --cpus-per-task=2

# Walltime — adjust to your largest config.  A 10-UE / 20-MHz run typically
# converges in 10-30 min.  Err on the side of caution.
#SBATCH --time=02:00:00

# Merge stdout + stderr into a single file named exp_<JOBID>.log
#SBATCH --output=logs/exp_%j.log
#SBATCH --error=logs/exp_%j.log

# ── Partition / QOS — EDIT these for your cluster ────────────────────────────
#SBATCH --partition=compute
##SBATCH --qos=normal          # uncomment if your cluster requires a QOS

# Optional: send email on job start/end/fail
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=your@email.com


# =============================================================================
# Environment setup
# =============================================================================

echo "============================================================"
echo "  Job ID        : ${SLURM_JOB_ID}"
echo "  Job name      : ${SLURM_JOB_NAME}"
echo "  Nodes         : ${SLURM_NNODES}"
echo "  Tasks/node    : ${SLURM_NTASKS_PER_NODE}"
echo "  CPUs/task     : ${SLURM_CPUS_PER_TASK}"
echo "  Node list     : ${SLURM_NODELIST}"
echo "  Submit dir    : ${SLURM_SUBMIT_DIR}"
echo "  Start time    : $(date)"
echo "============================================================"

# Load modules — EDIT names to match your cluster's module system.
# Common alternatives:
#   module load python/3.10  openmpi/4.1  (OpenHPC / Spack)
#   module load anaconda3    mpi4py        (some university clusters)
#   module load python3                   (minimal cluster)
module load python/3.10    2>/dev/null || true
module load openmpi/4.1    2>/dev/null || true

# Activate conda environment.
# Adjust CONDA_ENV if you use a named env instead of base.
CONDA_ENV="base"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "Python  : $(which python) — $(python --version 2>&1)"
echo "mpi4py  : $(python -c 'import mpi4py; print(mpi4py.__version__)' 2>&1)"
echo "PuLP    : $(python -c 'import pulp; print(pulp.__version__)' 2>&1)"
echo "HiGHS   : $(python -c '
import pulp
cmd_ok = False
api_ok = False
try:
    cmd_ok = pulp.HiGHS_CMD(msg=0).available()
except Exception:
    pass
try:
    api_ok = pulp.HiGHSpy(msg=0).available()
except Exception:
    pass
print(f"CMD={cmd_ok}  API(highspy)={api_ok}")
' 2>&1)"
echo "------------------------------------------------------------"

# Abort early if mpi4py is missing — prevents 4 independent serial solves.
python -c "import mpi4py" 2>/dev/null || {
    echo "ERROR: mpi4py not found. Install it first:"
    echo "  conda install -c conda-forge mpi4py"
    exit 1
}

# Warn if neither HiGHS path works — solver falls back to CBC (slower).
python -c "
import pulp, sys
cmd_ok = False
api_ok = False
try:
    cmd_ok = pulp.HiGHS_CMD(msg=0).available()
except Exception:
    pass
try:
    api_ok = pulp.HiGHSpy(msg=0).available()
except Exception:
    pass
if not (cmd_ok or api_ok):
    print('WARNING: HiGHS not available — using CBC (slower).')
    print('  Binary:  conda install -c conda-forge highs')
    print('  Python:  pip install highspy')
" 2>/dev/null || true


# =============================================================================
# Job parameters — edit here, not below
# =============================================================================

# Config file (relative to SLURM_SUBMIT_DIR = project root)
CONFIG="configs/realistic_10ue_20mhz.json"

# Output directory for the solution .txt file
OUTPUT_DIR="approx_sol"

# Output directory for plots
PLOT_DIR="figs"

# Max CG iterations
MAX_ITER=200

# Intra-rank workers for Phase 0 init MIPs on rank 0 only.
# Set to number of VUEs in the config (or cpu_count); 0 = auto.
N_WORKERS=0

# Slice priority weights
ALPHA_URLLC=2.0
ALPHA_EMBB=1.0

# Log verbosity: DEBUG | INFO | WARNING
LOG_LEVEL="INFO"


# =============================================================================
# Run
# =============================================================================

cd "${SLURM_SUBMIT_DIR}"
mkdir -p "${OUTPUT_DIR}" "${PLOT_DIR}" logs

echo "Config  : ${CONFIG}"
echo "Output  : ${OUTPUT_DIR}"
echo "Threads : auto (os.cpu_count() = ${SLURM_CPUS_PER_TASK} via SLURM cgroups)"
echo "============================================================"

# srun launches one MPI rank per node.
# --mpi=pmix    works with OpenMPI >= 3 and SLURM >= 18.
# If you see "No MPI backend" errors try:  --mpi=pmi2
srun --mpi=pmi2 \
    python -m column_gen \
        --config          "${CONFIG}"         \
        --output          "${OUTPUT_DIR}"     \
        --max-iterations  "${MAX_ITER}"       \
        --n-workers       "${N_WORKERS}"      \
        --alpha-urllc     "${ALPHA_URLLC}"    \
        --alpha-embb      "${ALPHA_EMBB}"     \
        --log-level       "${LOG_LEVEL}"      \
        --job-id          "${SLURM_JOB_ID}"

EXIT_CODE=$?

# ── Plot solution (runs on the head node after srun completes) ────────────────
if [ "${EXIT_CODE}" -eq 0 ]; then
    FILENAME=$(basename "${CONFIG}" .json)
    SOLUTION="${OUTPUT_DIR}/${FILENAME}_job${SLURM_JOB_ID}.txt"
    echo "Plotting solution: ${SOLUTION} → ${PLOT_DIR}/"
    python milp-5g-nr.py \
        --config   "${CONFIG}"   \
        --solution "${SOLUTION}" \
        --plot                   \
        --plot-output "${PLOT_DIR}/"
    PLOT_CODE=$?
    echo "Plot exit code: ${PLOT_CODE}"
else
    echo "Skipping plot — solver exited with code ${EXIT_CODE}"
fi

echo "============================================================"
echo "  End time : $(date)"
echo "  Exit code: ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
