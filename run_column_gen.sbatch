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
#     the LP master + integer master solver threads (--solver-threads below).
#   - Worker ranks run pricing subproblems sequentially (1 CPU at a time),
#     so extra CPUs on workers are idle.  Set lower if cores are scarce.
#SBATCH --cpus-per-task=8

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

# Activate your Python virtual environment.
# The venv must be on a filesystem visible from all compute nodes (NFS/Lustre).
VENV_PATH="${HOME}/Desktop/work/env"
source "${VENV_PATH}/bin/activate"

echo "Python  : $(which python) — $(python --version 2>&1)"
echo "mpi4py  : $(python -c 'from mpi4py import MPI; print(MPI.Get_library_version())' 2>&1 | head -1)"
echo "PuLP    : $(python -c 'import pulp; print(pulp.__version__)' 2>&1)"
echo "HiGHS   : $(python -c 'import pulp; print(pulp.HiGHS_CMD(msg=0).available())' 2>&1)"
echo "------------------------------------------------------------"


# =============================================================================
# Job parameters — edit here, not below
# =============================================================================

# Config file (relative to SLURM_SUBMIT_DIR = project root)
CONFIG="configs/cfg-bw-5.0M-time-ms-3.0-mcs-26-K-3-ue-3-embb-thr-4.0-embb-lat-3.0-urllc-thr-1.0-urllc-lat-0.5.json"

# Output directory for the solution .txt file
OUTPUT_DIR="approx_sol"

# Max CG iterations
MAX_ITER=200

# Threads for HiGHS LP master + integer master on rank 0.
# Set equal to --cpus-per-task (rank 0 runs the master problem alone).
SOLVER_THREADS=${SLURM_CPUS_PER_TASK}

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
mkdir -p "${OUTPUT_DIR}" logs

echo "Config  : ${CONFIG}"
echo "Output  : ${OUTPUT_DIR}"
echo "Threads : ${SOLVER_THREADS}  (HiGHS on rank 0)"
echo "============================================================"

# srun launches one MPI rank per node.
# --mpi=pmix    works with OpenMPI >= 3 and SLURM >= 18.
# If you see "No MPI backend" errors try:  --mpi=pmi2
srun --mpi=pmix \
    python -m column_gen \
        --config          "${CONFIG}"         \
        --output          "${OUTPUT_DIR}"     \
        --max-iterations  "${MAX_ITER}"       \
        --solver-threads  "${SOLVER_THREADS}" \
        --n-workers       "${N_WORKERS}"      \
        --alpha-urllc     "${ALPHA_URLLC}"    \
        --alpha-embb      "${ALPHA_EMBB}"     \
        --log-level       "${LOG_LEVEL}"

EXIT_CODE=$?

echo "============================================================"
echo "  End time : $(date)"
echo "  Exit code: ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
