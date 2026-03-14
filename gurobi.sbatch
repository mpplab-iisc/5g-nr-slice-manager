#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:10:00
#SBATCH --job-name=gurobi_milp
#SBATCH --output=gurobi_%j.log

# Load gurobi environment
export GUROBI_HOME=~/work/gurobi1301/linux64
export PATH=$GUROBI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$GUROBI_HOME/lib:$LD_LIBRARY_PATH
export GRB_LICENSE_FILE=~/gurobi.lic

cd /home/prarabdha_iisc_gmail_com/work

gurobi_cl MIPGap=0.01 TimeLimit=72000 Threads=2 \
  ResultFile=solution.sol \
  cg_master_bw_5_MHz_T_3.0_ms_mu_max_3_K_3_vues_6_cols_120.mps