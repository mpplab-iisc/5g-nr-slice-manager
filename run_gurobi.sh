#!/bin/bash

# 1. Setup environment (Updated to your current /opt path)
export GUROBI_HOME="/opt/gurobi1101/linux64"
export PATH="${GUROBI_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${GUROBI_HOME}/lib:${LD_LIBRARY_PATH}"
export GRB_LICENSE_FILE="/home/prarabdha-srivastava/Desktop/gurobi.lic"

# 2. Move to your work directory
# Replace with the path where your .mps file is located
cd /home/prarabdha-srivastava/Desktop/5g-nr-slice-manager

# 3. Run Gurobi
# We keep the Threads=2 to match your previous setup
gurobi_cl MIPGap=0.01 TimeLimit=72000 Threads=2 \
  LogFile=milp_util/gurobi.log \
  ResultFile=milp_util/solution.sol \
  milp_util/cg_master_bw_5_MHz_T_3.0_ms_mu_max_3_K_3_vues_6_cols_1050.mps
