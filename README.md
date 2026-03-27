- [5g-nr-slice-manager](#5g-nr-slice-manager)
  - [Problem](#problem)
  - [Install Highs Solver Packages](#install-highs-solver-packages)
  - [Create sample config files](#create-sample-config-files)
  - [Create MILPs](#create-milps)
  - [Solve MILPs](#solve-milps)
  - [Plot MILP Solution](#plot-milp-solution)
  - [Get BWP Allocation](#get-bwp-allocation)
- [Solution using Column-generation](#solution-using-column-generation)
  - [Column-gen solution](#column-gen-solution)
  - [Plot](#plot)
- [Reproducing Paper Results](#reproducing-paper-results)
  - [Create sample config files](#create-sample-config-files-1)

# 5g-nr-slice-manager
This project is a Python implementation of the MILP (Mixed Integer Linear Program) formulation described in Boutiba et al., "Optimal Radio Resource Management in 5G NR featuring Network Slicing," IEEE 2022.

## Problem 
The paper tackles radio resource allocation in 5G NR networks where multiple User Equipments (UEs) — each potentially belonging to multiple network slices (eMBB and uRLLC) — must be efficiently scheduled on a 2D time-frequency resource grid. The key challenges are:

- Mixed numerology: 5G NR allows different subcarrier spacings (µ = 0–4). Higher µ means shorter slot durations but wider frequency bands. Mixing numerologies in the same band causes Inter-Numerology Interference (INI), requiring guard bands.
- Network slicing: One physical UE may serve both an eMBB slice (high throughput, relaxed latency) and a uRLLC slice (low throughput, strict latency), each treated as a separate Virtual UE with SLA constraints.
- NP-hardness: The paper formally proves the problem is NP-hard by reduction from the Knapsack problem.
## Install Highs Solver Packages
```
pip install highspy --break-system-packages
```

## Create sample config files

BW: 20 MHz, Time Horizon: 3 ms, Number of UEs: 3,
slices per UE: 2, slice_SLA (Mbps,ms): (9,10),(0.2,1),
MCS: 16, K: 3

```
python milp-5g-nr.py --sample-config=configs --BW=20000000 --time-horizon-ms=3.0 --mcs=16 --ue-count=3 --embb-mbps=9 --embb-latency-ms=10 --urllc-mbps=0.2 --urllc-latency-ms=1 --K=3
```

## Create MILPs

```
python milp-5g-nr.py --config=configs/cfg-bw-20.0M-time-ms-3.0-mcs-16-K-3-ue-3-embb-thr-9.0-embb-lat-10.0-urllc-thr-0.2-urllc-lat-1.0.json --output=milp/
```

## Solve MILPs

```
python highs-solver.py milp/milp_bw_20_MHz_T_3.0_ms_mu_max_3_K_3_ues_3_vues_6.mps  --time_limit=1200 --solution_file=solved/ 
```

## Plot MILP Solution

```
python milp-5g-nr.py --config=configs/cfg-bw-20.0M-time-ms-3.0-mcs-16-K-3-ue-3-embb-thr-9.0-embb-lat-10.0-urllc-thr-0.2-urllc-lat-1.0.json --solution=solved/soln_milp_bw_20_MHz_T_3.0_ms_mu_max_3_K_3_ues_3_vues_6.txt  --plot --plot-output=figs/
```

## Get BWP Allocation

```
python milp-5g-nr.py --config=configs/cfg-bw-20.0M-time-ms-3.0-mcs-16-K-3-ue-3-embb-thr-9.0-embb-lat-10.0-urllc-thr-0.2-urllc-lat-1.0.json --solution=solved/soln_milp_bw_20_MHz_T_3.0_ms_mu_max_3_K_3_ues_3_vues_6.txt   --get-bwp --bwp-output=bwp/
```

# Solution using Column-generation

## Column-gen solution
```
python -m column_gen --config configs/cfg-bw-5.0M-time-ms-3.0-mcs-26-K-3-ue-3-embb-thr-4.0-embb-lat-3.0-urllc-thr-1.0-urllc-lat-0.5.json --output approx_sol/ 2>&1 | tee logs/logs_40M_column_gen.log
```
## Plot
```
python milp-5g-nr.py --config configs/cfg-bw-5.0M-time-ms-3.0-mcs-26-K-3-ue-3-embb-thr-4.0-embb-lat-3.0-urllc-thr-1.0-urllc-lat-0.5.json --solution approx_sol/cfg-bw-5.0M-time-ms-3.0-mcs-26-K-3-ue-3-embb-thr-4.0-embb-lat-3.0-urllc-thr-1.0-urllc-lat-0.5.txt --plot --plot-output figs/
```
# Reproducing Paper Results
## Create sample config files

BW: 5 MHz, Time Horizon: 3 ms, Number of UEs: 3,
slices per UE: 2, slice_SLA (Mbps,ms): (4,3),(1,0.5),
MCS: 26, K: 3
```
python milp-5g-nr.py --sample-config=configs/ --BW=5000000 --time-horizon-ms=3.0 --mcs=26 --ue-count=3  --embb-mbps=4 --embb-latency-ms=3 --urllc-mbps=1.0 --urllc-latency-ms=0.5 --K=3
```
Create MILP
```
python milp-5g-nr.py --config=configs/cfg-bw-5.0M-time-ms-3.0-mcs-26-K-3-ue-3-embb-thr-4.0-embb-lat-3.0-urllc-thr-1.0-urllc-lat-0.5.json --output=milp/
```

Solve MILP
```
python highs-solver.py milp/milp_bw_5_MHz_T_3.0_ms_mu_max_3_K_3_ues_3_vues_6.mps --time_limit=7200 --solution_file=solved/
```

Plot solution
```
python milp-5g-nr.py --config=configs/cfg-bw-5.0M-time-ms-3.0-mcs-26-K-10-ue-3-embb-thr-4.0-embb-lat-3.0-urllc-thr-1.0-urllc-lat-0.5.json --solution=solved/soln_milp_bw_5_MHz_T_3.0_ms_mu_max_3_K_3_ues_3_vues_6.txt --plot --plot-output=figs/
```