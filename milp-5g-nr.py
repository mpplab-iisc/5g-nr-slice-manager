"""
5G NR Radio Resource Management MILP
=====================================
Models the radio resource allocation problem in 5G NR featuring
network slicing as a Mixed Integer Linear Program (MILP).

Reference:
    Boutiba et al., "Optimal Radio Resource Management in 5G NR
    featuring Network Slicing", IEEE 2022.

Usage:
    # Generate a sample config then build the model:
    python milp_5g_nr.py --sample-config sample_config.json
    python milp_5g_nr.py --config sample_config.json

    # Specify a custom output directory:
    python milp_5g_nr.py --config sample_config.json --output /tmp/milp_out

    # Build and solve (CBC solver):
    python milp_5g_nr.py --config sample_config.json --solve
"""

import argparse
import os
import sys

from .src.radio_resource_milp import RadioResourceMILP
from .src.column_gen import write_sample_config, column_generation
from .src.utils import load_config, build_virtual_ues

# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="5G NR Radio Resource Management MILP Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # -- config default to current directory if not provided, but warn user to use --sample-config to generate one if they haven't provided one
    p.add_argument("--config", type=str, default=None,
                   help="Path to JSON config file.")
                   
    p.add_argument("--output", type=str, default=None,
                   help="Output directory for .lp/.mps files. "
                        "Defaults to same directory as --config.")

    p.add_argument("--sample-config", type=str, default=None, metavar="PATH",
                   help="Write a sample config to PATH and exit.")
    p.add_argument("--K", type=int, default=10,
                   help="Maximum number of UE transmissions (default: 10).")
    p.add_argument("--mu-max", type=int, default=3,
                   help="Maximum numerology index (default: 3).")
    p.add_argument("--G-guard", type=int, default=1,
                   help="Guard band in number of frequency rows (default: 1).")
    p.add_argument("--BW", type=int, default=5_000_000,
                   help="System bandwidth in Hz (default: 5 MHz).")
    p.add_argument("--time-horizon-ms", type=float, default=3.0,
                   help="Time horizon in milliseconds (default: 3 ms).")
    p.add_argument("--ue-count", type=int, default=3,
                   help="Number of UEs in the sample config (default: 3).")
    p.add_argument("--mcs", type=int, default=26,
                   help="MCS index for all UEs in the sample config (default: 26).")
    p.add_argument("--embb-mbps", type=float, default=4.0,
                   help="eMBB slice throughput requirement in Mbps (default: 4 Mbps).")
    p.add_argument("--embb-latency-ms", type=float, default=3.0,
                   help="eMBB slice latency requirement in ms (default: 3 ms).")
    p.add_argument("--urllc-mbps", type=float, default=1.0,
                   help="uRLLC slice throughput requirement in Mbps (default: 1 Mbps).")
    p.add_argument("--urllc-latency-ms", type=float, default=0.5,
                   help="uRLLC slice latency requirement in ms (default: 0.5 ms).")

    p.add_argument("--solve", action="store_true", default=False,
                   help="Solve the MILP after building (CBC solver).")
    p.add_argument("--time-limit", type=int, default=50_000,
                   help="Solver time limit in seconds.")

    p.add_argument("--cg", action="store_true", default=False,
                   help="Use Column Generation instead of the monolithic MILP. "
                        "Requires --config.  Eliminates O(N²K²) interaction "
                        "variables; LP gap typically 1-5%% vs ~285%%.")
    p.add_argument("--cg-max-iter", type=int, default=200,
                   help="Maximum CG iterations (default: 200).")
    p.add_argument("--cg-output", type=str, default=None, metavar="DIR",
                   help="Output directory for the CG integer master .lp and .mps "
                        "files. The MPS file includes OBJSENSE MAX so HiGHS and "
                        "other external solvers maximise correctly. "
                        "If omitted, no files are written.")
    
    p.add_argument("--plot", action="store_true", default=False,
                   help="Plot a solution as a resource-grid diagram. "
                        "Requires --config and --solution.")
    p.add_argument("--solution", type=str, default=None,
                   help="Path to solution .txt file (used with --plot).")
    p.add_argument("--plot-title", type=str, default=None,
                   help="Title for the resource-grid plot (used with --plot).")
    p.add_argument("--plot-output", type=str, default=None,
                   help="Output path for the resource-grid plot PNG (used with --plot). "
                        "Defaults to same directory as --solution with a '_grid.png' suffix.")

    p.add_argument("--cg-solution", type=str, default=None, metavar="PATH",
                   help="Path to .sol or .txt solution file from an external solver "
                        "(Gurobi, HiGHS) that solved the CG master MPS. "
                        "Requires --config and --plot or generates a plot automatically. "
                        "The _columns.json registry is auto-detected from the same "
                        "directory; use --cg-columns to specify it explicitly.")
    p.add_argument("--cg-columns", type=str, default=None, metavar="PATH",
                   help="Path to _columns.json registry written alongside the CG MPS. "
                        "Auto-detected when --cg-solution is used if omitted.")

    p.add_argument("--get-bwp", action="store_true", default=False,
                   help="Extract BWP allocation schedule from a solution file. "
                        "Requires --solution and --config. Outputs a JSON file with the same base name as the solution.")
    p.add_argument("--bwp-output", type=str, default=None,
                   help="Output path for the extracted BWP schedule JSON (used with --get-bwp). "
                        "Defaults to same directory as --solution with a '_bwp_schedule.json' suffix.")
    return p.parse_args()


def main():
    args = parse_args()

    # --sample-config: write and exit
    if args.sample_config:
        write_sample_config(args.sample_config, K=args.K, mu_max=args.mu_max, G_guard=args.G_guard,
            BW=args.BW, time_horizon_ms=args.time_horizon_ms, ue_count=args.ue_count, mcs=args.mcs,
            embb_mbps=args.embb_mbps, embb_latency_ms=args.embb_latency_ms,
            urllc_mbps=args.urllc_mbps, urllc_latency_ms=args.urllc_latency_ms)
        return

    if not args.config:
        print("ERROR: --config required. Use --sample-config PATH to generate one.")
        sys.exit(1)

    # Output directory defaults to same folder as config
    output_dir = args.output or os.path.dirname(os.path.abspath(args.config))

    # Load
    print(f"Loading config: {args.config}")
    config, physical_ues = load_config(args.config)

    # --plot: visualise an existing solution and exit
    if args.plot:
        if not args.solution:
            print("ERROR: --solution <path> is required with --plot.")
            sys.exit(1)
        # If plot_output is not specified, generate a path based on the solution filename
        plot_out = args.plot_output
        if not plot_out and args.output:
             # If an output directory was provided, put the file there
             base = os.path.basename(args.solution).replace(".txt", "_professional_grid.png")
             plot_out = os.path.join(args.output, base)

        out = RadioResourceMILP.plot_solution(
            config_path=args.config,
            solution_path=args.solution,
            output_path=plot_out,
            title=args.plot_title,
        )
        print(f"Plot saved → {out}")
        return

    # --cg-solution: plot CG master solution from external solver and exit
    if args.cg_solution:
        plot_out = args.plot_output
        out = RadioResourceMILP.plot_cg_solution(
            config_path=args.config,
            solution_path=args.cg_solution,
            columns_path=args.cg_columns,
            output_path=plot_out,
            title=args.plot_title,
        )
        print(f"CG plot saved → {out}")
        return

    # --get-bwp: extract BWP allocation schedule and exit
    if args.get_bwp:
        if not args.solution:
            print("ERROR: --solution <path> is required with --get-bwp.")
            sys.exit(1)
        if not args.config:
            print("ERROR: --config <path> is required with --get-bwp.")
            sys.exit(1)
        out = RadioResourceMILP.get_bwp_allocation_schedule(
            config_path=args.config,
            solution_path=args.solution,
            bwp_output=args.bwp_output
        )
         
        print(f"BWP allocation saved → {out}")
        return

    # Column Generation path
    if args.cg:
        virtual_ues, groups = build_virtual_ues(physical_ues, config)
        milp_summary = RadioResourceMILP(config, physical_ues)
        print(milp_summary.summary())
        column_generation(virtual_ues, config, groups, max_iter=args.cg_max_iter,
                          output_dir=args.cg_output)
        return

    # Build
    milp = RadioResourceMILP(config, physical_ues)
    print(milp.summary())
    print("Building MILP model …")
    milp.build()
    print(milp.model_report())



    # Write
    paths = milp.write(output_dir)
    print(f"LP  → {paths['lp']}")
    print(f"MPS → {paths['mps']}")

    # Optionally solve
    


if __name__ == "__main__":
    main()