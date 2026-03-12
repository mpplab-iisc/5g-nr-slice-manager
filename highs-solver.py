import highspy
import argparse
import os
import time
import sys

def solve_milp(mps_file, time_limit, solution_file, warm_start_file=None):
    h = highspy.Highs()
    
    # Set solver options
    h.setOptionValue('time_limit', float(time_limit))
    
    # Read the model
    status = h.readModel(mps_file)
    if status != highspy.HighsStatus.kOk:
        print(f"Error reading MPS file: {status}")
        return

    # Load warm start if provided
    if warm_start_file and os.path.exists(warm_start_file):
        print(f"Loading warm start solution from: {warm_start_file}")
        # Style 0 is the raw HiGHS format typically used for machine-readable solutions
        ws_status = h.readSolution(warm_start_file, 0)
        if ws_status != highspy.HighsStatus.kOk:
            print(f"Warning: Could not load warm start file ({ws_status}). Proceeding without it.")
        else:
            print("Warm start solution loaded successfully.")

    if h.getLp().sense_ != highspy.ObjSense.kMaximize:
        print("Warning: objective sense is not MAX — overriding to Maximize.")
        h.changeObjectiveSense(highspy.ObjSense.kMaximize)

    print(f"Solving model: {mps_file}")
    start_time = time.time()
    
    try:
        h.run()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Ctrl+C detected. Saving current best solution...")
    
    end_time = time.time()
    solve_duration = end_time - start_time

    # --- EXTRACT DATA ---
    model_status = h.getModelStatus()
    solution = h.getSolution()
    info = h.getInfo()
    lp = h.getLp()
    variable_names = lp.col_names_

    has_solution = solution.value_valid or info.mip_node_count > 0

    if not has_solution:
        print(f"Solver stopped ({model_status}). No feasible solution found.")
        return

    variable_values = solution.col_value
    best_obj = info.objective_function_value
    gap = getattr(info, 'mip_rel_gap', -1) 

    # --- SAVE RESULTS ---
    base_name = os.path.basename(mps_file).rsplit(".", 1)[0]
    os.makedirs(solution_file, exist_ok=True)
    out_path = os.path.join(solution_file, f"soln_{base_name}.txt")
    
    # Also save a RAW version specifically for highspy.readSolution() to use next time
    raw_path = os.path.join(solution_file, f"raw_{base_name}.sol")
    h.writeSolution(raw_path, 0) # Style 0 is the raw format for warm starting

    with open(out_path, "w") as f:
        f.write(f"Status: {model_status}\n")
        f.write(f"Objective: {best_obj}\n")
        if gap >= 0:
            f.write(f"MIP Gap: {gap * 100:.2f}%\n")
        f.write(f"Duration: {solve_duration:.2f}s\n\n")
        
        for name, val in zip(variable_names, variable_values):
            if abs(val) > 1e-9:
                f.write(f"{name} {val}\n")

    print(f"Success! Best results saved to {out_path}")
    print(f"Raw solution for warm start saved to {raw_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mps_file", help="Path to the MPS file")
    parser.add_argument("--time_limit", type=int, default=100, help="Time limit in seconds")
    parser.add_argument("--solution_file", default="solved/", help="Output directory")
    parser.add_argument("--warm_start", help="Optional path to a .sol file for warm starting")
    args = parser.parse_args()

    solve_milp(args.mps_file, args.time_limit, args.solution_file, args.warm_start)