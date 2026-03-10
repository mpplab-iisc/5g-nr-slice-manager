import highspy
import argparse
import os
import time

# CMD: 
# python highs-solver.py milp/milp_bw_5_MHz_T_3.0_ms_mu_max_3_K_2_ues_3_vues_6.mps  
# --time_limit=7200 --solution_file=solved/ 2>&1 | tee solver.log

def solve_milp(mps_file, time_limit, solution_file):
    h = highspy.Highs()
    
    # Set solver options
    h.setOptionValue('time_limit', float(time_limit))
    
    # highs-solver.py options

    # h.setOptionValue('presolve', 'off')      # Turn off the presolver
    # h.setOptionValue('mip_max_nodes', 100000) # Stop before it hits that 124k crash zone
    # h.setOptionValue('heuristics_max_nodes', 100) # Reduce heuristic complexity

    # h.setOptionValue('mip_rel_gap', 0.1) # Stop if you're within 10% of optimal
    # h.setOptionValue('mip_heuristic_effort', 0.5) # Increase effort to find a BestSol early 
    
    # Read the model
    status = h.readModel(mps_file)
    if status != highspy.HighsStatus.kOk:
        print(f"Error reading MPS file: {status}")
        return

    # Solve the model and track time manually
    print(f"Solving model: {mps_file}")
    start_time = time.time()
    h.run()
    end_time = time.time()
    solve_duration = end_time - start_time

    # --- EXTRACT DATA ---
    model_status = h.getModelStatus()
    info = h.getInfo()
    lp = h.getLp()
    variable_names = lp.col_names_
    solution = h.getSolution()

    if not solution.value_valid:
        print(f"Solver stopped ({model_status}). No feasible solution found before crash/timeout.")
        return

    variable_values = solution.col_value

    # --- SAVE RESULTS ---
    base_name = os.path.basename(mps_file).rsplit(".", 1)[0]
    if not os.path.exists(solution_file):
        os.makedirs(solution_file, exist_ok=True)
    
    out_path = os.path.join(solution_file, f"soln_{base_name}.txt")
    
    with open(out_path, "w") as f:
        f.write(f"Status: {model_status}\n")
        f.write(f"Objective: {info.objective_function_value}\n")
        f.write(f"Duration: {solve_duration:.2f}s\n\n")
        
        # Write named variables
        for name, val in zip(variable_names, variable_values):
            # if abs(val) > 1e-7:
            f.write(f"{name} {val}\n")

    print(f"Success! Results saved to {out_path}")

    print(f"Solver duration (seconds): {solve_duration:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mps_file", help="Path to the MPS file")
    parser.add_argument("--time_limit", type=int, default=100, help="Time limit in seconds")
    parser.add_argument("--solution_file", default="solved/", help="Output file or directory")
    args = parser.parse_args()

    solve_milp(args.mps_file, args.time_limit, args.solution_file)