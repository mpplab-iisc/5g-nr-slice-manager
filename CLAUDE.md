# Smarter Formulation for 5G NR Radio Resource MILP
## Why Your Current Formulation Cannot Scale, and How to Fix It

---

## Part 1: The Exact Problem with Your Current Code

### The culprit: O(N²K²) interaction variables

Your code creates these variable families:

```python
self.Y[(i,j)][(k,l)]      # binary: do slots k,l of UEs i,j overlap in time?
self.Z[(i,j)][(k,l)]      # binary: time direction selector
self.W[(i,j)][(k,l)]      # binary: frequency direction selector
self.I_gb[(i,j)][(k,l)]   # binary: guard band indicator
```

For **every ordered pair of distinct virtual UEs (i,j)** and **every pair of slots (k,l)**. The count is:

| Scale | Virtual UEs | Y+Z+W vars | I_gb vars | Total vars | Constraints |
|-------|-------------|------------|-----------|------------|-------------|
| Current (3 UEs) | 6 | 9,000 | 12,000 | ~24,000 | ~90,000 |
| 50 UEs | 100 | 2,970,000 | 3,960,000 | ~7.1M | ~30M |
| 100 UEs | 200 | 11,940,000 | 15,920,000 | ~28M | ~120M |

**This is O(N²K²|M|) — quadratic in both UEs and slots simultaneously.**

### Why the LP relaxation is so loose (285% gap)

Your big-M values are:
```python
self.PHI_TIME = n_time_cols + 2   # = 26 for 5MHz
self.PHI_FREQ = n_freq_rows + 2   # = 31 for 5MHz
```

In constraints (13)–(14), these big-M values multiply binary variables Y, Z, W.
In the LP relaxation (integrality dropped), Y/Z/W ∈ [0,1] continuously.
A fractional Y=0.5 with PHI_TIME=26 gives a slack of 13 — enormously loose.

The LP upper bound (336 in your log) is almost meaningless because these
big-M constraints dominate and the relaxation is completely uncorrelated
with integer feasibility. This is the root cause of the 285% gap.

---

## Part 2: The Key Structural Insight

Your problem has a **natural decomposition structure** that the paper formulation
completely ignores by coupling everything through Y/Z/W variables.

**Observation**: The coupling between UEs is ONLY through shared resources:
- They cannot use the same time-frequency rectangle simultaneously
- Guard bands must be inserted between different-numerology allocations

This means: **if you fix the resource grid partition (who gets which rectangle),
each UE's allocation problem becomes INDEPENDENT.**

This is the key insight that enables Column Generation.

---

## Part 3: Column Generation Reformulation

### The Master Problem

Instead of optimizing over (T, F, X, Y, Z, W, I_gb) directly, define a **column**
as a complete feasible allocation plan for a single virtual UE i:

```
Column c for VUE i = {
    for each slot k:
        t_k  : start time (grid column, or -1 if unassigned)
        f_k  : start frequency (grid row, or -1 if unassigned)
        mu_k : numerology used
        w_k  : number of contiguous PRBs
}
```

A column is **feasible for UE i** if it satisfies constraints (5)–(11), (16), (17)
— all the single-UE constraints — completely independently.

Let:
- `C_i` = set of all feasible columns for VUE i
- `lambda_{i,c}` ∈ {0,1} = 1 if column c is selected for VUE i

**Master Problem (MP)**:

```
maximize   Σ_i Σ_{c ∈ C_i}  obj(c) * lambda_{i,c}

subject to:

(1) Exactly one column per VUE:
    Σ_{c ∈ C_i} lambda_{i,c} = 1    ∀i

(2) No two columns conflict in time-frequency:
    Σ_i Σ_{c ∈ C_i : c uses resource r} lambda_{i,c} ≤ 1    ∀r
    
    where resource r = (time_col, freq_row) grid cell
    
(3) Guard band constraints between different-numerology allocations:
    (handled via resource packing with guard band rows)

lambda_{i,c} ∈ {0,1}
```

**The LP relaxation of this master problem is DRAMATICALLY tighter** because:
- Columns are pre-validated as feasible single-UE solutions
- The fractional LP only needs to express "fraction of a valid plan" 
- No big-M values anywhere

### The Pricing Subproblem

Since |C_i| is exponentially large, you never enumerate all columns.
Instead, the Column Generation algorithm:

1. Solve LP relaxation of Master with current columns
2. Get dual prices π_r (for each resource cell r) and μ_i (for each UE i)
3. For each VUE i, solve the **Pricing Subproblem**:

```
maximize   obj(c) - Σ_r π_r * [c uses resource r] - μ_i

subject to:
    Single-UE constraints (5)–(11), (16), (17) only
    (NO interaction with other UEs)
```

4. If any pricing subproblem has positive reduced cost → add that column to MP
5. If all pricing subproblems have zero/negative reduced cost → LP optimal

**The pricing subproblem for a single UE is a small MIP with K*|M|*max_omega
binary variables** — typically solvable in milliseconds even for large K.

### Why this destroys the N²K² scaling problem

| Component | Original formulation | Column Generation |
|-----------|---------------------|-------------------|
| Interaction variables | O(N²K²|M|) binary | ZERO |
| Big-M constraints | O(N²K²|M|) | ZERO |
| Main difficulty | Joint optimization of all pairs | N independent pricing subproblems |
| Pricing subproblem | Not applicable | Single-UE MIP, tiny |
| LP gap | 285% (completely loose) | Typically 1-5% (tight) |

---

## Part 4: Concrete Implementation Plan

### Step 1: Generate initial columns (Phase 0)

Before CG starts, you need an initial feasible set of columns.
The easiest approach: solve each UE's problem independently ignoring
interference → gives N feasible (but possibly conflicting) columns.

```python
def generate_initial_columns(vue, config):
    """
    Solve single-UE MILP: maximize PRBs for VUE i,
    ignoring all other UEs. Returns one feasible column.
    Constraints: (5),(6),(7),(8),(9),(10),(11),(16),(17) only.
    Solvable in < 1 second per UE.
    """
    prob = pulp.LpProblem(f"init_col_vue{vue.virtual_id}", pulp.LpMaximize)
    # ... add single-UE variables and constraints only ...
    # ... solve with CBC (trivially fast) ...
    return extract_column(prob)
```

### Step 2: Resource grid representation

Represent the time-frequency grid as a set of "resource cells":

```python
# Each resource cell = (time_col, freq_row)
# A column "uses" a set of cells
def cells_used_by_column(col, config):
    cells = set()
    for k in range(config.K):
        if col.t[k] == -1:
            continue
        mu = col.mu[k]
        duration = E(mu, config.mu_max)
        width = col.w[k] * G_rows(mu)
        for dt in range(duration):
            for df in range(width):
                cells.add((col.t[k] + dt, col.f[k] + df))
    return cells
```

### Step 3: LP Master Problem

```python
def solve_lp_master(columns_per_vue, resource_cells_used):
    prob = pulp.LpProblem("CG_Master_LP", pulp.LpMaximize)
    
    # lambda[i][c] ∈ [0,1] for LP relaxation
    lam = {(i,c): pulp.LpVariable(f"lam_{i}_{c}", 0, 1) 
           for i, cols in columns_per_vue.items() 
           for c in range(len(cols))}
    
    # Objective: maximize total PRBs
    prob += pulp.lpSum(cols[c].total_prbs * lam[(i,c)] 
                       for i, cols in columns_per_vue.items()
                       for c in range(len(cols)))
    
    # (1) Convexity: exactly one column per VUE
    for i in columns_per_vue:
        prob += pulp.lpSum(lam[(i,c)] for c in range(len(columns_per_vue[i]))) == 1
    
    # (2) Resource non-conflict: each (t,f) cell used at most once
    for cell in all_resource_cells:
        usage = pulp.lpSum(
            lam[(i,c)] 
            for i, cols in columns_per_vue.items()
            for c, col in enumerate(cols)
            if cell in resource_cells_used[(i,c)]
        )
        prob += usage <= 1
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # Extract dual prices
    pi = {cell: constraint.pi for cell, constraint in resource_constraints.items()}
    mu = {i: convexity_constraint[i].pi for i in columns_per_vue}
    
    return prob.objective.value(), pi, mu
```

### Step 4: Pricing Subproblem (one per VUE, run in parallel)

```python
def solve_pricing_subproblem(vue, config, pi, mu_i):
    """
    Find column for VUE i with maximum reduced cost.
    This is a SINGLE-UE MIP — small and fast.
    """
    prob = pulp.LpProblem(f"pricing_{vue.virtual_id}", pulp.LpMaximize)
    
    # Same variables as original but ONLY for this one UE
    T = {k: pulp.LpVariable(f"T_{k}", -1, config.n_time_cols-1, cat='Integer') 
         for k in range(config.K)}
    F = {k: pulp.LpVariable(f"F_{k}", -1, config.n_freq_rows-1, cat='Integer')
         for k in range(config.K)}
    X = {(k,mu,w): pulp.LpVariable(f"X_{k}_{mu}_{w}", cat='Binary')
         for k in range(config.K)
         for mu in config.numerologies
         for w in range(1, max_omega(mu, config.n_freq_rows)+1)}
    
    # Objective: original PRB count MINUS shadow prices of used resources
    # The shadow price term makes columns avoid already-contested resources
    resource_cost = pulp.lpSum(
        pi.get((t+dt, f+df), 0) * X[(k,mu,w)]
        for k in range(config.K)
        for mu in config.numerologies
        for w in range(1, max_omega(mu, config.n_freq_rows)+1)
        # ... expand to all cells used by this (k,mu,w,T,F) assignment ...
    )
    
    prob += pulp.lpSum(w * X[(k,mu,w)] ...) - resource_cost - mu_i
    
    # Add ONLY single-UE constraints: (5),(6),(7),(8),(9),(10),(11),(16),(17)
    add_single_ue_constraints(prob, vue, T, F, X, config)
    
    prob.solve(pulp.PULP_CBC_CMD(timeLimit=30, msg=0))
    
    reduced_cost = pulp.value(prob.objective)
    if reduced_cost > 1e-6:
        return extract_column(T, F, X, config)  # new column to add
    return None  # no improving column
```

### Step 5: Main CG Loop

```python
def column_generation(vues, config, max_iter=200):
    # Phase 0: initialize with single-UE solutions
    columns = {vue.virtual_id: [generate_initial_column(vue, config)] 
               for vue in vues}
    
    for iteration in range(max_iter):
        # Solve LP master
        obj, pi, mu = solve_lp_master(columns)
        print(f"Iter {iteration}: LP obj = {obj:.4f}")
        
        # Pricing: find new columns in parallel
        new_cols_found = 0
        for vue in vues:
            new_col = solve_pricing_subproblem(vue, config, pi, mu[vue.virtual_id])
            if new_col is not None:
                columns[vue.virtual_id].append(new_col)
                new_cols_found += 1
        
        if new_cols_found == 0:
            print(f"LP optimal at iteration {iteration}")
            break
    
    # Phase 2: solve integer master (restrict lambda to binary)
    return solve_integer_master(columns)
```

---

## Part 5: Expected Performance Improvement

### Theoretical complexity

| Phase | Complexity | Notes |
|-------|-----------|-------|
| CG iterations to LP optimality | O(N * iter) | Each iter: N pricing subproblems |
| Each pricing subproblem | O(K * |M| * max_ω) vars | Single-UE MIP, ~100-500 vars |
| Integer master (final step) | O(N * |columns|) | Set partitioning, usually fast |

### Practical expectation for your problem sizes

| Scale | Original formulation | Column Generation |
|-------|---------------------|-------------------|
| 6 VUEs (current) | 2+ hours, 285% gap | < 1 minute |
| 12 VUEs (6 UEs) | Intractable | 5-15 minutes |
| 100 VUEs (50 UEs) | Completely impossible | 30-120 minutes |
| 200 VUEs (100 UEs) | 28M vars, impossible | Hours (but feasible) |

### LP bound quality

The LP relaxation of the column generation master problem will give a gap
of typically **2-10%** instead of 285%, because:
- No big-M values in the master problem
- Columns are pre-validated against all single-UE constraints
- The only coupling is through resource cells — tight and natural

---

## Part 6: Guard Band Handling in Column Generation

The guard band between different-numerology allocations needs special treatment.

**Approach**: Expand each column's cell footprint to include guard band rows.

```python
def cells_used_with_guard_band(col, other_cols, config):
    """
    When checking if column c conflicts with column c',
    add G_guard rows above/below if numerologies differ.
    
    In practice: represent each column's footprint as
    (time cells) × (freq cells + potential guard bands)
    and let the resource conflict constraint handle it.
    """
    # Conservative approach: always include guard band in footprint
    # This is slightly suboptimal but correct and simple
    cells = set()
    for k in range(config.K):
        if col.t[k] == -1:
            continue
        mu = col.mu[k]
        duration = E(mu, config.mu_max)
        width = col.w[k] * G_rows(mu)
        for dt in range(duration):
            # Add G_guard rows on each side as buffer
            for df in range(-config.G_guard, width + config.G_guard):
                f = col.f[k] + df
                if 0 <= f < config.n_freq_rows:
                    cells.add((col.t[k] + dt, f))
    return cells
```

---

## Part 7: What to Implement First (Practical Roadmap)

### Week 1: Validate the pricing subproblem
Implement and test `solve_pricing_subproblem()` for a single VUE.
Confirm it produces feasible single-UE schedules matching constraints (5)–(17).
This is the hardest part to get right.

### Week 2: Implement the LP master + dual extraction
PuLP gives dual prices via `constraint.pi` after solving the LP relaxation.
Implement `solve_lp_master()` and verify the duals make sense
(resource cells that are heavily contested should have high π).

### Week 3: Main CG loop
Connect pricing → master → pricing. Run on your 6-VUE instance.
You should see LP bound converge in 10-50 iterations.
Compare final LP bound to Gurobi's LP relaxation — they should match.

### Week 4: Integer master (branch-and-price)
The integer master is a Set Partitioning Problem:
```
min/max  Σ_{i,c} obj(c) * lambda_{i,c}
s.t.     Σ_c lambda_{i,c} = 1  ∀i           (one column per VUE)
         Σ_{i,c: c uses r} lambda_{i,c} ≤ 1  ∀r   (resource packing)
         lambda_{i,c} ∈ {0,1}
```
This is a Set Partitioning Problem — well-studied, SCIP handles it well
with the columns you've generated. Gurobi handles it even better.

### Use OR-Tools for the integer master
Google's CP-SAT is particularly well-suited for set partitioning:
```python
from ortools.sat.python import cp_model
# CP-SAT handles binary set partitioning very efficiently
# Has native parallel solving, no license needed
```


---

## Summary: The One Thing That Matters

Your current formulation has **O(N²K²) binary variables** because it models
pairwise interactions between ALL slots of ALL UEs explicitly.

The fix is to **reformulate to avoid pairwise interaction variables entirely**
by working with complete single-UE allocation plans (columns) as the decision
unit. The only coupling then is through shared resource cells — and that coupling
is tight (no big-M values needed).

For 50-100 UEs this is the difference between impossible and tractable.
For your current 3-UE instance, it is the difference between 2 hours and 1 minute.
The formulation is fundamentally broken for this problem size.
work on  Column Generation code based on the  existing codebase.
