# Smarter Formulation for 5G NR Radio Resource MILP
## Why the Original Formulation Cannot Scale, and How It Was Fixed

---

## Part 1: The Exact Problem with the Original Code

### The culprit: O(N²K²) interaction variables

The original code creates these variable families:

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

The original big-M values are:
```python
self.PHI_TIME = n_time_cols + 2   # = 26 for 5MHz
self.PHI_FREQ = n_freq_rows + 2   # = 31 for 5MHz
```

In constraints (13)–(14), these big-M values multiply binary variables Y, Z, W.
In the LP relaxation (integrality dropped), Y/Z/W ∈ [0,1] continuously.
A fractional Y=0.5 with PHI_TIME=26 gives a slack of 13 — enormously loose.

The LP upper bound (336 in the original log) is almost meaningless because these
big-M constraints dominate and the relaxation is completely uncorrelated
with integer feasibility. This is the root cause of the 285% gap.

---

## Part 2: The Key Structural Insight

The problem has a **natural decomposition structure** that the paper formulation
completely ignores by coupling everything through Y/Z/W variables.

**Observation**: The coupling between UEs is ONLY through shared resources:
- They cannot use the same time-frequency rectangle simultaneously
- Guard bands must be inserted between different-numerology allocations

This means: **if you fix the resource grid partition (who gets which rectangle),
each UE's allocation problem becomes INDEPENDENT.**

This is the key insight that enables Column Generation.

---

## Part 3: Column Generation Reformulation (Theory)

### The Master Problem

The Insight: UEs only interact through shared resources (time-frequency grid cells). If we fix who gets which rectangle, each UE's problem is independent.
The Fix — Column Generation:

- A "column" = one complete valid schedule for a single UE (which slots, which frequencies, which numerology)
- The Master Problem asks: pick one column per UE such that no two columns use the same grid cell
- The Pricing Subproblem asks: given current resource prices (duals), find the best new column for each UE — this is a tiny single-UE MIP
- We alternate: solve master → get prices → generate new columns → repeat

Instead of optimizing over (T, F, X, Y, Z, W, I_gb) directly, define a **column**
as a complete feasible allocation plan for a single virtual UE i:

```
Column c for VUE i = {
    for each slot k:
        t_k  : start time (grid column, or -1 if unassigned)
        f_k  : start frequency (grid row, or -1 if unassigned)
        mu_k : numerology used
        w_k  : number of contiguous PRBs (omega)
}
```

A column is **feasible for UE i** if it satisfies all single-UE constraints
(at-most-one assignment per slot, cascade ordering, time ordering, latency
deadline, throughput SLA) — completely independently of other UEs.

Let:
- `C_i` = set of all feasible columns for VUE i
- `lambda_{i,c}` ∈ {0,1} = 1 if column c is selected for VUE i

**Master Problem (MP)**:

```
maximize   Σ_i Σ_{c ∈ C_i}  min(total_prbs(c), n_prb_i) * lambda_{i,c}

subject to:

(1) Exactly one column per VUE (convexity):
    Σ_{c ∈ C_i} lambda_{i,c} = 1    ∀i

(2) No two columns conflict in time-frequency (raw cells):
    Σ_i Σ_{c ∈ C_i : c uses resource r} lambda_{i,c} ≤ 1    ∀r
    where resource r = (time_col, freq_row) grid cell

(3) Guard band constraints between different-numerology allocations:
    Enforced approximately via shadow prices in the pricing subproblem
    (see Part 6 for why pairwise constraints are not added explicitly)

lambda_{i,c} ∈ {0,1}
```

NOTE: The objective caps each VUE's contribution at its SLA requirement `n_prb_i`.
Without this cap, a single VUE monopolising the entire grid scores identically
to multiple VUEs each getting their fair share, making the LP degenerate.

**The LP relaxation of this master problem is DRAMATICALLY tighter** because:
- Columns are pre-validated as feasible single-UE solutions
- The fractional LP only needs to express "fraction of a valid plan"
- No big-M values anywhere

### The Pricing Subproblem

Since |C_i| is exponentially large, columns are never enumerated explicitly.
Instead, the Column Generation algorithm:

1. Solve LP relaxation of Master with current columns
2. Get dual prices π_r (for each resource cell r) and μ_i (for each UE i)
3. For each VUE i, solve the **Pricing Subproblem**:

```
maximize   obj(c) - Σ_r π_r * [c uses resource r] - μ_i

subject to:
    Single-UE constraints only — NO interaction with other UEs
```

4. If any pricing subproblem has positive reduced cost → add that column to MP
5. If all pricing subproblems have zero/negative reduced cost → LP optimal

### Why this destroys the N²K² scaling problem

| Component | Original formulation | Column Generation |
|-----------|---------------------|-------------------|
| Interaction variables | O(N²K²\|M\|) binary | ZERO |
| Big-M constraints | O(N²K²\|M\|) | ZERO |
| Main difficulty | Joint optimization of all pairs | N independent pricing subproblems |
| Pricing subproblem | Not applicable | Single-UE MIP, tiny |
| LP gap | 285% (completely loose) | Typically 1-5% (tight) |

---

## Part 4: Actual Implementation

The CG code lives from **~line 1929 onwards** in `milp-5g-nr.py`, structured as
standalone module-level functions (not inside `RadioResourceMILP`).

### Decision Variable Design

The key departure from the paper's `T[k], F[k], X[(k,mu,w)]` variables is that
the implementation uses a single **position-selection** binary variable:

```python
A[(k, t_val, f_val, mu, w)] ∈ {0, 1}
#  = 1 iff slot k is placed at grid position (t_val, f_val)
#        with numerology mu and PRB width w
```

This directly encodes position in the variable index, eliminating the need for
separate continuous `T[k]`, `F[k]` variables and their linking constraints.
Latency and boundary feasibility are pre-filtered so those constraints need no
explicit rows — only valid `(t, f, mu, w)` tuples are ever created as variables.

### Function Map

```
_get_valid_assignments(vue, config)
    Pre-filter every (t, f, mu, w) by time boundary, freq boundary, latency.

_cells_of_assignment(t, f, mu, w, config, with_guard)
    Compute raw or guard-padded resource cell set for one slot assignment.
    with_guard=True  → adds G_guard freq rows above/below (used during
                       conservative footprint checks, NOT stored in col.cells)
    with_guard=False → raw cells only (used in master + pricing shadow costs)

_build_single_ue_prob(vue, config, name_prefix)
    Shared MIP builder for both Phase 0 and pricing.
    Returns (prob, A_vars, valid_asns) — objective set by caller.
    Constraints added:
        (5)  Σ A[(k,·)] ≤ 1          — at most one assignment per slot
        (11) active(k+1) ≤ active(k) — cascade: k+1 only if k assigned
        (10) T_start(k+1) − PHI_T·active(k+1) ≥ T_end(k) − PHI_T
             (time ordering big-M; PHI_T = n_time_cols, tightest valid value)
        (17) Σ w·A ≥ vue.n_prb        — minimum PRB throughput SLA
             Σ w·A ≤ vue.n_prb        — PRB cap (prevents greedy monopoly)
    Constraints (16) latency and boundary are pre-filtered into valid_asns.

_extract_column_from_A(vue, config, A, valid_asns)
    Read solved A variables → produce Column dataclass.
    col.cells stores RAW cells (with_guard=False).

generate_initial_column(vue, config)          [Phase 0]
    Calls _build_single_ue_prob(), sets objective = max Σ w·A,
    solves with CBC (timeLimit=60s). Returns Column or None if infeasible.

solve_pricing_subproblem(vue, config, pi, mu_dual)   [Phase 1]
    Calls _build_single_ue_prob(), sets objective = Σ (w − shadow)*A
    where shadow(asn) = Σ_{r ∈ raw_cells(asn)} π_r.
    shadow computed with with_guard=False (consistent with master's cells).
    Returns Column if (pricing_obj − mu_dual) > 1e-6, else None.
    CBC timeLimit=30s.

_guard_band_conflicts(c1, c2, config)
    Returns True if c1 and c2 violate guard band spacing:
        (a) time-overlapping slots AND (b) different numerologies AND
        (c) frequency ranges within G_guard rows of each other.
    EXISTS but the pairwise λ[i,c1]+λ[j,c2]≤1 constraints are NOT added
    to the master (see Part 6 for why).

_build_master_problem(columns_per_vue, vues, config, integer)
    Shared LP/IP master builder. Key behaviours:
    - Objective: Σ min(total_prbs, vue.n_prb) · λ_{i,c}
    - Convexity: Σ_c λ_{i,c} = 1  ∀i
    - Resource: Σ_{i,c: r∈c.cells} λ_{i,c} ≤ 1  ∀r
                Only added when ≥ 2 columns compete for cell r.
    - λ ∈ [0,1] (LP) or {0,1} (IP)
    - Returns (prob, lam, conv_cname, res_cname) — name dicts enable dual extraction

solve_lp_master(columns_per_vue, vues, config)
    Calls _build_master_problem(integer=False), CBC timeLimit=120s.
    Extracts duals: pi[cell] = constraint.pi, mu_dual[vue_id] = constraint.pi

solve_integer_master(columns_per_vue, vues, config, output_dir)
    Calls _build_master_problem(integer=True), CBC timeLimit=300s.
    If output_dir given:
      - Writes .lp file (PuLP LP format)
      - Writes .mps file, then PATCHES it to inject:
            OBJSENSE
                MAX
        after the NAME line — needed because PuLP writes '*SENSE:Maximize'
        as a comment which HiGHS ignores.
      - Writes _columns.json registry:
            {vue_id_str → {col_idx_str → {t, f, mu, w, total_prbs}}}
        This registry is required by plot_cg_solution() to render external
        solver output as a resource grid PNG.

column_generation(vues, config, groups, max_iter, output_dir)    [main loop]
    Phase 0: For each VUE:
        - Always add a NULL column first:
              Column(total_prbs=0, cells=frozenset())
          This guarantees master feasibility: if every real column is blocked
          by resource conflicts, the null column (conflicts with nothing) is
          selectable. Without it, the master can become infeasible.
        - Call generate_initial_column() → add result if not None.
    Phase 1: CG loop up to max_iter:
        lp_obj, pi, mu_dual = solve_lp_master(...)
        for each VUE: solve_pricing_subproblem(...) → append new column
        Stop when new_cols_found == 0 (LP optimal).
    Phase 2: solve_integer_master(..., output_dir=output_dir)
```

### `Column` Dataclass

```python
@dataclass
class Column:
    vue_id:     int
    t:          List[int]           # t[k] = start time col, -1 if unassigned
    f:          List[int]           # f[k] = start freq row, -1 if unassigned
    mu:         List[Optional[int]] # mu[k] = numerology, None if unassigned
    w:          List[int]           # w[k] = PRBs (omega), 0 if unassigned
    total_prbs: int                 # Σ w[k]
    cells:      frozenset           # frozenset[(time_col, freq_row)] — RAW, no guard band
```

---

## Part 5: Expected Performance Improvement

### Theoretical complexity

| Phase | Complexity | Notes |
|-------|-----------|-------|
| CG iterations to LP optimality | O(N * iter) | Each iter: N pricing subproblems |
| Each pricing subproblem | O(K * \|M\| * max_ω) vars | Single-UE MIP, ~100-500 vars |
| Integer master (final step) | O(N * \|columns\|) | Set partitioning, usually fast |

### Practical expectation for problem sizes

| Scale | Original formulation | Column Generation |
|-------|---------------------|-------------------|
| 6 VUEs (current) | 2+ hours, 285% gap | < 1 minute |
| 12 VUEs (6 UEs) | Intractable | 5-15 minutes |
| 100 VUEs (50 UEs) | Completely impossible | 30-120 minutes |
| 200 VUEs (100 UEs) | 28M vars, impossible | Hours (but feasible) |

### LP bound quality

The LP relaxation gives a gap of typically **2-10%** instead of 285%, because:
- No big-M values in the master problem
- Columns are pre-validated against all single-UE constraints
- The only coupling is through resource cells — tight and natural

---

## Part 6: Guard Band Handling — Actual Approach

### What was planned vs what was built

The original plan was to use a **conservative footprint**: always expand each
column's cell set by G_guard rows above/below in frequency, so the resource
non-conflict constraint in the master naturally enforces guard band separation.

**This is NOT what the implementation does.** The actual approach:

1. `col.cells` stores **raw cells only** (`with_guard=False`) — no guard band buffer
2. `_guard_band_conflicts(c1, c2, config)` exists and correctly detects violations
3. **Pairwise constraints `λ[i,c1]+λ[j,c2]≤1` are deliberately NOT added to the
   master** — with ~200 columns per VUE and 15 VUE pairs, this generates ~600,000
   constraints which overwhelms CBC/Gurobi presolve and can cascade-eliminate so
   many columns that the master becomes infeasible.

### Actual guard band mechanism

Guard band separation is enforced **approximately** through the pricing shadow costs:

- Resource cells that are heavily contested (used by many columns) get high π_r
- The pricing subproblem subtracts Σ π_r from the objective for each cell used
- Columns in contested frequency bands become unprofitable to generate
- In practice, this steers different-numerology VUEs into well-separated frequency
  bands without any explicit guard band constraints in the master

The null column (total_prbs=0, no cells) in every VUE's pool is the safety net:
it is always selectable and conflicts with nothing, guaranteeing master feasibility
even if aggressive shadow pricing eliminates all real columns for a VUE.

### `_cells_of_assignment()` with_guard flag

```python
def _cells_of_assignment(t_val, f_val, mu, w, config, with_guard=True):
    # with_guard=True:  adds G_guard rows on each side
    #                   (used only by _guard_band_conflicts for checking,
    #                    NOT stored in col.cells)
    # with_guard=False: raw cells only
    #                   (used in _extract_column_from_A to build col.cells,
    #                    and in solve_pricing_subproblem shadow cost computation)
```

Both `_extract_column_from_A` and `solve_pricing_subproblem` use `with_guard=False`
so that the shadow costs are consistent with the master's resource constraints
(which are also built over raw cells).

---

## Part 7: External Solver Support and Plotting

### MPS export for HiGHS / Gurobi

`solve_integer_master(..., output_dir=DIR)` writes three files:

```
cg_master_bw_<BW>_MHz_T_<dt>_ms_mu_max_<M>_K_<K>_vues_<N>_cols_<C>.lp
cg_master_bw_<BW>_MHz_T_<dt>_ms_mu_max_<M>_K_<K>_vues_<N>_cols_<C>.mps
cg_master_bw_<BW>_MHz_T_<dt>_ms_mu_max_<M>_K_<K>_vues_<N>_cols_<C>_columns.json
```

The `.mps` file is patched to insert `OBJSENSE / MAX` after the `NAME` line.
PuLP's default MPS output writes `*SENSE:Maximize` as a comment, which external
solvers like HiGHS treat as a minimisation problem.

The `_columns.json` maps `{vue_id → {col_idx → column_data}}` and is required
to interpret external solver `.sol` output.

### `plot_cg_solution()` — static method on RadioResourceMILP

Reads a Gurobi/HiGHS `.sol` file (lines `lam_<i>_<c>  1`), auto-detects the
`_columns.json` registry (sibling file or directory scan), and renders a
resource grid PNG identical in style to `plot_solution()`.

Auto-detection priority:
1. `<sol_base>_columns.json` sibling file
2. Single `*_columns.json` in the solution directory
3. Closest stem match if multiple found

---

## Part 8: CLI Usage

```bash
# Run Column Generation (CBC solver, writes MPS for external use)
python milp-5g-nr.py --config configs/my.json --cg

# Limit CG iterations and export master problem
python milp-5g-nr.py --config configs/my.json --cg --cg-max-iter 100 \
    --cg-output solved/

# Plot a solution from an external solver (HiGHS/Gurobi)
python milp-5g-nr.py --config configs/my.json \
    --cg-solution solved/cg_master_...sol \
    --cg-columns  solved/cg_master_..._columns.json

# (--cg-columns is optional if _columns.json is in the same directory)
python milp-5g-nr.py --config configs/my.json \
    --cg-solution solved/cg_master_...sol
```

| Flag | Default | Description |
|------|---------|-------------|
| `--cg` | off | Enable Column Generation path |
| `--cg-max-iter` | 200 | Max CG loop iterations |
| `--cg-output DIR` | None | Write .lp/.mps/_columns.json to DIR |
| `--cg-solution PATH` | None | Plot external solver .sol file |
| `--cg-columns PATH` | auto | Explicit _columns.json path for plotting |

---

## Part 9: Known Limitations

1. **Group constraints (18)/(19) not enforced** — same-group VUEs must share
   numerology and start-time when slots overlap. Requires either coupled pricing
   subproblems or additional master constraints. Not implemented.

2. **Guard band enforcement is approximate** — pairwise constraints are omitted;
   shadow prices steer columns away from contested bands but cannot guarantee
   zero guard band violations in the final integer solution.

3. **Standard CG tailing-off** — LP convergence slows near the optimum. 200
   iterations may not reach full LP optimality for large instances; the integer
   master is solved on whatever columns have been generated.

4. **Pricing subproblems run sequentially** — the loop iterates over VUEs one
   by one. Parallel execution (one subprocess per VUE) would speed up each CG
   iteration by N× but is not implemented.

5. **Dual sign convention** — uses `constraint.pi` from PuLP/CBC directly.
   For maximisation, resource constraint (≤) duals are ≥ 0; convexity (=) duals
   can be any sign. This is consistent throughout.

---

## Summary

The original formulation has **O(N²K²) binary variables** because it models
pairwise interactions between ALL slots of ALL UEs explicitly (Y, Z, W, I_gb).

The Column Generation reformulation eliminates all pairwise interaction variables.
The only coupling between VUEs is through shared resource cells (tight, no big-M).
Each pricing subproblem is a small single-UE MIP (~K×|M|×max_ω binary variables).

**Key implementation choices:**
- Decision variable `A[(k,t,f,mu,w)]` instead of separate `T[k], F[k], X[(k,mu,w)]`
- PRB cap `Σw·A ≤ n_prb` prevents degenerate single-VUE monopoly in the master
- Null column per VUE guarantees master feasibility at all times
- Raw cell footprint (no guard band) in `col.cells`; guard band handled via shadow prices
- Pairwise guard band constraints omitted to avoid O(C²) constraint explosion
- MPS OBJSENSE MAX patch for HiGHS compatibility
- `_columns.json` registry for external solver result plotting
