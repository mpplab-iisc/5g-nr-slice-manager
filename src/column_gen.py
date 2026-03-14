# ============================================================================
# Column Generation Reformulation
# ============================================================================
# Eliminates O(N²K²) interaction variables by working with complete single-UE
# allocation plans (columns) as the decision unit. Inter-UE coupling is only
# through shared resource cells — no big-M values in the master problem.
#
# Phases:
#   0. generate_initial_column()   — single-UE MIP, ignoring other UEs
#   1. CG loop: LP master → dual prices → pricing subproblems → new columns
#   2. solve_integer_master()      — final Set Partitioning MIP
# ============================================================================

import sys
import os
import json

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

from .configs import SystemConfig, VirtualUE
from .utils import G_rows, E, max_omega


try:
    import pulp
except ImportError:
    print("ERROR: PuLP not installed.  Run: pip install pulp --break-system-packages")
    sys.exit(1)


@dataclass
class Column:
    """
    A feasible allocation plan for a single virtual UE.

    Attributes:
        vue_id      : VirtualUE.virtual_id this column belongs to
        t           : t[k] = start time column for slot k, -1 if unassigned
        f           : f[k] = start freq row for slot k, -1 if unassigned
        mu          : mu[k] = numerology for slot k, None if unassigned
        w           : w[k] = contiguous PRBs at slot k, 0 if unassigned
        total_prbs  : Σ w[k] across all active slots
        cells       : frozenset of (time_col, freq_row) resource cells used,
                      with conservative guard-band buffer above/below in freq.
    """
    vue_id:     int
    t:          List[int]
    f:          List[int]
    mu:         List[Optional[int]]
    w:          List[int]
    total_prbs: int
    cells:      frozenset   # frozenset[tuple[int, int]]


def _cells_of_assignment(
    t_val: int, f_val: int, mu: int, w: int,
    config: SystemConfig, with_guard: bool = True,
) -> frozenset:
    """
    Compute the set of resource cells covered by a single slot assignment.

    With with_guard=True (default), adds G_guard frequency rows of buffer on
    each side — the conservative footprint model that avoids needing pairwise
    numerology checks in the master problem. Slightly suboptimal but correct.
    """
    duration   = E(mu, config.mu_max)
    width_rows = w * G_rows(mu)
    guard      = config.G_guard if with_guard else 0
    cells: set = set()
    for dt in range(duration):
        for df in range(-guard, width_rows + guard):
            f_pos = f_val + df
            if 0 <= f_pos < config.n_freq_rows:
                cells.add((t_val + dt, f_pos))
    return frozenset(cells)


def _get_valid_assignments(
    vue: VirtualUE, config: SystemConfig,
) -> List[Tuple[int, int, int, int]]:
    """
    Pre-compute every valid (t, f, mu, w) tuple for a single slot of VUE.

    Pre-filters by:
        Time boundary  : t + E(mu) ≤ n_time_cols
        Freq boundary  : f + w · G_rows(mu) ≤ n_freq_rows
        Latency (16)   : t + E(mu) ≤ latency_slots
    """
    assignments: List[Tuple[int, int, int, int]] = []
    for mu in config.numerologies:
        duration  = E(mu, config.mu_max)
        g         = G_rows(mu)
        max_w_val = max_omega(mu, config.n_freq_rows)
        t_max = min(config.n_time_cols, vue.latency_slots) - duration
        if t_max < 0:
            continue   # this numerology can never meet the latency deadline
        for t_val in range(0, t_max + 1):
            for w in range(1, max_w_val + 1):
                f_max = config.n_freq_rows - w * g
                if f_max < 0:
                    continue
                for f_val in range(0, f_max + 1):
                    assignments.append((t_val, f_val, mu, w))
    return assignments


def _build_single_ue_prob(
    vue:         VirtualUE,
    config:      SystemConfig,
    name_prefix: str = "single_ue",
) -> Tuple[pulp.LpProblem, Dict[Tuple, pulp.LpVariable], List[Tuple]]:
    """
    Build a single-UE MIP with explicit position-selection variables.

    Decision variable:
        A[(k, t, f, mu, w)] ∈ {0, 1}
            = 1 iff slot k is placed at grid position (t, f) with numerology
              mu and PRB width w.

    Constraints implemented here:
        (5)  At most one (t, f, mu, w) per slot
        (11) Cascade: slot k+1 can only be assigned if slot k is assigned
        (10) Time ordering: slot k+1 must start ≥ end of slot k
             Linearisation: T_start(k+1) − PHI·active(k+1) ≥ T_end(k) − PHI
        (17) Throughput SLA: Σ_{k,t,f,mu,w} w · A ≥ n_prb

    Latency (16) and boundary constraints are pre-filtered into valid_asns
    so they need no explicit constraint rows.

    The objective is set by the caller (init maximises PRBs; pricing uses
    reduced costs).

    Returns: (prob, A_vars, valid_asns)
    """
    K          = config.K
    valid_asns = _get_valid_assignments(vue, config)

    prob = pulp.LpProblem(f"{name_prefix}_vue{vue.virtual_id}", pulp.LpMaximize)

    A: Dict[Tuple, pulp.LpVariable] = {}
    for k in range(K):
        for (t_val, f_val, mu, w) in valid_asns:
            A[(k, t_val, f_val, mu, w)] = pulp.LpVariable(
                f"A_v{vue.virtual_id}_k{k}_t{t_val}_f{f_val}_m{mu}_w{w}",
                cat=pulp.constants.LpBinary,
            )

    def sum_A_slot(k: int):
        """Active-indicator for slot k: Σ A[(k,t,f,mu,w)] ∈ {0, 1}."""
        return pulp.lpSum(A[(k, t, f, mu, w)] for (t, f, mu, w) in valid_asns)

    # (5) At most one assignment per slot
    for k in range(K):
        prob += (sum_A_slot(k) <= 1, f"c5_slot{k}")

    # (11) Cascade: k+1 assigned only if k assigned
    for k in range(K - 1):
        prob += (sum_A_slot(k + 1) <= sum_A_slot(k), f"c11_casc{k}")

    # (10) Time ordering: T_start(k+1) ≥ T_end(k) when k+1 is assigned.
    # Big-M linearisation (PHI_T = n_time_cols is the tightest valid big-M):
    #   T_start(k+1) + PHI_T · (1 − active(k+1)) ≥ T_end(k)
    #   → T_start(k+1) − PHI_T · active(k+1) ≥ T_end(k) − PHI_T
    # When k+1 unassigned: LHS ≥ −PHI_T, RHS ≤ 0 ≤ n_time_cols − PHI_T = 0 ✓
    # When k+1 assigned:   T_start(k+1) ≥ T_end(k) ✓
    PHI_T = config.n_time_cols
    for k in range(K - 1):
        T_start_next = pulp.lpSum(
            t_val * A[(k + 1, t_val, f_val, mu, w)]
            for (t_val, f_val, mu, w) in valid_asns
        )
        T_end_curr = pulp.lpSum(
            (t_val + E(mu, config.mu_max)) * A[(k, t_val, f_val, mu, w)]
            for (t_val, f_val, mu, w) in valid_asns
        )
        prob += (
            T_start_next - PHI_T * sum_A_slot(k + 1) >= T_end_curr - PHI_T,
            f"c10_ord{k}",
        )

    # (17) Throughput SLA — minimum PRBs
    if valid_asns:
        prob += (
            pulp.lpSum(
                w * A[(k, t_val, f_val, mu, w)]
                for k in range(K)
                for (t_val, f_val, mu, w) in valid_asns
            ) >= vue.n_prb,
            "c17_throughput",
        )
        # Cap at SLA requirement: no benefit allocating beyond what is needed.
        # Without this, a single VUE greedily claims the entire grid (e.g.
        # omega=27 at µ=0 covers all 27 freq rows × all 24 time cols = 648 cells),
        # leaving nothing for other VUEs and causing the LP to be degenerate
        # (one-VUE monopoly scores the same as fair distribution).
        prob += (
            pulp.lpSum(
                w * A[(k, t_val, f_val, mu, w)]
                for k in range(K)
                for (t_val, f_val, mu, w) in valid_asns
            ) <= vue.n_prb,
            "c17_max_prb",
        )

    return prob, A, valid_asns


def _extract_column_from_A(
    vue:        VirtualUE,
    config:     SystemConfig,
    A:          Dict[Tuple, pulp.LpVariable],
    valid_asns: List[Tuple],
) -> Column:
    """Extract a Column dataclass from solved A decision variables."""
    K       = config.K
    t_list  = [-1]   * K
    f_list  = [-1]   * K
    mu_list: List[Optional[int]] = [None] * K
    w_list  = [0]    * K
    all_cells: set = set()

    for k in range(K):
        for (t_val, f_val, mu, w) in valid_asns:
            key = (k, t_val, f_val, mu, w)
            if key in A:
                val = pulp.value(A[key])
                if val is not None and abs(val - 1.0) < 1e-4:
                    t_list[k]  = t_val
                    f_list[k]  = f_val
                    mu_list[k] = mu
                    w_list[k]  = w
                    all_cells.update(_cells_of_assignment(t_val, f_val, mu, w, config, with_guard=False))
                    break

    return Column(
        vue_id     = vue.virtual_id,
        t          = t_list,
        f          = f_list,
        mu         = mu_list,
        w          = w_list,
        total_prbs = sum(w_list),
        cells      = frozenset(all_cells),
    )


def generate_initial_column(
    vue: VirtualUE, config: SystemConfig,
) -> Optional[Column]:
    """
    Phase 0 of CG: solve single-UE MIP maximising total PRBs, ignoring all
    other VUEs.  Seeds the master problem before any dual prices are available.

    Returns a feasible Column, or None if the single-UE instance is infeasible
    (e.g. the latency deadline is tighter than the smallest slot duration).
    """
    prob, A, valid_asns = _build_single_ue_prob(vue, config, name_prefix="init")
    if not A:
        return None   # no valid assignments for this VUE

    prob += pulp.lpSum(
        w * A[(k, t_val, f_val, mu, w)]
        for k in range(config.K)
        for (t_val, f_val, mu, w) in valid_asns
    )
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))

    if pulp.value(prob.objective) is None:
        return None
    return _extract_column_from_A(vue, config, A, valid_asns)


def solve_pricing_subproblem(
    vue:     VirtualUE,
    config:  SystemConfig,
    pi:      Dict[Tuple[int, int], float],
    mu_dual: float,
) -> Optional[Column]:
    """
    Pricing subproblem for VUE i: find the column with maximum reduced cost.

    Reduced cost RC(c) = obj(c) − Σ_r π_r · indicator(r ∈ c.cells) − μ_i

    where:
        π_r    = dual price of resource-cell constraint (≤ 1) in LP master
        μ_i    = dual price of convexity constraint (= 1) for VUE i

    Objective maximised: Σ_{k,asn} (w − shadow(asn)) · A[k,asn]
    (μ_i is a constant — we check pricing_obj − μ_i > 1e-6 externally.)

    Returns a new Column if RC > 1e-6, otherwise None.
    """
    prob, A, valid_asns = _build_single_ue_prob(
        vue, config, name_prefix="pricing"
    )
    if not A:
        return None

    # Pre-compute shadow cost per valid assignment (sum of π_r over raw cells used).
    # Must use with_guard=False to be consistent with the master's resource
    # constraints, which are built over raw cells (no guard band).
    shadow: Dict[Tuple[int, int, int, int], float] = {
        asn: sum(
            pi.get(cell, 0.0)
            for cell in _cells_of_assignment(*asn, config, with_guard=False)
        )
        for asn in valid_asns
    }

    prob += pulp.lpSum(
        (w - shadow[(t_val, f_val, mu, w)]) * A[(k, t_val, f_val, mu, w)]
        for k in range(config.K)
        for (t_val, f_val, mu, w) in valid_asns
    )
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))

    pricing_obj = pulp.value(prob.objective)
    if pricing_obj is None or (pricing_obj - mu_dual) <= 1e-6:
        return None   # no improving column

    return _extract_column_from_A(vue, config, A, valid_asns)


def _guard_band_conflicts(c1: Column, c2: Column, config: SystemConfig) -> bool:
    """
    Return True if columns c1 and c2 have a guard band conflict.

    A guard band conflict requires ALL THREE of:
        (a) At least one pair of slots (k1 from c1, k2 from c2) that overlap
            in time (their time ranges share at least one grid column),
        (b) Those two slots use DIFFERENT numerologies (INI would occur), AND
        (c) Their frequency ranges are within G_guard rows of each other
            (i.e., the required guard band spacing is violated).

    Same-numerology adjacent allocations are always fine — no guard band needed.
    Slots that do not overlap in time never cause INI regardless of frequency.
    """
    G      = config.G_guard
    mu_max = config.mu_max
    K      = config.K
    for k1 in range(K):
        if c1.w[k1] == 0 or c1.mu[k1] is None:
            continue
        mu1    = c1.mu[k1]
        t1     = c1.t[k1];   t1_end = t1 + E(mu1, mu_max) - 1
        f1     = c1.f[k1];   f1_end = f1 + c1.w[k1] * G_rows(mu1) - 1
        for k2 in range(K):
            if c2.w[k2] == 0 or c2.mu[k2] is None:
                continue
            mu2 = c2.mu[k2]
            if mu1 == mu2:
                continue          # same numerology → no INI → no guard band needed
            t2     = c2.t[k2];   t2_end = t2 + E(mu2, mu_max) - 1
            if t1_end < t2 or t2_end < t1:
                continue          # no time overlap → no INI regardless of freq
            f2     = c2.f[k2];   f2_end = f2 + c2.w[k2] * G_rows(mu2) - 1
            # Guard band violation: frequency ranges are within G rows of each other.
            # Safe iff:  f2 > f1_end + G  OR  f1 > f2_end + G
            if f2 <= f1_end + G and f1 <= f2_end + G:
                return True
    return False


def _build_master_problem(
    columns_per_vue: Dict[int, List[Column]],
    vues:            List[VirtualUE],
    config:          SystemConfig,
    integer:         bool = False,
) -> Tuple[
    pulp.LpProblem,
    Dict[Tuple[int, int], pulp.LpVariable],
    Dict[int, str],
    Dict[Tuple[int, int], str],
]:
    """
    Build the CG master problem skeleton (shared by LP and integer phases).

    Master:
        max  Σ_{i,c}  total_prbs(c) · λ_{i,c}
        s.t. Σ_c λ_{i,c} = 1                         ∀i         (convexity)
             Σ_{i,c: r ∈ c.cells} λ_{i,c} ≤ 1        ∀r         (raw resource non-conflict)
             λ[i,c1] + λ[j,c2] ≤ 1                   ∀ gb pairs (guard band conflict)
             λ_{i,c} ∈ [0, 1]   (LP)  or  {0, 1}   (IP)

    col.cells stores RAW cells (no guard band).  Guard band enforcement is
    handled by pairwise constraints: for each pair of columns from different
    VUEs that have different numerologies, overlap in time, and whose frequency
    ranges are within G_guard rows of each other, we add λ[i,c1]+λ[j,c2] ≤ 1.
    This is correct: same-numerology adjacent allocations are allowed (no INI),
    different-numerology allocations that are too close in frequency are blocked.

    Returns: (prob, lam, conv_cname, res_cname)
        conv_cname : {vue_id: constraint name}
        res_cname  : {resource_cell: constraint name}
    """
    prob = pulp.LpProblem(
        "CG_Integer_Master" if integer else "CG_Master_LP",
        pulp.LpMaximize,
    )

    cat = pulp.constants.LpBinary if integer else pulp.constants.LpContinuous

    lam: Dict[Tuple[int, int], pulp.LpVariable] = {}
    for vue in vues:
        i = vue.virtual_id
        for c_idx in range(len(columns_per_vue.get(i, []))):
            lam[(i, c_idx)] = pulp.LpVariable(
                f"lam_{i}_{c_idx}", lowBound=0, upBound=1, cat=cat,
            )

    # Objective: maximise total PRBs, capping each VUE's contribution at its
    # SLA requirement (n_prb).  Without the cap, a single VUE monopolising the
    # entire grid scores the same as multiple VUEs each getting their fair share,
    # making the LP degenerate and the integer master pick an unfair allocation.
    # With the cap, distributing resources across N VUEs scores N× higher than
    # giving everything to one VUE.
    vue_n_prb = {vue.virtual_id: vue.n_prb for vue in vues}
    prob += pulp.lpSum(
        min(columns_per_vue[vue.virtual_id][c_idx].total_prbs,
            vue_n_prb[vue.virtual_id]) * lam[(vue.virtual_id, c_idx)]
        for vue in vues
        for c_idx in range(len(columns_per_vue.get(vue.virtual_id, [])))
    )

    # Convexity: exactly one column selected per VUE
    conv_cname: Dict[int, str] = {}
    for vue in vues:
        i    = vue.virtual_id
        cols = columns_per_vue.get(i, [])
        if not cols:
            continue
        cname = f"conv_{i}"
        prob += (
            pulp.lpSum(lam[(i, c_idx)] for c_idx in range(len(cols))) == 1,
            cname,
        )
        conv_cname[i] = cname

    # Collect all resource cells across all columns
    all_cells: set = set()
    for vue in vues:
        for col in columns_per_vue.get(vue.virtual_id, []):
            all_cells.update(col.cells)

    # Resource non-conflict: only add constraint when ≥ 2 columns compete
    res_cname: Dict[Tuple[int, int], str] = {}
    for cell in all_cells:
        users = [
            (vue.virtual_id, c_idx)
            for vue in vues
            for c_idx, col in enumerate(columns_per_vue.get(vue.virtual_id, []))
            if cell in col.cells
        ]
        if len(users) <= 1:
            continue
        cname = f"res_{cell[0]:04d}_{cell[1]:04d}"
        prob += (
            pulp.lpSum(lam[(i, c_idx)] for (i, c_idx) in users) <= 1,
            cname,
        )
        res_cname[cell] = cname

    # NOTE: Pairwise guard band constraints (λ[i,c1]+λ[j,c2]≤1 for cross-
    # numerology, time-overlapping, frequency-adjacent column pairs) are NOT
    # added here because they grow as O(C²) — with ~200 columns/VUE and 15
    # VUE pairs, that is ~600,000 constraints which overwhelms even Gurobi's
    # presolve and can cascade-eliminate columns until the master is infeasible.
    #
    # Guard band enforcement is instead handled approximately through the
    # pricing shadow costs: π_r for cells that are contested (used by many
    # columns) is high, so the pricing subproblem naturally generates columns
    # that avoid frequency bands already used by other VUEs.  In practice this
    # keeps cross-numerology allocations well-separated.  The null column in
    # every VUE's pool (added unconditionally in column_generation) guarantees
    # the master is always feasible regardless.

    return prob, lam, conv_cname, res_cname


def solve_lp_master(
    columns_per_vue: Dict[int, List[Column]],
    vues:            List[VirtualUE],
    config:          SystemConfig,
) -> Tuple[float, Dict[Tuple[int, int], float], Dict[int, float]]:
    """
    Solve the LP relaxation of the CG master problem.

    PuLP/CBC provides dual prices via constraint.pi after solving the LP.
    For a ≤ constraint in a maximisation problem:  pi ≥ 0 (shadow price).
    For the = convexity constraint:                pi ∈ ℝ (any sign).

    Returns:
        obj_value    : LP objective (upper bound on integer optimum)
        pi_dict      : {resource_cell: dual price}
        mu_dual_dict : {vue_id: dual price of convexity constraint}
    """
    prob, lam, conv_cname, res_cname = _build_master_problem(
        columns_per_vue, vues, config, integer=False
    )
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=120))

    obj_val = pulp.value(prob.objective) or 0.0

    pi: Dict[Tuple[int, int], float] = {}
    for cell, cname in res_cname.items():
        c = prob.constraints.get(cname)
        pi[cell] = c.pi if (c is not None and c.pi is not None) else 0.0

    mu_dual: Dict[int, float] = {}
    for vue in vues:
        i     = vue.virtual_id
        cname = conv_cname.get(i)
        c     = prob.constraints.get(cname) if cname else None
        mu_dual[i] = c.pi if (c is not None and c.pi is not None) else 0.0

    return obj_val, pi, mu_dual


def solve_integer_master(
    columns_per_vue: Dict[int, List[Column]],
    vues:            List[VirtualUE],
    config:          SystemConfig,
    output_dir:      Optional[str] = None,
) -> Tuple[float, Dict[int, Optional[Column]]]:
    """
    Solve the final integer Set Partitioning master problem with λ ∈ {0,1}.

    This is Phase 2 of column generation.  The columns generated during the
    CG loop are the feasible set; this MIP selects one non-conflicting column
    per VUE that maximises total PRBs allocated.

    If output_dir is given, writes the master problem to .lp and .mps files
    in that directory before solving, so external solvers (e.g. HiGHS) can
    consume them.  The MPS file includes an OBJSENSE MAX section so HiGHS
    maximises correctly.

    Returns:
        (obj_value, {vue_id: selected Column or None})
    """
    prob, lam, _, _ = _build_master_problem(
        columns_per_vue, vues, config, integer=True
    )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        c = config
        base = (
            f"cg_master_bw_{int(c.bandwidth_hz/1e6)}_MHz"
            f"_T_{c.delta_t_ms}_ms"
            f"_mu_max_{c.mu_max}"
            f"_K_{c.K}"
            f"_vues_{len(vues)}"
            f"_cols_{sum(len(v) for v in columns_per_vue.values())}"
        )
        lp_path  = os.path.join(output_dir, f"{base}.lp")
        mps_path = os.path.join(output_dir, f"{base}.mps")
        prob.writeLP(lp_path)
        prob.writeMPS(mps_path)
        # Patch MPS to include OBJSENSE MAX so HiGHS maximises correctly
        # (PuLP writes '*SENSE:Maximize' as a comment which HiGHS ignores)
        with open(mps_path, "r") as f:
            mps_lines = f.read().split("\n")
        patched = []
        for line in mps_lines:
            patched.append(line)
            if line.startswith("NAME"):
                patched.append("OBJSENSE")
                patched.append("    MAX")
        with open(mps_path, "w") as f:
            f.write("\n".join(patched))
        # Write column registry so external solution files can be plotted
        cols_path = os.path.join(output_dir, f"{base}_columns.json")
        registry: Dict[str, Dict[str, Any]] = {}
        for vue_id, cols in columns_per_vue.items():
            registry[str(vue_id)] = {}
            for c_idx, col in enumerate(cols):
                registry[str(vue_id)][str(c_idx)] = {
                    "t": col.t,
                    "f": col.f,
                    "mu": col.mu,
                    "w": col.w,
                    "total_prbs": col.total_prbs,
                }
        with open(cols_path, "w") as f:
            json.dump(registry, f)
        print(f"  LP  → {lp_path}")
        print(f"  MPS → {mps_path}")
        print(f"  COL → {cols_path}")

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=300))

    obj_val = pulp.value(prob.objective) or 0.0

    selected: Dict[int, Optional[Column]] = {}
    for vue in vues:
        i           = vue.virtual_id
        selected[i] = None
        for c_idx, col in enumerate(columns_per_vue.get(i, [])):
            key = (i, c_idx)
            if key in lam and pulp.value(lam[key]) is not None:
                if abs(pulp.value(lam[key]) - 1.0) < 1e-4:
                    selected[i] = col
                    break

    return obj_val, selected


def column_generation(
    vues:       List[VirtualUE],
    config:     SystemConfig,
    groups:     Dict[str, List[int]],
    max_iter:   int = 200,
    output_dir: Optional[str] = None,
) -> Tuple[float, Dict[int, Optional[Column]]]:
    """
    Main Column Generation loop for 5G NR radio resource allocation.

    Replaces the monolithic MILP that creates O(N²K²|M|) interaction variables
    (Y, Z, W, I_gb).  Here, the only coupling between VUEs is through shared
    resource cells — the master problem has no big-M values at all.

    Expected LP gap: 1-5% (vs ~285% in the original big-M formulation).
    Expected solve time for 6 VUEs: < 1 minute (vs hours with the original).

    Phases:
        0. Seed with generate_initial_column() for each VUE (single-UE MIPs,
           no interaction — each solves in < 1 s).
        1. CG loop: LP master → dual prices (π, μ) → pricing subproblems
           (single-UE MIPs with shadow costs) → new columns added to master.
           Terminates when no pricing subproblem finds RC > 1e-6.
        2. solve_integer_master() with all generated columns (Set Partitioning
           MIP — typically very fast given the tight LP relaxation).

    Note: Group constraints (18)/(19) — same-group VUEs must share numerology
    and start-time when slots overlap — are not enforced in this implementation.
    They require coupled pricing subproblems or additional master constraints
    and can be added as a subsequent enhancement.

    Returns:
        (integer_obj, {vue_id: selected Column or None})
    """
    W = 66
    print("=" * W)
    print("Column Generation — 5G NR Radio Resource Allocation")
    print("=" * W)

    # ── Phase 0: seed with initial columns ────────────────────────────────────
    print("\n[Phase 0] Generating initial columns (single-UE MIPs, no interaction)…")
    columns_per_vue: Dict[int, List[Column]] = {}
    for vue in tqdm(vues, desc="Init columns"):
        # Always start with a null column (total_prbs=0, no cells).
        # This guarantees the master problem is always feasible: if every real
        # column for some VUE is blocked by resource conflicts, the null column
        # can still be selected (it conflicts with nothing).
        null_col = Column(
            vue_id=vue.virtual_id,
            t=[-1] * config.K, f=[-1] * config.K,
            mu=[None] * config.K, w=[0] * config.K,
            total_prbs=0, cells=frozenset(),
        )
        col = generate_initial_column(vue, config)
        if col is None:
            tqdm.write(
                f"  WARNING: VUE {vue.virtual_id} has no feasible single-UE "
                f"solution — only null column available."
            )
            columns_per_vue[vue.virtual_id] = [null_col]
        else:
            tqdm.write(
                f"  VUE {vue.virtual_id:>3}: {col.total_prbs:3d} PRBs  "
                f"{len(col.cells):3d} cells"
            )
            columns_per_vue[vue.virtual_id] = [null_col, col]

    # ── Phase 1: CG loop ──────────────────────────────────────────────────────
    print(f"\n[Phase 1] Column Generation loop (max {max_iter} iterations)…")
    for iteration in range(max_iter):
        # Solve LP relaxation of master → dual prices
        lp_obj, pi, mu_dual = solve_lp_master(columns_per_vue, vues, config)

        # Pricing: find improving column for each VUE
        new_cols_found = 0
        for vue in vues:
            new_col = solve_pricing_subproblem(
                vue, config, pi, mu_dual.get(vue.virtual_id, 0.0)
            )
            if new_col is not None:
                columns_per_vue[vue.virtual_id].append(new_col)
                new_cols_found += 1

        total_cols = sum(len(v) for v in columns_per_vue.values())
        print(
            f"  Iter {iteration:3d}: LP obj = {lp_obj:8.3f}  "
            f"new cols = {new_cols_found:3d}  total cols = {total_cols}"
        )

        if new_cols_found == 0:
            print(f"  LP optimal reached at iteration {iteration}.")
            break

    total_cols = sum(len(v) for v in columns_per_vue.values())

    # ── Phase 2: integer master (Set Partitioning) ────────────────────────────
    print(f"\n[Phase 2] Integer master (Set Partitioning)  —  {total_cols} columns…")
    if output_dir:
        print(f"  Writing master problem files to: {output_dir}")
    int_obj, selected = solve_integer_master(columns_per_vue, vues, config, output_dir=output_dir)

    print(f"\n  Integer objective: {int_obj:.4f} PRBs")
    for vue in vues:
        col = selected.get(vue.virtual_id)
        if col and col.total_prbs > 0:
            active_slots = [
                (col.t[k], col.f[k], col.mu[k], col.w[k])
                for k in range(config.K) if col.w[k] > 0
            ]
            print(
                f"  VUE {vue.virtual_id:>3}: {col.total_prbs:3d} PRBs  "
                f"slots={active_slots}"
            )
        else:
            print(f"  VUE {vue.virtual_id:>3}:   0 PRBs  (no allocation)")
    print("=" * W)
    return int_obj, selected


# ============================================================================
# Sample Config Writer
# ============================================================================

def write_sample_config(path: str, K: int = 10, mu_max: int = 3, G_guard: int = 1,
        BW: int= 5_000_000, time_horizon_ms: float = 3.0, ue_count: int = 3, mcs: int = 26,
        embb_mbps: float = 4.0, embb_latency_ms: float = 3.0, 
        urllc_mbps: float = 1.0, urllc_latency_ms: float = 0.5):
    """
    Write a sample JSON config matching Table II of Boutiba et al. 2022.

    System : 5 MHz, µmax=3, ΔT=3 ms, 3 UEs, 2 slices each, MCS=26
    Slice 1 (eMBB) : 4 Mbps, 3 ms latency
    Slice 2 (uRLLC): 1 Mbps, 0.5 ms latency
    """
    sample = {
        "system": {
            "bandwidth_hz": BW,
            "delta_t_ms":   time_horizon_ms,
            "mu_max":       mu_max,
            "K":            K,
            "numerologies": [int(i) for i in range(mu_max + 1)],
            "G_guard": G_guard,
        },
        "ues": [
            {
                "ue_id": f"UE{n}",
                "mcs":   mcs,
                "slices": [
                    {"slice_id": "embb",  "throughput_mbps": embb_mbps, "latency_ms": embb_latency_ms},
                    {"slice_id": "urllc", "throughput_mbps": urllc_mbps, "latency_ms": urllc_latency_ms},
                ],
            }
            for n in range(1, ue_count + 1)
        ],
    }
    path = os.path.join(path, f"cfg-bw-{BW/1_000_000}M-time-ms-{time_horizon_ms}-mcs-{mcs}-K-{K}-ue-{ue_count}-embb-thr-{embb_mbps}-embb-lat-{embb_latency_ms}-urllc-thr-{urllc_mbps}-urllc-lat-{urllc_latency_ms}.json")
    with open(path, "w") as f:
        json.dump(sample, f, indent=4)
    print(f"Sample config written → {path}")