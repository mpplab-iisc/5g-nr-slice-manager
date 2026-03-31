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
import re
from collections import Counter
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

    Scalability constraints:
        w cap          : w ≤ vue.n_prb (master obj capped at n_prb; using more
                         cells never improves reduced cost)
        f alignment    : f is a multiple of G_rows(mu) (physical PRB alignment;
                         each PRB at µ occupies G_rows(µ) contiguous rows so the
                         start row must be grid-aligned — also avoids O(n_freq_rows)
                         dense enumeration for large bandwidths)
    """
    assignments: List[Tuple[int, int, int, int]] = []
    for mu in config.numerologies:
        duration  = E(mu, config.mu_max)
        g         = G_rows(mu)
        # Cap at vue.n_prb: no benefit to more PRBs (master obj capped; extra
        # cells increase shadow-cost shadow with no objective gain).
        max_w_val = min(max_omega(mu, config.n_freq_rows), vue.n_prb)
        t_max = min(config.n_time_cols, vue.latency_slots) - duration
        if t_max < 0:
            continue   # this numerology can never meet the latency deadline
        for t_val in range(0, t_max + 1):
            for w in range(1, max_w_val + 1):
                f_max = config.n_freq_rows - w * g
                if f_max < 0:
                    continue
                # Step f by g (PRB alignment): f must be a multiple of G_rows(µ).
                # At µ=0, g=1 so every row is valid (no change). At µ=1, g=2 so
                # only even rows are valid (halves the search space), etc.
                for f_val in range(0, f_max + 1, g):
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
        # No upper cap on total PRBs — VUEs are allowed to use more than n_prb
        # when resources are available. The serve_bonus in the integer master
        # ensures all VUEs are served first (SLA met), and after that the master
        # naturally assigns leftover cells to VUEs that can fill them, maximising
        # grid utilisation.

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


def generate_initial_column_at_time(
    vue: VirtualUE, config: SystemConfig, t_lo: int, t_hi: int,
) -> Optional[Column]:
    """
    Phase 0 variant: generate a feasible column where slot 0 must start in
    [t_lo, t_hi).  Used to seed the master with diverse time positions so the
    integer master always has non-overlapping single-slot options for every VUE.

    Returns None if no feasible solution exists in this time window (e.g. a
    strict latency deadline prevents the VUE from starting that late).
    """
    prob, A, valid_asns = _build_single_ue_prob(vue, config, name_prefix="init_t")
    if not A:
        return None

    # Force slot 0 to start within [t_lo, t_hi) by zeroing the upper bound of
    # every variable where t_val is outside the window.  Cascade constraints then
    # force slots 1, 2 to also move later, so the whole allocation shifts right.
    for (t_val, f_val, mu, w) in valid_asns:
        if not (t_lo <= t_val < t_hi):
            A[(0, t_val, f_val, mu, w)].upBound = 0

    # Primary objective: satisfy n_prb (equality constraint already enforces this).
    # Tiebreaker: prefer single-slot, earliest-time, lowest-frequency allocations.
    #   - Penalise higher slot index k  → discourages multi-slot fragmentation
    #   - Penalise higher t_val         → places slot as early as possible in segment
    #   - Penalise higher f_val         → packs toward frequency row 0
    # EPSILON is tiny so it never overrides the primary PRB objective.
    EPSILON = 1e-3 / max(1, config.n_time_cols + config.n_freq_rows)
    prob += pulp.lpSum(
        (w - EPSILON * (t_val + f_val + k * config.n_time_cols))
        * A[(k, t_val, f_val, mu, w)]
        for k in range(config.K)
        for (t_val, f_val, mu, w) in valid_asns
    )
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))

    if pulp.value(prob.objective) is None or pulp.value(prob.objective) < 0.5:
        return None
    col = _extract_column_from_A(vue, config, A, valid_asns)
    if col.total_prbs == 0:
        return None
    return col


def generate_initial_column_at_freq(
    vue: VirtualUE, config: SystemConfig, f_lo: int,
) -> Optional[Column]:
    """
    Phase 0 variant: generate a feasible column where slot 0 must start at
    frequency row ≥ f_lo.  Used to seed VUEs above the eMBB frequency band so
    the integer master has non-conflicting high-frequency options from the start.

    The tiebreaker includes a term proportional to w * G_rows(mu) so that
    narrower-footprint numerologies (mu=1, G_rows=1) are preferred over wider
    ones (mu=3, G_rows=2) at the same w and f.  This prevents different seed
    calls from producing overlapping frequency bands.

    Returns None if no feasible column exists at f ≥ f_lo.
    """
    prob, A, valid_asns = _build_single_ue_prob(vue, config, name_prefix="init_f")
    if not A:
        return None

    # Restrict ALL slots to f ≥ f_lo so the entire column stays above the
    # eMBB frequency band.  Restricting only slot 0 would still allow later
    # slots to fall back to low-f positions (the solver uses them to reach
    # the n_prb equality target), producing multi-slot columns that partially
    # overlap with eMBB allocations and defeat the purpose of high-f seeding.
    for kk in range(config.K):
        for (t_val, f_val, mu, w) in valid_asns:
            if f_val < f_lo:
                A[(kk, t_val, f_val, mu, w)].upBound = 0

    # Tiebreaker: prefer single-slot, earliest-time, lowest-f, narrowest footprint.
    # The w * G_rows(mu) penalty differentiates mu at the same (t, f, w):
    #   mu=1 (G_rows=1): footprint = 4 rows for w=4 → small penalty
    #   mu=3 (G_rows=2): footprint = 8 rows for w=4 → larger penalty
    # This ensures seeds at adjacent f_lo values use disjoint narrow bands.
    EPSILON = 1e-3 / max(1, config.n_time_cols + config.n_freq_rows)
    prob += pulp.lpSum(
        (w - EPSILON * (t_val + f_val + k * config.n_time_cols + w * G_rows(mu)))
        * A[(k, t_val, f_val, mu, w)]
        for k in range(config.K)
        for (t_val, f_val, mu, w) in valid_asns
    )
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))

    if pulp.value(prob.objective) is None or pulp.value(prob.objective) < 0.5:
        return None
    col = _extract_column_from_A(vue, config, A, valid_asns)
    # Reject columns that don't meet the SLA: the solver may return a partial
    # solution (upBound restrictions make the equality constraint infeasible,
    # and CBC sometimes returns a best-effort result rather than declaring
    # infeasibility).  Any column with fewer than n_prb PRBs is geometrically
    # incompatible with this f_lo and should not enter the master pool.
    if col.total_prbs < vue.n_prb:
        return None
    return col


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
    serve_bonus:     float = 0.0,
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
    #
    # Compactness bonus: add a tiny bonus (COMPACT_BONUS ≪ 1 PRB) for single-slot
    # columns (only slot k=0 active, all others idle).  Applied ONLY to the
    # integer master (integer=True); the LP relaxation uses COMPACT_BONUS=0 to
    # avoid the LP pricing loop running forever (bonus > 1e-6 stopping threshold
    # causes pricing to always find "more compact" columns with positive RC).
    K = config.K
    COMPACT_BONUS = 1e-3 if integer else 0.0
    # serve_bonus: added to any non-null column's coefficient in the integer master
    # (serve_bonus=0 in the LP so duals remain meaningful for pricing).
    # Setting serve_bonus > max_possible_PRBs_per_VUE guarantees the master always
    # prefers serving all N VUEs over monopolising resources for fewer VUEs.
    prob += pulp.lpSum(
        (col.total_prbs
         + serve_bonus * int(col.total_prbs > 0)
         + COMPACT_BONUS * int(all(col.w[kk] == 0 for kk in range(1, K))))
        * lam[(vue.virtual_id, c_idx)]
        for vue in vues
        for c_idx, col in enumerate(columns_per_vue.get(vue.virtual_id, []))
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


def plot_lp_shadow_prices(
    columns_per_vue: Dict[int, List[Column]],
    vues:            List[VirtualUE],
    config:          SystemConfig,
    output_path:     str,
) -> str:
    """
    Diagnostic: solve the LP master, extract π_r dual prices, and save a
    (n_time_cols × n_freq_rows) heatmap PNG.

    Cells with π_r = 0 are uncontested (no VUE wants them → they go empty in
    the integer solution).  High π_r cells are heavily contested — the master
    is actively rationing them between VUEs.

    Returns output_path.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise ImportError("matplotlib and numpy are required.") from exc

    _, pi, _ = solve_lp_master(columns_per_vue, vues, config)

    grid = np.zeros((config.n_freq_rows, config.n_time_cols))
    for (t_col, f_row), price in pi.items():
        if 0 <= t_col < config.n_time_cols and 0 <= f_row < config.n_freq_rows:
            grid[f_row, t_col] = max(price, 0.0)

    fig, ax = plt.subplots(figsize=(min(16, max(8, config.n_time_cols * 0.5)),
                                    min(24, max(5, config.n_freq_rows * 0.05))))
    im = ax.imshow(grid, aspect='auto', origin='lower', cmap='YlOrRd',
                   extent=[-0.5, config.n_time_cols - 0.5,
                            -0.5, config.n_freq_rows - 0.5])
    plt.colorbar(im, ax=ax, label='Shadow price π_r  (0 = uncontested, white = empty)')
    ax.set_xlabel(f'Grid Columns (Unit: δT = {config.delta_t_ms/config.n_time_cols*1000:.3f}ms)')
    ax.set_ylabel(f'Grid Rows (Unit: δF = 180kHz)')
    ax.set_title('LP Master Shadow Prices — contested cells are red, idle cells are white')
    ax.grid(True, linewidth=0.3, color='grey', alpha=0.4)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Shadow price heatmap → {output_path}")
    return output_path


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
    # serve_bonus > max possible PRBs any single VUE can contribute.
    # This ensures the integer master always prefers serving all VUEs over
    # letting a few monopolise the grid (which would leave others with null columns).
    max_col_prbs = max(
        (col.total_prbs for cols in columns_per_vue.values() for col in cols),
        default=0,
    )
    serve_bonus = float(max_col_prbs + 1)

    prob, lam, _, _ = _build_master_problem(
        columns_per_vue, vues, config, integer=True, serve_bonus=serve_bonus
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

    # Write a CBC solution file in Gurobi/HiGHS .sol format so that
    # plot_cg_solution() can render the result without an external solver.
    sol_path: Optional[str] = None
    if output_dir is not None:
        sol_path = os.path.join(output_dir, f"{base}.sol")
        with open(sol_path, "w") as f:
            f.write(f"# Objective value = {obj_val}\n")
            for vue in vues:
                i = vue.virtual_id
                for c_idx, col in enumerate(columns_per_vue.get(i, [])):
                    key = (i, c_idx)
                    if key in lam and pulp.value(lam[key]) is not None:
                        if abs(pulp.value(lam[key]) - 1.0) < 1e-4:
                            f.write(f"lam_{i}_{c_idx} 1\n")
        print(f"  SOL → {sol_path}")

    return obj_val, selected, sol_path


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

    # Large-grid flag: grids with many frequency rows need special handling.
    # For n_freq_rows > 100, MIP-based Phase 0 seeds (which call CBC for each VUE)
    # are too slow (100 VUEs × 3 seeds × 30s ≈ 2.5 hrs). Phase 0d/0e direct seeds
    # are sufficient and run in microseconds.
    LARGE_GRID = config.n_freq_rows > 100

    # ── Phase 0: seed with diverse initial columns ────────────────────────────
    # Two kinds of diversity are seeded:
    #
    # (a) Time diversity: divide the time grid into 3 equal segments and generate
    #     one column per segment per VUE.  eMBB VUEs end up with non-overlapping
    #     single-slot options at t=0/8/16, all at f=0.
    #
    # (b) Frequency diversity: generate columns starting above the eMBB frequency
    #     band (rows 0..max_embb_n_prb-1) at evenly-spaced positions.  This gives
    #     URLLC VUEs (which are forced to t=0 by latency) non-conflicting high-f
    #     options from the very first iteration, so the integer master does not
    #     have to rely solely on pricing-generated columns to avoid eMBB overlap.
    #
    # Together these ensure the master pool contains a complete non-fragmented,
    # non-conflicting assignment for every VUE before any CG iterations run.
    print("\n[Phase 0] Generating diverse initial columns (time + frequency)…")
    columns_per_vue: Dict[int, List[Column]] = {}

    seg_size  = max(1, config.n_time_cols // 3)
    t_segments = list(range(0, config.n_time_cols, seg_size))

    # Compute frequency-diverse seed positions above the eMBB band.
    # eMBB at f=0, w=max_embb_n_prb, G_rows(0)=1 → top row = max_embb_n_prb - 1.
    # Guard-band check triggers when f2 ≤ f1_end + G_guard, so the first safe
    # URLLC start is f = max_embb_n_prb + G_guard (= 15 for 5 MHz, G_guard=1).
    # At G_rows=1 (mu=1, narrowest above mu=0) each seed occupies exactly
    # max_urllc_n_prb rows with no inter-URLLC guard band needed (same mu).
    max_embb_n_prb  = max((v.n_prb for v in vues if v.sla.slice_id == 'embb'),  default=0)
    max_urllc_n_prb = max((v.n_prb for v in vues if v.sla.slice_id == 'urllc'), default=0)
    # min_urllc_latency: the earliest column at which URLLC must have finished.
    # eMBB seeded at t=0 by the tiebreaker conflicts with URLLC (which locks
    # rows 3-26 for cols 0..latency-1).  Seeding eMBB at t=min_urllc_latency
    # (typically t=4 for 0.5ms URLLC latency) fills the gap at cols 4-7 that
    # would otherwise be completely empty.
    min_urllc_latency = min(
        (v.latency_slots for v in vues if v.sla.slice_id == 'urllc'),
        default=0,
    )
    # f_above_embb: first frequency row that avoids guard-band conflict with eMBB.
    # eMBB at f=0, w=max_embb_n_prb, G_rows(0)=1 → top row = max_embb_n_prb - 1.
    # Guard check triggers when f2 ≤ top_row + G_guard → safe start = top_row + G_guard + 1.
    # = max_embb_n_prb - 1 + G_guard + 1 = max_embb_n_prb + G_guard.
    f_above_embb = max_embb_n_prb + config.G_guard      # first safe high-f row
    # urllc_step: frequency rows consumed per URLLC VUE at the finest feasible mu.
    # For URLLC with latency constraints, mu=0 is typically infeasible (E(0) > latency),
    # so the finest usable mu is mu=1 with G_rows(1)=2.  Each URLLC VUE at mu=1 needs
    # max_urllc_n_prb * G_rows(1) rows.  Using G_rows(0)=1 would undercount, producing
    # geometrically infeasible seed positions.
    min_urllc_mu = min(
        (mu for mu in config.numerologies
         if min(config.n_time_cols, max((v.latency_slots for v in vues
                                         if v.sla.slice_id == 'urllc'), default=0))
            >= E(mu, config.mu_max)),
        default=1,
    )
    urllc_step = max_urllc_n_prb * G_rows(min_urllc_mu) if max_urllc_n_prb > 0 else 8
    n_freq_seeds = max(3, sum(1 for v in vues if v.sla.slice_id == 'urllc'))
    f_diverse_seeds = [
        f_above_embb + i * urllc_step
        for i in range(n_freq_seeds)
        if f_above_embb + i * urllc_step + urllc_step <= config.n_freq_rows
    ]

    for vue in tqdm(vues, desc="Init columns"):
        # Always include a null column so the master is always feasible.
        null_col = Column(
            vue_id=vue.virtual_id,
            t=[-1] * config.K, f=[-1] * config.K,
            mu=[None] * config.K, w=[0] * config.K,
            total_prbs=0, cells=frozenset(),
        )
        columns_per_vue[vue.virtual_id] = [null_col]

        n_added = 0

        if not LARGE_GRID:
            # (a) Time-diverse seeds (3 time segments)
            for t_lo in t_segments:
                t_hi = min(t_lo + seg_size, config.n_time_cols)
                col = generate_initial_column_at_time(vue, config, t_lo, t_hi)
                if col is not None:
                    columns_per_vue[vue.virtual_id].append(col)
                    n_added += 1

            # (c) Post-URLLC seed for eMBB VUEs.
            # URLLC occupies cols 0..(min_urllc_latency-1) at rows 3-26, so any
            # eMBB column starting at t=0 conflicts with URLLC.  The time-diverse
            # seed for segment [0, seg_size) picks t=0 by tiebreaker, which the
            # master can never select — leaving cols 4-7 completely empty.
            # This seed forces one column to start exactly at t=min_urllc_latency
            # (t=4 for 0.5ms URLLC latency), filling the dead-zone gap.
            if vue.sla.slice_id == 'embb' and min_urllc_latency > 0:
                col = generate_initial_column_at_time(
                    vue, config, min_urllc_latency, min_urllc_latency + seg_size
                )
                if col is not None:
                    columns_per_vue[vue.virtual_id].append(col)
                    n_added += 1

            # (b) Frequency-diverse seeds above the eMBB band.
            # Only applied to non-eMBB (URLLC) VUEs.  eMBB VUEs have n_prb too
            # large to fit in a single slot above f_above_embb, so they would
            # generate multi-slot high-f columns that compete with URLLC seeds
            # and waste frequency space.  eMBB VUEs already get full diversity
            # from the 3 time-segment seeds at f=0.
            if vue.sla.slice_id != 'embb':
                for f_lo in f_diverse_seeds:
                    col = generate_initial_column_at_freq(vue, config, f_lo)
                    if col is not None:
                        columns_per_vue[vue.virtual_id].append(col)
                        n_added += 1

        if n_added == 0 and not LARGE_GRID:
            tqdm.write(
                f"  WARNING: VUE {vue.virtual_id} has no feasible single-UE "
                f"solution — only null column available."
            )
        elif n_added > 0:
            tqdm.write(
                f"  VUE {vue.virtual_id:>3}: {n_added} initial columns seeded"
            )

    # ── Phase 0d: Symmetry-breaking exclusive seeds for eMBB VUEs ────────────
    # When multiple eMBB VUEs have identical SLA parameters, the standard time-
    # diverse seeds (step a) generate the exact same columns for each VUE.  They
    # all conflict at f=0, so the master assigns null to all but one.
    #
    # Fix: divide the frequency grid into exclusive bands (one per eMBB VUE) and
    # directly construct a Column for each VUE that occupies only its own band.
    # eMBB VUEs can only use mu=0 (higher mu exceed grid height), so G_rows(0)=1
    # and E(0,mu_max)=8.  Two slots at t=t0 and t=t0+E each contribute n_prb/2
    # PRBs, staying strictly within the assigned frequency band.  Because the
    # bands are disjoint the columns cannot conflict, regardless of time overlap.
    embb_vues_sorted = sorted(
        [v for v in vues if v.sla.slice_id == 'embb'], key=lambda v: v.virtual_id
    )
    n_embb = len(embb_vues_sorted)
    embb_freq_watermark = 0  # tracks highest freq row used by eMBB seeds
    if n_embb > 1:
        # Pick the coarsest mu feasible for ALL eMBB VUEs (typically mu=0).
        mu_embb = min(
            (mu for mu in config.numerologies
             if max((v.latency_slots for v in embb_vues_sorted), default=0)
                >= E(mu, config.mu_max)),
            default=0,
        )
        e_embb   = E(mu_embb, config.mu_max)
        g_embb   = G_rows(mu_embb)
        band_size = config.n_freq_rows // n_embb  # rows available per VUE (equal-band fallback)

        # t0 fallback: if the uRLLC zone leaves no room for even one eMBB slot,
        # start eMBB at t=0 (they share time with uRLLC but at different frequencies).
        t0 = min_urllc_latency  # start after uRLLC zone
        if t0 + e_embb > config.n_time_cols:
            t0 = 0

        # 1-slot packing: if all eMBB VUEs fit in the frequency grid with one slot
        # each (using exactly w=n_prb PRBs), pack them with cumulative f offsets.
        # This handles large grids where n_prb varies across VUEs (heterogeneous).
        # The equal-band multi-slot approach is used otherwise (small grids like 5MHz).
        total_embb_rows_1slot = sum(v.n_prb * g_embb for v in embb_vues_sorted)
        use_1slot_packing = (
            total_embb_rows_1slot <= config.n_freq_rows
            and t0 + e_embb <= config.n_time_cols
        )

        if use_1slot_packing:
            # ── Cumulative 1-slot seeds: one slot per VUE at w=n_prb ──────────
            # Each VUE gets a contiguous band of exactly n_prb rows starting right
            # after the previous VUE's band.  No intra-band waste.  Works for
            # heterogeneous n_prb values and small time grids (n_time_cols = e_embb).
            f_lo_cursor = 0
            for rank, vue in enumerate(embb_vues_sorted):
                w_single = vue.n_prb
                if f_lo_cursor + w_single * g_embb > config.n_freq_rows:
                    break  # no more room
                if t0 + e_embb > vue.latency_slots:
                    f_lo_cursor += w_single * g_embb
                    continue

                K = config.K
                t_arr  = [t0]         + [-1]   * (K - 1)
                f_arr  = [f_lo_cursor]+ [-1]   * (K - 1)
                mu_arr = [mu_embb]    + [None] * (K - 1)
                w_arr  = [w_single]   + [0]    * (K - 1)

                all_cells: set = set()
                all_cells |= _cells_of_assignment(
                    t0, f_lo_cursor, mu_embb, w_single, config, with_guard=False,
                )

                col = Column(
                    vue_id=vue.virtual_id,
                    t=t_arr, f=f_arr, mu=mu_arr, w=w_arr,
                    total_prbs=w_single,
                    cells=frozenset(all_cells),
                )
                columns_per_vue[vue.virtual_id].append(col)
                print(
                    f"  VUE {vue.virtual_id:>3}: exclusive eMBB seed "
                    f"f=[{f_lo_cursor},{f_lo_cursor + w_single * g_embb})  "
                    f"mu={mu_embb}  slots=1  {col.total_prbs} PRBs"
                )
                f_lo_cursor += w_single * g_embb

            embb_freq_watermark = f_lo_cursor

        else:
            # ── Equal-band multi-slot seeds (original approach) ───────────────
            # Used when 1-slot packing would overflow the frequency grid
            # (e.g., 3-VUE 5MHz: 3×14=42 rows > 27).  Each VUE gets a band of
            # band_size rows and uses ceil(n_prb/max_w) slots to meet its SLA.
            for rank, vue in enumerate(embb_vues_sorted):
                f_lo = rank * band_size
                # Max PRBs that fit in one slot within this band
                max_w = band_size // g_embb
                if max_w <= 0:
                    continue  # band too narrow — skip

                # Determine how many mu_embb slots are needed to satisfy n_prb SLA.
                slots_needed = (vue.n_prb + max_w - 1) // max_w  # ceiling div
                if slots_needed > config.K:
                    continue  # can't meet SLA in this band — skip

                # Build slot start times: t0, t0+e, t0+2e, ...
                slot_times = [t0 + s * e_embb for s in range(slots_needed)]
                # Check all slots fit within time horizon and latency deadline
                if any(t + e_embb > config.n_time_cols for t in slot_times):
                    continue
                if any(t + e_embb > vue.latency_slots for t in slot_times):
                    continue

                # Fill the entire band (max_w PRBs per slot) rather than distributing
                # n_prb evenly. This covers all rows in the band, eliminating the
                # intra-band frequency gaps (rows 7-8, 16-17, 25-26) that previously
                # went empty. sum(w_vals) ≥ n_prb satisfies the SLA lower bound.
                w_vals = [max_w] * slots_needed

                # ── Extra gap-fill slot at t=20-23 using mu=1 ────────────────────
                # With mu=0 (E=8) and starting at t=min_urllc_latency=4, only 2 slots
                # fit (t=4-11, t=12-19) — t=20-23 is completely unreachable.  A single
                # mu=1 slot (E=4) at t=20 fills these 4 time columns.  Different ranks
                # get non-overlapping mu=1 frequency bands so all n_embb VUEs can
                # contribute to the t=20-23 region without conflicting.
                extra_t  = []
                extra_f  = []
                extra_mu = []
                extra_w  = []
                mu_fine = 1
                if mu_fine in config.numerologies:
                    e_fine = E(mu_fine, config.mu_max)   # 4 cols
                    g_fine = G_rows(mu_fine)              # 2 rows/PRB
                    t_fine = t0 + slots_needed * e_embb  # 4 + 2*8 = 20
                    if (slots_needed < config.K and
                            t_fine + e_fine <= config.n_time_cols and
                            t_fine + e_fine <= vue.latency_slots):
                        # Distribute the freq grid (27 rows) among n_embb ranks at mu=1
                        total_fine = config.n_freq_rows // g_fine   # 27//2=13
                        base_fine  = total_fine // n_embb            # 13//3=4
                        xtra_fine  = total_fine  % n_embb            # 1
                        w_fine = base_fine + (1 if rank < xtra_fine else 0)
                        f_fine = sum(
                            (base_fine + (1 if r < xtra_fine else 0)) * g_fine
                            for r in range(rank)
                        )
                        if w_fine > 0 and f_fine + w_fine * g_fine <= config.n_freq_rows:
                            extra_t  = [t_fine]
                            extra_f  = [f_fine]
                            extra_mu = [mu_fine]
                            extra_w  = [w_fine]

                # Pad to length K
                K = config.K
                active_slots = slots_needed + len(extra_t)
                t_arr  = slot_times + extra_t  + [-1]   * (K - active_slots)
                f_arr  = [f_lo] * slots_needed + extra_f  + [-1]   * (K - active_slots)
                mu_arr = [mu_embb] * slots_needed + extra_mu + [None] * (K - active_slots)
                w_arr  = w_vals + extra_w + [0] * (K - active_slots)

                # Compute raw cells (with_guard=False, consistent with master)
                all_cells: set = set()
                for s in range(active_slots):
                    all_cells |= _cells_of_assignment(
                        t_arr[s], f_arr[s], mu_arr[s], w_arr[s],
                        config, with_guard=False,
                    )

                col = Column(
                    vue_id=vue.virtual_id,
                    t=t_arr, f=f_arr, mu=mu_arr, w=w_arr,
                    total_prbs=sum(w_arr),
                    cells=frozenset(all_cells),
                )
                columns_per_vue[vue.virtual_id].append(col)
                extra_info = (
                    f" + mu=1 t={extra_t[0]}-{extra_t[0]+E(mu_fine,config.mu_max)-1}"
                    f" f={extra_f[0]} w={extra_w[0]}"
                    if extra_t else ""
                )
                print(
                    f"  VUE {vue.virtual_id:>3}: exclusive eMBB seed "
                    f"f=[{f_lo},{f_lo+band_size})  mu={mu_embb}  "
                    f"slots={slots_needed}{extra_info}  {col.total_prbs} PRBs"
                )

            embb_freq_watermark = n_embb * band_size

    # ── Phase 0e: Symmetry-breaking exclusive frequency seeds for uRLLC VUEs ──
    # All uRLLC VUEs are confined to cols 0..(min_urllc_latency-1) (latency),
    # so they cannot be time-multiplexed like eMBB.  Using generate_initial_
    # column_at_freq() lets the solver use up to K*n_prb PRBs, causing the seeds
    # to overflow their intended band and conflict with each other.
    #
    # Fix: directly construct Column objects with exactly n_prb PRBs per VUE,
    # each restricted to a non-overlapping frequency band.  Use the coarsest
    # feasible mu (min_urllc_mu) to minimise the frequency footprint.
    urllc_vues_sorted = sorted(
        [v for v in vues if v.sla.slice_id == 'urllc'], key=lambda v: v.virtual_id
    )
    n_urllc = len(urllc_vues_sorted)
    if n_urllc > 1 and min_urllc_latency > 0:
        g_urllc = G_rows(min_urllc_mu)
        e_urllc = E(min_urllc_mu, config.mu_max)

        # For large grids (LARGE_GRID), eMBB seeds already use rows 0..watermark-1,
        # so uRLLC seeds must start at the watermark to avoid resource conflicts that
        # would force the master to assign null columns to uRLLC VUEs.
        # For small grids (5MHz), eMBB and uRLLC use different time columns (eMBB at
        # t≥min_urllc_latency, uRLLC at t=0..min_urllc_latency-1), so they don't
        # conflict in cells even when using the same frequency rows → start at 0.
        f_urllc_base = embb_freq_watermark if LARGE_GRID else 0
        available_urllc_rows = config.n_freq_rows - f_urllc_base
        if available_urllc_rows <= 0:
            # No room — all uRLLC VUEs will rely on pricing to find their columns.
            total_urllc_prbs = 0
        else:
            # Distribute ALL available frequency rows among uRLLC VUEs, not just the
            # bare n_prb minimum.  With G_rows=2 and 27 rows, each VUE can serve more
            # PRBs than its SLA requires (total_fine = 13 PRBs fills 26 rows, leaving
            # only 1 row wasted vs 3 rows wasted with the old equal-band approach).
            # base/extra logic mirrors Phase 0d's mu=1 frequency distribution.
            total_urllc_prbs = available_urllc_rows // g_urllc
        base_urllc  = total_urllc_prbs // n_urllc if n_urllc > 0 else 0
        extra_urllc = total_urllc_prbs  % n_urllc if n_urllc > 0 else 0

        for rank, vue in enumerate(urllc_vues_sorted):
            if total_urllc_prbs == 0:
                break  # no frequency space for uRLLC seeds
            # Width: spread the available PRBs as evenly as possible;
            # rank 0 gets 1 extra PRB when total doesn't divide evenly.
            w_seed = base_urllc + (1 if rank < extra_urllc else 0)
            if w_seed == 0:
                break  # no more PRBs to distribute
            f_offset = sum(
                base_urllc + (1 if r < extra_urllc else 0)
                for r in range(rank)
            ) * g_urllc
            f_lo = f_urllc_base + f_offset

            # Verify the slot fits within the grid and latency
            if f_lo + w_seed * g_urllc > config.n_freq_rows:
                break  # no more room
            if e_urllc > min_urllc_latency:
                continue  # shouldn't happen

            # Single slot at t=0; cascade: only slot 0 active
            K = config.K
            t_arr  = [0]             + [-1]   * (K - 1)
            f_arr  = [f_lo]          + [-1]   * (K - 1)
            mu_arr = [min_urllc_mu]  + [None] * (K - 1)
            w_arr  = [w_seed]        + [0]    * (K - 1)

            all_cells: set = set()
            all_cells |= _cells_of_assignment(
                t_arr[0], f_arr[0], mu_arr[0], w_arr[0],
                config, with_guard=False,
            )

            col = Column(
                vue_id=vue.virtual_id,
                t=t_arr, f=f_arr, mu=mu_arr, w=w_arr,
                total_prbs=w_seed,
                cells=frozenset(all_cells),
            )
            columns_per_vue[vue.virtual_id].append(col)
            urllc_band_end = f_lo + w_seed * g_urllc
            print(
                f"  VUE {vue.virtual_id:>3}: exclusive uRLLC seed "
                f"f=[{f_lo},{urllc_band_end})  mu={min_urllc_mu}  "
                f"t=0  {col.total_prbs} PRBs"
            )

    # ── Phase 1: CG loop ──────────────────────────────────────────────────────
    print(f"\n[Phase 1] Column Generation loop (max {max_iter} iterations)…")
    if LARGE_GRID:
        print(
            f"  Skipping CG iterations for large grid "
            f"(n_freq_rows={config.n_freq_rows} > 100): Phase 0d/0e seeds are "
            f"optimal; pricing subproblems too large for CBC."
        )
    else:
        prev_lp_obj   = -float("inf")
        lp_stall_cnt  = 0
        LP_STALL_STOP = max(3, len(vues) // 10)  # scales with problem size (10 for 100 VUEs)
        for iteration in range(max_iter):
            # Solve LP relaxation of master → dual prices
            lp_obj, pi, mu_dual = solve_lp_master(columns_per_vue, vues, config)

            # Track LP stall: if objective doesn't improve, CG has effectively converged
            if abs(lp_obj - prev_lp_obj) < 1e-6:
                lp_stall_cnt += 1
            else:
                lp_stall_cnt = 0
            prev_lp_obj = lp_obj

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
            if lp_stall_cnt >= LP_STALL_STOP:
                print(
                    f"  LP stalled for {LP_STALL_STOP} consecutive iterations "
                    f"(obj={lp_obj:.3f}) — stopping CG early."
                )
                break

    total_cols = sum(len(v) for v in columns_per_vue.values())

    # ── Phase 2: integer master (Set Partitioning) ────────────────────────────
    print(f"\n[Phase 2] Integer master (Set Partitioning)  —  {total_cols} columns…")
    if output_dir:
        print(f"  Writing master problem files to: {output_dir}")
    int_obj, selected, sol_path = solve_integer_master(columns_per_vue, vues, config, output_dir=output_dir)

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
    return int_obj, selected, sol_path, columns_per_vue


# ============================================================================
# Solution Statistics
# ============================================================================

def _group_consecutive(nums: List[int]) -> List[str]:
    """Convert a sorted list of ints into compact range strings, e.g. [0,1,2,5] → ['0-2','5']."""
    if not nums:
        return []
    ranges, start, end = [], nums[0], nums[0]
    for n in nums[1:]:
        if n == end + 1:
            end = n
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = end = n
    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ranges


def _connected_components(cells: set) -> List[List]:
    """4-connected flood-fill; returns list of component cell-lists."""
    visited, components = set(), []
    for seed in cells:
        if seed in visited:
            continue
        stack, component = [seed], []
        while stack:
            c = stack.pop()
            if c in visited or c not in cells:
                continue
            visited.add(c)
            component.append(c)
            t, f = c
            for dt, df in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (t + dt, f + df)
                if nb in cells and nb not in visited:
                    stack.append(nb)
        components.append(component)
    return components


def print_solution_stats(
    config,
    vues: List[VirtualUE],
    solution_path: str,
    columns_path: str,
) -> None:
    """
    Print quantitative grid-utilisation and gap statistics for a CG solution.
    Designed to be called after plot generation so the log is self-describing.
    """
    W = 68
    DIV  = "=" * W
    DIV2 = "-" * W

    # ── Load columns registry ────────────────────────────────────────────────
    with open(columns_path) as fh:
        cols_data = json.load(fh)

    # ── Parse solution file (Gurobi / CBC .sol) ──────────────────────────────
    selected_lam: Dict[int, int] = {}  # vue_id → col_idx
    obj_val: Optional[float] = None
    with open(solution_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#') and 'Objective' in line:
                m = re.search(r'[=:]\s*([-\d.e+]+)', line)
                if m:
                    try:
                        obj_val = float(m.group(1))
                    except ValueError:
                        pass
                continue
            # Gurobi: "lam_VID_CIDX  1" or "lam_VID_CIDX  0.9999..."
            m = re.match(r'^lam_(\d+)_(\d+)\s+([\d.e+\-]+)', line)
            if m and float(m.group(3)) > 0.5:
                selected_lam[int(m.group(1))] = int(m.group(2))

    # ── Reconstruct allocated cells ──────────────────────────────────────────
    vue_map   = {v.virtual_id: v for v in vues}
    occupied  : set = set()
    alloc     : Dict[int, dict] = {}

    for vue_id, col_idx in selected_lam.items():
        cdata = cols_data.get(str(vue_id), {}).get(str(col_idx), {})
        if not cdata:
            continue
        t_arr  = cdata['t']
        f_arr  = cdata['f']
        mu_arr = cdata['mu']
        w_arr  = cdata['w']
        cells: set = set()
        for k in range(len(t_arr)):
            if w_arr[k] > 0 and t_arr[k] >= 0:
                cells |= _cells_of_assignment(
                    t_arr[k], f_arr[k], mu_arr[k], w_arr[k],
                    config, with_guard=False,
                )
        occupied |= cells
        alloc[vue_id] = {
            'total_prbs'  : cdata['total_prbs'],
            'cells'       : cells,
            'active_slots': sum(1 for w in w_arr if w > 0),
            'slice_id'    : vue_map[vue_id].sla.slice_id if vue_id in vue_map else '?',
            'n_prb_sla'   : vue_map[vue_id].n_prb        if vue_id in vue_map else 0,
        }

    total_cells  = config.n_time_cols * config.n_freq_rows
    n_occupied   = len(occupied)
    n_unoccupied = total_cells - n_occupied
    utilization  = 100.0 * n_occupied / total_cells

    # ── Header ───────────────────────────────────────────────────────────────
    print()
    print(DIV)
    print("  SOLUTION STATISTICS")
    if obj_val is not None:
        print(f"  Objective value : {obj_val:.4f}")
    print(DIV)

    # ── Per-VUE allocation ───────────────────────────────────────────────────
    print(f"\nVUE allocation  ({len(vues)} VUEs):")
    served = 0
    total_prbs_alloc = 0
    for vue in sorted(vues, key=lambda v: v.virtual_id):
        a = alloc.get(vue.virtual_id)
        if a and a['total_prbs'] > 0:
            served += 1
            total_prbs_alloc += a['total_prbs']
            pct = 100 * a['total_prbs'] / vue.n_prb
            tag = f"SERVED   {a['total_prbs']:>3d} PRBs  SLA={vue.n_prb} ({pct:.0f}%)  " \
                  f"{a['active_slots']} slot(s)  {len(a['cells'])} cells"
        else:
            tag = "NULL     0 PRBs  — SLA NOT met"
        print(f"  VUE {vue.virtual_id} ({vue.sla.slice_id:>5s}): {tag}")

    total_n_prb_sla = sum(vue.n_prb for vue in vues)
    sla_rate = 100 * served / len(vues)
    print(f"\n  {served}/{len(vues)} VUEs served  |  "
          f"total PRBs allocated = {total_prbs_alloc}  |  "
          f"SLA rate = {sla_rate:.0f}%")

    # ── Overall grid utilisation ─────────────────────────────────────────────
    print(f"\n{DIV2}")
    print(f"Grid utilisation  "
          f"({config.n_time_cols} cols × {config.n_freq_rows} rows = {total_cells} cells):")
    print(f"  Occupied   : {n_occupied:>5d} cells  ({utilization:.1f}%)")
    print(f"  Unoccupied : {n_unoccupied:>5d} cells  ({100-utilization:.1f}%)")

    if not occupied:
        print("\n  [No cells allocated — nothing to analyse]")
        print(DIV)
        return

    # ── Bounding-box analysis ────────────────────────────────────────────────
    t_vals       = [t for t, _ in occupied]
    f_vals       = [f for _, f in occupied]
    min_t, max_t = min(t_vals), max(t_vals)
    min_f, max_f = min(f_vals), max(f_vals)
    bbox_t       = max_t - min_t + 1
    bbox_f       = max_f - min_f + 1
    bbox_total   = bbox_t * bbox_f
    bbox_eff     = 100.0 * n_occupied / bbox_total
    internal_gap = bbox_total - n_occupied

    print(f"\nActive bounding box  (cols {min_t}–{max_t}, rows {min_f}–{max_f}):")
    print(f"  Bounding box      : {bbox_t} cols × {bbox_f} rows = {bbox_total} cells")
    print(f"  Packed efficiency : {n_occupied}/{bbox_total} = {bbox_eff:.1f}%")
    print(f"  Internal gaps     : {internal_gap} cells  ({100-bbox_eff:.1f}% of bbox)  ← primary gap metric")
    tail_t = config.n_time_cols - (max_t + 1)
    tail_f = config.n_freq_rows - (max_f + 1)
    print(f"  Unused time tail  : {tail_t} cols after col {max_t}  ({100*tail_t/config.n_time_cols:.1f}% of timeline)")
    print(f"  Unused freq tail  : {tail_f} rows above row {max_f}  ({100*tail_f/config.n_freq_rows:.1f}% of bandwidth)")

    # ── Dimension-wise utilisation histograms ────────────────────────────────
    t_counter = Counter(t for t, _ in occupied)
    f_counter = Counter(f for _, f in occupied)

    empty_t = [t for t in range(config.n_time_cols) if t_counter[t] == 0]
    empty_f = [f for f in range(config.n_freq_rows) if f_counter[f] == 0]
    used_t  = config.n_time_cols - len(empty_t)
    used_f  = config.n_freq_rows - len(empty_f)

    print(f"\nTime-column utilisation  ({used_t}/{config.n_time_cols} cols used):")
    if empty_t:
        grps = _group_consecutive(empty_t)
        print(f"  Empty cols : {grps}  ({len(empty_t)} cols, {100*len(empty_t)/config.n_time_cols:.1f}%)")
        # Characterise each gap
        for g in grps:
            if '-' in g:
                a, b = map(int, g.split('-'))
                cause = "pre-allocation head" if b < min_t else \
                        "post-allocation tail" if a > max_t else \
                        "internal hole ← indicates fragmentation"
            else:
                a = int(g)
                cause = "pre-allocation head" if a < min_t else \
                        "post-allocation tail" if a > max_t else \
                        "internal hole ← indicates fragmentation"
            print(f"    cols [{g}]: {cause}")
    else:
        print(f"  All {config.n_time_cols} time columns used — no temporal gaps")

    print(f"\nFreq-row utilisation  ({used_f}/{config.n_freq_rows} rows used):")
    if empty_f:
        grps = _group_consecutive(empty_f)
        print(f"  Empty rows : {grps}  ({len(empty_f)} rows, {100*len(empty_f)/config.n_freq_rows:.1f}%)")
        for g in grps:
            a = int(g.split('-')[0])
            b = int(g.split('-')[-1])
            cause = "head (below lowest allocation)" if b < min_f else \
                    "tail (above highest allocation)" if a > max_f else \
                    "guard-band / numerology boundary ← fragmentation"
            print(f"    rows [{g}]: {cause}")
    else:
        print(f"  All {config.n_freq_rows} frequency rows used — no frequency gaps")

    # ── Internal gap region analysis ─────────────────────────────────────────
    active_zone_gaps = {
        (t, f)
        for t in range(min_t, max_t + 1)
        for f in range(min_f, max_f + 1)
        if (t, f) not in occupied
    }

    frag_idx = 0.0
    print(f"\nInternal gap analysis (within active bbox, {len(active_zone_gaps)} gap cells):")
    if active_zone_gaps:
        gap_comps = _connected_components(active_zone_gaps)
        sizes     = sorted([len(c) for c in gap_comps], reverse=True)
        frag_idx  = 1.0 - sizes[0] / len(active_zone_gaps) if active_zone_gaps else 0.0
        print(f"  Disconnected gap regions : {len(gap_comps)}")
        print(f"  Region sizes (desc)      : {sizes[:15]}{'...' if len(sizes) > 15 else ''}")
        print(f"  Largest gap block        : {sizes[0]} cells")
        print(f"  Fragmentation index      : {frag_idx:.3f}  "
              f"(0 = one solid block, 1 = highly scattered)")
        # Locate the largest gap block
        for comp in gap_comps:
            if len(comp) == sizes[0]:
                ct = [c[0] for c in comp]
                cf = [c[1] for c in comp]
                print(f"  Largest gap location     : cols {min(ct)}–{max(ct)}, "
                      f"rows {min(cf)}–{max(cf)}")
                break
    else:
        print("  No internal gaps — active zone is fully packed")

    # ── Actionable metrics ───────────────────────────────────────────────────
    print(f"\n{DIV2}")
    print("Metrics for objective tuning:")
    print(f"  packed_efficiency  = {bbox_eff:.1f}%   "
          f"← penalise with −α·(internal_gap_cells) in master objective")
    print(f"  temporal_tail      = {tail_t} idle cols  "
          f"← penalise with −β·last_active_time_col")
    print(f"  freq_tail          = {tail_f} idle rows  "
          f"← penalise with −γ·max_active_freq_row")
    print(f"  fragmentation_idx  = {frag_idx:.3f}       "
          f"← high value → many small holes; improve seeds or add compactness bonus")
    print(f"  sla_rate           = {sla_rate:.0f}%       "
          f"← SERVE_BONUS already handles this")
    print()
    print(DIV)


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
    if not path.endswith(".json"):
        # Treat as directory: append the auto-generated filename
        path = os.path.join(path, f"cfg-bw-{BW/1_000_000}M-time-ms-{time_horizon_ms}-mcs-{mcs}-K-{K}-ue-{ue_count}-embb-thr-{embb_mbps}-embb-lat-{embb_latency_ms}-urllc-thr-{urllc_mbps}-urllc-lat-{urllc_latency_ms}.json")
    with open(path, "w") as f:
        json.dump(sample, f, indent=4)
    print(f"Sample config written → {path}")