# ============================================================================
# Column Generation Solver — 5G NR Radio Resource Allocation
# ============================================================================
# Self-contained module: no imports from .configs or .utils.
# Structure mirrors coord_search.py exactly (dataclasses → helpers → class → CLI).
#
# Three-phase decomposition:
#   Phase 0  _generate_initial_columns()  — single-UE MIPs, no interaction
#   Phase 1  _run_cg_loop()               — LP master → dual prices → pricing
#   Phase 2  _solve_integer_master()      — Set Partitioning MIP over column pool
#
# CLI usage:
#   python -m src.column_gen \
#       --config  configs/<cfg>.json \
#       --output  approx_sol/ \
#       2>&1 | tee logs/logs_column_gen.log
# ============================================================================

import argparse
import os
import sys
import time
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import pulp
except ImportError:
    print("ERROR: PuLP not installed. Run: pip install pulp --break-system-packages")
    sys.exit(1)


# ============================================================================
# Core 5G NR Grid Utilities (Self-Contained)
# ============================================================================

def E(mu: int, mu_max: int) -> int:
    """
    Slot duration of numerology µ in integer grid column units.

    E(µ) = 2^(µmax − µ)

    With µmax = 3:
        µ=0 → 8 cols   (1 ms    slot)
        µ=1 → 4 cols   (0.5 ms  slot)
        µ=2 → 2 cols   (0.25 ms slot)
        µ=3 → 1 col    (0.125 ms slot)
    """
    return 2 ** (mu_max - mu)


def G_rows(mu: int) -> int:
    """
    Frequency width of ONE PRB at numerology µ in integer grid row units.

    G_rows(µ) = 2^µ  (delta_F = 180 kHz per row at µ=0)

    µ=0 → 1 row, µ=1 → 2 rows, µ=2 → 4 rows, µ=3 → 8 rows.
    """
    return 2 ** mu


def max_omega(mu: int, n_freq_rows: int) -> int:
    """Upper bound on ω (contiguous PRBs) at numerology µ."""
    return n_freq_rows // G_rows(mu)


# ============================================================================
# Data Classes (Self-Contained)
# ============================================================================

@dataclass
class SystemConfig:
    """System-level parameters defining the resource grid."""
    bandwidth_hz: float
    delta_t_ms:   float
    mu_max:       int
    K:            int
    numerologies: List[int]
    G_guard:      int   = 1       # guard band rows between BWPs of different numerology

    # Derived — populated by __post_init__
    delta_T:      float = field(init=False)
    n_time_cols:  int   = field(init=False)
    n_freq_rows:  int   = field(init=False)
    Omega:        int   = field(init=False)

    def __post_init__(self):
        self.delta_T     = 2 ** (-self.mu_max)                    # ms per grid column
        self.n_time_cols = round(self.delta_t_ms / self.delta_T)
        self.n_freq_rows = int(self.bandwidth_hz / (12 * 15 * 1000))  # rows at δF=180 kHz
        self.Omega       = self.n_freq_rows


@dataclass
class VirtualUE:
    """
    A virtual UE — one (physical UE, slice) pair.

    Matches the flattened representation used in coord_search.py so the two
    solvers can share the same JSON configs and load_and_build() output.
    """
    virtual_id:    int
    group_id:      str   # equals the parent physical UE's ue_id
    n_prb:         int   # minimum PRBs to satisfy throughput SLA
    latency_slots: int   # latency deadline in grid column units
    sla_slice_id:  str   # e.g. "embb" | "urllc" | "mmtc"


@dataclass
class Column:
    """
    A feasible allocation plan for a single virtual UE.

    One Column represents a complete K-slot schedule:
        t[k]  : start time column for slot k  (-1 = unassigned)
        f[k]  : start freq  row  for slot k  (-1 = unassigned)
        mu[k] : numerology for slot k          (None = unassigned)
        w[k]  : contiguous PRBs at slot k     (0 = unassigned)

    cells stores RAW resource cells (no guard band) for the master problem's
    resource non-conflict constraints.  Guard-band enforcement is handled
    implicitly through shadow costs in the pricing subproblems.
    """
    vue_id:     int
    t:          List[int]
    f:          List[int]
    mu:         List[Optional[int]]
    w:          List[int]
    total_prbs: int
    cells:      frozenset   # frozenset[tuple[int, int]]  — raw (time_col, freq_row)


# ============================================================================
# MCS Table & N_PRB Computation (Self-Contained)
# ============================================================================

# 3GPP TS 38.214 Table 5.1.3.1 — MCS index → (modulation order Q, code rate r)
_MCS_TABLE: Dict[int, Tuple[int, float]] = {
    0:  (2, 0.1172),  1:  (2, 0.1533),  2:  (2, 0.1885),  3:  (2, 0.2451),
    4:  (2, 0.3008),  5:  (2, 0.3770),  6:  (2, 0.4492),  7:  (2, 0.5547),
    8:  (2, 0.6016),  9:  (2, 0.6504), 10:  (4, 0.3672), 11:  (4, 0.4238),
    12: (4, 0.4785), 13:  (4, 0.5352), 14:  (4, 0.6016), 15:  (4, 0.6426),
    16: (4, 0.6953), 17:  (6, 0.4385), 18:  (6, 0.4951), 19:  (6, 0.5537),
    20: (6, 0.6016), 21:  (6, 0.6504), 22:  (6, 0.7021), 23:  (6, 0.7539),
    24: (8, 0.5527), 25:  (8, 0.6016), 26:  (8, 0.6504), 27:  (8, 0.7021),
    28: (8, 0.7539),
}
_RE_PER_PRB: int = 168   # 12 subcarriers × 14 OFDM symbols


def _compute_n_prb(throughput_mbps: float, delta_t_ms: float, mcs: int) -> int:
    """
    Minimum PRBs to satisfy a throughput SLA.

    TBS_per_PRB = RE_per_PRB * Q * r
    total_bits  = throughput_mbps * delta_t_ms * 1000   (Mbps * ms = kbits → ×1000 = bits)
    N_PRB       = ceil(total_bits / TBS_per_PRB)
    """
    Q, r        = _MCS_TABLE.get(mcs, (2, 0.1172))
    tbs_per_prb = _RE_PER_PRB * Q * r
    total_bits  = throughput_mbps * delta_t_ms * 1000
    return max(1, math.ceil(total_bits / tbs_per_prb))


# ============================================================================
# Config Loader & VUE Builder (Self-Contained)
# ============================================================================

def load_and_build(path: str) -> Tuple[SystemConfig, List[VirtualUE]]:
    """
    Parse a JSON config into (SystemConfig, list[VirtualUE]).

    Identical interface to coord_search.load_and_build() so the two solvers
    accept the same config files without modification.

    Expected JSON structure:
    {
        "system": { "bandwidth_hz": …, "delta_t_ms": …, "mu_max": …,
                    "K": …, "numerologies": […], "G_guard": … },
        "ues": [
            { "ue_id": "UE1", "mcs": 26,
              "slices": [
                  { "slice_id": "embb",  "throughput_mbps": 13.0, "latency_ms": 10.0 },
                  { "slice_id": "urllc", "throughput_mbps": 0.7,  "latency_ms": 0.25 }
              ]
            }, …
        ]
    }
    """
    with open(path) as f:
        raw = json.load(f)

    s = raw["system"]
    config = SystemConfig(
        bandwidth_hz = float(s["bandwidth_hz"]),
        delta_t_ms   = float(s["delta_t_ms"]),
        mu_max       = int(s["mu_max"]),
        K            = int(s["K"]),
        numerologies = [int(m) for m in s.get("numerologies", [0, 1, 2, 3])],
        G_guard      = int(s.get("G_guard", 1)),
    )

    vues: List[VirtualUE] = []
    vid = 0
    for u in raw["ues"]:
        mcs = int(u["mcs"])
        for sl in u["slices"]:
            n_prb     = _compute_n_prb(float(sl["throughput_mbps"]), config.delta_t_ms, mcs)
            lat_slots = round(float(sl["latency_ms"]) / config.delta_T)
            vues.append(VirtualUE(
                virtual_id    = vid,
                group_id      = str(u["ue_id"]),
                n_prb         = n_prb,
                latency_slots = lat_slots,
                sla_slice_id  = str(sl["slice_id"]),
            ))
            vid += 1

    return config, vues


# ============================================================================
# Standalone Helpers (shared by all three phases)
# ============================================================================

def _cells_of_assignment(
    t_val: int, f_val: int, mu: int, w: int,
    config: SystemConfig,
    with_guard: bool = True,
) -> frozenset:
    """
    Compute the set of resource cells covered by a single slot assignment.

    with_guard=True  adds G_guard frequency rows of buffer on each side.
    with_guard=False gives the raw footprint used for resource constraints.
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


def _get_valid_assignments(vue: VirtualUE, config: SystemConfig) -> List[Tuple[int, int, int, int]]:
    """
    Pre-compute every valid (t, f, mu, w) tuple for one slot of a VUE.

    Pre-filters:
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
            = 1 iff slot k is placed at (t, f) with numerology mu and PRB width w.

    Constraints:
        (5)  At most one (t, f, mu, w) per slot
        (11) Cascade: slot k+1 only if slot k is assigned
        (10) Time ordering: T_start(k+1) ≥ T_end(k) when k+1 assigned
             Linearisation: T_start(k+1) − PHI · active(k+1) ≥ T_end(k) − PHI
        (17) Throughput SLA: Σ w·A ∈ [n_prb, n_prb]  (pinned for init/pricing)

    Latency (16) and boundary constraints are pre-filtered into valid_asns
    so they need no explicit constraint rows.

    The objective is set by the caller (init → maximise PRBs; pricing → reduced cost).

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
        """Active-indicator for slot k: Σ_{t,f,mu,w} A[(k,t,f,mu,w)] ∈ {0, 1}."""
        return pulp.lpSum(A[(k, t, f, mu, w)] for (t, f, mu, w) in valid_asns)

    # (5) At most one assignment per slot
    for k in range(K):
        prob += (sum_A_slot(k) <= 1, f"c5_slot{k}")

    # (11) Cascade: slot k+1 can only be assigned if slot k is assigned
    for k in range(K - 1):
        prob += (sum_A_slot(k + 1) <= sum_A_slot(k), f"c11_casc{k}")

    # (10) Time ordering: T_start(k+1) ≥ T_end(k) when k+1 is assigned.
    # Big-M linearisation with PHI_T = n_time_cols:
    #   T_start(k+1) − PHI_T · active(k+1) ≥ T_end(k) − PHI_T
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

    # (17) Throughput SLA — pin to exactly n_prb.
    # Pinning (not just ≥) prevents a single VUE from monopolising the grid during
    # Phase 0, leaving room for other VUEs and avoiding LP degeneracy.
    if valid_asns:
        prb_sum = pulp.lpSum(
            w * A[(k, t_val, f_val, mu, w)]
            for k in range(K)
            for (t_val, f_val, mu, w) in valid_asns
        )
        prob += (prb_sum >= vue.n_prb, "c17_min_prb")
        prob += (prb_sum <= vue.n_prb, "c17_max_prb")

    return prob, A, valid_asns


def _extract_column(
    vue:        VirtualUE,
    config:     SystemConfig,
    A:          Dict[Tuple, pulp.LpVariable],
    valid_asns: List[Tuple],
) -> Column:
    """Extract a Column dataclass from the solved A decision variables."""
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
                    # raw cells only (no guard) — for resource constraints
                    all_cells.update(
                        _cells_of_assignment(t_val, f_val, mu, w, config, with_guard=False)
                    )
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


# ============================================================================
# Solver Class
# ============================================================================

class ColumnGenerationSolver:
    """
    Column Generation solver for 5G NR resource allocation.

    Eliminates the O(N²K²|M|) interaction variables of the monolithic MILP by
    working with complete single-UE allocation plans (columns) as the decision
    unit.  Inter-UE coupling is only through shared resource cells — no big-M
    values appear in the master problem.

    Args:
        config   : SystemConfig instance.
        vues     : list of VirtualUE instances.
        alpha    : per-slice priority weights (applied in master objective).
        max_iter : maximum Phase 1 CG loop iterations.

    Phases:
        0  _generate_initial_columns() — single-UE MIPs, no interaction
        1  _run_cg_loop()              — LP master → dual prices → pricing
        2  _solve_integer_master()     — Set Partitioning MIP over column pool
    """

    def __init__(
        self,
        config:   SystemConfig,
        vues:     List[VirtualUE],
        alpha:    Dict[str, float] = None,
        max_iter: int = 200,
    ):
        self.config   = config
        self.vues     = vues
        self.alpha    = alpha or {"urllc": 2.0, "embb": 1.0}
        self.max_iter = max_iter
        # column pool: {vue_id → [Column, …]}
        self.columns: Dict[int, List[Column]] = {v.virtual_id: [] for v in vues}

    # ── Phase 0 : initial columns ────────────────────────────────────────────

    def _generate_initial_columns(self):
        """
        Seed each VUE with a null column and (if feasible) an initial column.

        The null column (total_prbs=0, cells=∅) guarantees the master problem
        is always feasible: if every real column for some VUE is blocked by
        resource conflicts, the null column can still be selected.
        """
        logging.info(
            "── Phase 0: Initial columns (single-UE MIPs, no interaction) ───"
        )
        for vue in self.vues:
            null_col = Column(
                vue_id = vue.virtual_id,
                t      = [-1]   * self.config.K,
                f      = [-1]   * self.config.K,
                mu     = [None] * self.config.K,
                w      = [0]    * self.config.K,
                total_prbs = 0,
                cells      = frozenset(),
            )

            prob, A, valid_asns = _build_single_ue_prob(vue, self.config, "init")
            if not A:
                logging.warning(
                    f"  VUE {vue.virtual_id:3d} ({vue.group_id} {vue.sla_slice_id:5s}): "
                    f"no valid assignments — null column only."
                )
                self.columns[vue.virtual_id] = [null_col]
                continue

            # Phase 0 objective: maximise total PRBs
            prob += pulp.lpSum(
                w * A[(k, t_val, f_val, mu, w)]
                for k in range(self.config.K)
                for (t_val, f_val, mu, w) in valid_asns
            )
            prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))

            if pulp.value(prob.objective) is None:
                logging.warning(
                    f"  VUE {vue.virtual_id:3d} ({vue.group_id} {vue.sla_slice_id:5s}): "
                    f"infeasible single-UE MIP — null column only."
                )
                self.columns[vue.virtual_id] = [null_col]
            else:
                col = _extract_column(vue, self.config, A, valid_asns)
                logging.info(
                    f"  VUE {vue.virtual_id:3d} ({vue.group_id} {vue.sla_slice_id:5s}):  "
                    f"{col.total_prbs:3d} PRBs  {len(col.cells):3d} cells"
                )
                self.columns[vue.virtual_id] = [null_col, col]

    # ── Phase 1 : CG loop helpers ────────────────────────────────────────────

    def _build_master_problem(
        self, integer: bool = False,
    ) -> Tuple[
        pulp.LpProblem,
        Dict[Tuple[int, int], pulp.LpVariable],
        Dict[int, str],
        Dict[Tuple[int, int], str],
    ]:
        """
        Build the LP or IP master problem from the current column pool.

        Master:
            max  Σ_{i,c}  min(total_prbs(c), n_prb_i) · λ_{i,c}
            s.t. Σ_c λ_{i,c} = 1                   ∀i  (convexity)
                 Σ_{i,c: r ∈ c.cells} λ_{i,c} ≤ 1  ∀r  (resource non-conflict)
                 λ ∈ [0,1]  (LP)  or  {0,1}  (IP)

        The PRB cap min(total_prbs, n_prb_i) prevents any single VUE from
        monopolising the grid: distributing resources across N VUEs scores N×
        higher than giving everything to one VUE.

        Resource constraints use RAW cells (no guard band); guard-band
        enforcement is handled implicitly through shadow costs in pricing.

        Returns: (prob, lam, conv_cname, res_cname)
        """
        cat = pulp.constants.LpBinary if integer else pulp.constants.LpContinuous

        prob = pulp.LpProblem(
            "CG_Integer_Master" if integer else "CG_Master_LP",
            pulp.LpMaximize,
        )

        # Decision variables λ_{i, c_idx}
        lam: Dict[Tuple[int, int], pulp.LpVariable] = {}
        for vue in self.vues:
            i = vue.virtual_id
            for c_idx in range(len(self.columns.get(i, []))):
                lam[(i, c_idx)] = pulp.LpVariable(
                    f"lam_{i}_{c_idx}", lowBound=0, upBound=1, cat=cat,
                )

        # Objective: capped per-VUE PRB contribution
        vue_n_prb = {v.virtual_id: v.n_prb for v in self.vues}
        prob += pulp.lpSum(
            min(self.columns[v.virtual_id][c_idx].total_prbs, vue_n_prb[v.virtual_id])
            * lam[(v.virtual_id, c_idx)]
            for v in self.vues
            for c_idx in range(len(self.columns.get(v.virtual_id, [])))
        )

        # Convexity: exactly one column selected per VUE
        conv_cname: Dict[int, str] = {}
        for vue in self.vues:
            i    = vue.virtual_id
            cols = self.columns.get(i, [])
            if not cols:
                continue
            cname = f"conv_{i}"
            prob += (
                pulp.lpSum(lam[(i, c_idx)] for c_idx in range(len(cols))) == 1,
                cname,
            )
            conv_cname[i] = cname

        # Resource non-conflict: collect all raw cells
        all_cells: set = set()
        for vue in self.vues:
            for col in self.columns.get(vue.virtual_id, []):
                all_cells.update(col.cells)

        # Only add a constraint when ≥ 2 columns compete for the same cell
        res_cname: Dict[Tuple[int, int], str] = {}
        for cell in all_cells:
            users = [
                (vue.virtual_id, c_idx)
                for vue in self.vues
                for c_idx, col in enumerate(self.columns.get(vue.virtual_id, []))
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

        return prob, lam, conv_cname, res_cname

    def _solve_lp_master(self) -> Tuple[float, Dict, Dict]:
        """
        Solve the LP relaxation of the master problem.

        Returns:
            obj_val  : LP objective (upper bound on integer optimum)
            pi       : {resource_cell → dual price}  (shadow price ≥ 0 for ≤ constraint)
            mu_dual  : {vue_id → dual price of convexity constraint}  (any sign)
        """
        prob, lam, conv_cname, res_cname = self._build_master_problem(integer=False)
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=120))

        obj_val = pulp.value(prob.objective) or 0.0

        pi: Dict[Tuple[int, int], float] = {}
        for cell, cname in res_cname.items():
            c       = prob.constraints.get(cname)
            pi[cell] = c.pi if (c is not None and c.pi is not None) else 0.0

        mu_dual: Dict[int, float] = {}
        for vue in self.vues:
            i     = vue.virtual_id
            cname = conv_cname.get(i)
            c     = prob.constraints.get(cname) if cname else None
            mu_dual[i] = c.pi if (c is not None and c.pi is not None) else 0.0

        return obj_val, pi, mu_dual

    def _solve_pricing(
        self,
        vue:     VirtualUE,
        pi:      Dict[Tuple[int, int], float],
        mu_dual: float,
    ) -> Optional[Column]:
        """
        Pricing subproblem: find the column with maximum reduced cost for VUE i.

        Reduced cost:
            RC(c) = obj(c) − Σ_r π_r · 1[r ∈ c.cells] − μ_i

        Maximised objective (μ_i is constant, checked externally):
            Σ_{k,asn} (w − shadow(asn)) · A[(k,asn)]

        where shadow(asn) = Σ_{r ∈ raw_cells(asn)} π_r

        Returns a new Column if RC > 1e-6, otherwise None.
        """
        prob, A, valid_asns = _build_single_ue_prob(vue, self.config, "pricing")
        if not A:
            return None

        # Pre-compute shadow cost per valid assignment using RAW cells (no guard),
        # consistent with the master's resource constraints.
        shadow: Dict[Tuple, float] = {
            asn: sum(
                pi.get(cell, 0.0)
                for cell in _cells_of_assignment(*asn, self.config, with_guard=False)
            )
            for asn in valid_asns
        }

        prob += pulp.lpSum(
            (w - shadow[(t_val, f_val, mu, w)]) * A[(k, t_val, f_val, mu, w)]
            for k in range(self.config.K)
            for (t_val, f_val, mu, w) in valid_asns
        )
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))

        pricing_obj = pulp.value(prob.objective)
        if pricing_obj is None or (pricing_obj - mu_dual) <= 1e-6:
            return None   # no improving column

        return _extract_column(vue, self.config, A, valid_asns)

    def _run_cg_loop(self):
        """
        Phase 1: iterate LP solve + pricing until no improving columns are found.

        Termination: when no pricing subproblem returns RC > 1e-6, the LP
        relaxation is optimal for the current column pool (i.e., all reduced
        costs are non-positive).
        """
        logging.info(
            f"── Phase 1: Column Generation loop (max {self.max_iter} iters) ──"
        )
        for iteration in range(self.max_iter):
            lp_obj, pi, mu_dual = self._solve_lp_master()

            new_cols_found = 0
            for vue in self.vues:
                new_col = self._solve_pricing(vue, pi, mu_dual.get(vue.virtual_id, 0.0))
                if new_col is not None:
                    self.columns[vue.virtual_id].append(new_col)
                    new_cols_found += 1

            total_cols = sum(len(v) for v in self.columns.values())
            logging.info(
                f"  Iter {iteration:3d}: LP obj = {lp_obj:8.3f}  "
                f"new cols = {new_cols_found:3d}  total cols = {total_cols}"
            )

            if new_cols_found == 0:
                logging.info(f"  LP optimal reached at iteration {iteration}.")
                break

    # ── Phase 2 : integer master ─────────────────────────────────────────────

    def _solve_integer_master(self) -> Tuple[float, Dict[int, Optional[Column]]]:
        """
        Phase 2: Set Partitioning MIP over the full column pool.

        Typically very fast given the tight LP relaxation from Phase 1
        (expected integrality gap < 5%).

        Returns:
            int_obj  : integer objective value (total capped PRBs)
            selected : {vue_id → chosen Column (or None if unscheduled)}
        """
        total_cols = sum(len(v) for v in self.columns.values())
        logging.info(
            f"── Phase 2: Integer master (Set Partitioning)  "
            f"—  {total_cols} columns ──"
        )

        prob, lam, conv_cname, res_cname = self._build_master_problem(integer=True)
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=300))

        int_obj = pulp.value(prob.objective) or 0.0

        selected: Dict[int, Optional[Column]] = {}
        for vue in self.vues:
            i    = vue.virtual_id
            cols = self.columns.get(i, [])
            best = None
            for c_idx, col in enumerate(cols):
                val = pulp.value(lam.get((i, c_idx)))
                if val is not None and val > 0.5:
                    best = col
                    break
            selected[i] = best

        # Per-VUE summary log
        for vue in self.vues:
            col = selected.get(vue.virtual_id)
            if col and col.total_prbs > 0:
                active = [
                    (col.t[k], col.f[k], col.mu[k], col.w[k])
                    for k in range(self.config.K) if col.w[k] > 0
                ]
                logging.info(
                    f"  VUE {vue.virtual_id:3d} ({vue.group_id} {vue.sla_slice_id:5s}): "
                    f"{col.total_prbs:3d} PRBs  slots={active}"
                )
            else:
                logging.info(
                    f"  VUE {vue.virtual_id:3d} ({vue.group_id} {vue.sla_slice_id:5s}): "
                    f"  0 PRBs  (null / unscheduled)"
                )

        return int_obj, selected

    # ── Output ───────────────────────────────────────────────────────────────

    def _write_output(
        self,
        path:     str,
        selected: Dict[int, Optional[Column]],
        int_obj:  float,
    ):
        """
        Write the solution in the same key–value text format as coord_search.py.

        Format:
            Status: ColumnGeneration
            Objective: <float>

            T_<vid>_<k> <t>
            F_<vid>_<k> <f>
            X_<vid>_<k>_<mu>_<w> 1
            …
        """
        with open(path, "w") as fh:
            fh.write("Status: ColumnGeneration\n")
            fh.write(f"Objective: {int_obj:.6f}\n\n")
            for vue in self.vues:
                col = selected.get(vue.virtual_id)
                if col is None:
                    continue
                for k in range(self.config.K):
                    if col.w[k] > 0 and col.t[k] >= 0:
                        fh.write(f"T_{vue.virtual_id}_{k} {col.t[k]}\n")
                        fh.write(f"F_{vue.virtual_id}_{k} {col.f[k]}\n")
                        fh.write(f"X_{vue.virtual_id}_{k}_{col.mu[k]}_{col.w[k]} 1\n")

    # ── Main entry point ─────────────────────────────────────────────────────

    def solve(self, output_path: str):
        """
        Run all three phases and write the solution to output_path.

        Args:
            output_path : full path to the .txt output file.
        """
        logging.info(
            f"Starting ColumnGeneration solver.  "
            f"Grid: {self.config.n_time_cols}t × {self.config.n_freq_rows}f  |  "
            f"VUEs: {len(self.vues)}  |  K={self.config.K}  |  "
            f"max_iter={self.max_iter}"
        )

        self._generate_initial_columns()
        self._run_cg_loop()
        int_obj, selected = self._solve_integer_master()

        logging.info(f"Final integer objective: {int_obj:.4f} PRBs")
        self._write_output(output_path, selected, int_obj)


# ============================================================================
# CLI  (mirrors coord_search.py argument names exactly)
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Column Generation solver for 5G NR resource allocation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for the solution .txt file.",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=200,
        help="Max Phase 1 CG loop iterations.",
    )
    parser.add_argument(
        "--alpha-urllc", type=float, default=2.0,
        help="Priority weight for uRLLC slices (master objective).",
    )
    parser.add_argument(
        "--alpha-embb", type=float, default=1.0,
        help="Priority weight for eMBB slices (master objective).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Logging verbosity.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level  = getattr(logging, args.log_level),
        format = "%(asctime)s %(message)s",
    )

    start_time = time.time()

    cfg, vues = load_and_build(args.config)
    alpha     = {"urllc": args.alpha_urllc, "embb": args.alpha_embb}

    solver = ColumnGenerationSolver(
        config   = cfg,
        vues     = vues,
        alpha    = alpha,
        max_iter = args.max_iterations,
    )

    os.makedirs(args.output, exist_ok=True)
    filename    = os.path.splitext(os.path.basename(args.config))[0]
    output_file = os.path.join(args.output, f"{filename}.txt")

    solver.solve(output_file)

    print(f"Solution written to: {output_file}")
    print(f"Total solve time: {time.time() - start_time:.2f} seconds")