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
#   python -m column_gen \
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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

try:
    import pulp
except ImportError:
    print("ERROR: PuLP not installed. Run: pip install pulp --break-system-packages")
    sys.exit(1)


# ============================================================================
# Solver Selection (HiGHS preferred, CBC fallback)
# ============================================================================

def _highs_available() -> bool:
    # Prefer HiGHS_CMD (binary in PATH — installed via conda-forge highs).
    try:
        if pulp.HiGHS_CMD(msg=0).available():
            return True
    except Exception:
        pass
    # Fallback: highspy Python API (pip install highspy).
    # PuLP 3.x exposes this as HiGHSpy when highspy is installed.
    try:
        solver = pulp.HiGHSpy(msg=0)
        return solver.available()
    except Exception:
        pass
    return False


def _highs_solver(threads: int, time_limit: int) -> "pulp.LpSolver":
    """Return the best available HiGHS solver variant."""
    try:
        if pulp.HiGHS_CMD(msg=0).available():
            return pulp.HiGHS_CMD(msg=0, timeLimit=time_limit, threads=threads)
    except Exception:
        pass
    # highspy Python API — no binary needed.
    try:
        return pulp.HiGHSpy(msg=0, timeLimit=time_limit, threads=threads)
    except Exception:
        pass
    raise RuntimeError("HiGHS reported available but no solver could be constructed")


_USE_HIGHS: bool = _highs_available()


def _make_solver(threads: int = 1, time_limit: int = 60) -> "pulp.LpSolver":
    """
    Return HiGHS solver if available, otherwise CBC.

    threads   : number of solver threads.
                Use > 1 only for the LP master and integer master (sequential,
                large problems).  Keep threads=1 for pricing/init workers since
                they already run as parallel processes.
    time_limit: wall-clock limit in seconds.
    """
    if _USE_HIGHS:
        return _highs_solver(threads, time_limit)
    return pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, threads=threads)


# ============================================================================
# MPI Initialisation  (graceful fallback when mpi4py is absent)
# ============================================================================
#
# Import is attempted once at module load.  If mpi4py is not installed the
# solver falls back silently to the existing ProcessPoolExecutor path.
#
# Usage (4 nodes, e.g. via SLURM srun or plain mpirun):
#   mpirun -n 4 python -m column_gen --config <cfg>.json --output approx_sol/
#
# When _MPI_SIZE == 1 the code is identical to non-MPI execution — no
# overhead, no changed behaviour.

try:
    from mpi4py import MPI as _MPI
    _COMM     = _MPI.COMM_WORLD
    _MPI_RANK: int  = _COMM.Get_rank()
    _MPI_SIZE: int  = _COMM.Get_size()
    _USE_MPI: bool  = _MPI_SIZE > 1
except ImportError:
    _COMM     = None
    # Fall back to SLURM env vars so non-zero ranks don't run as master when
    # mpi4py is absent but srun has launched multiple tasks.
    _MPI_RANK = int(os.environ.get("SLURM_PROCID", 0))
    _MPI_SIZE = int(os.environ.get("SLURM_NTASKS", 1))
    _USE_MPI  = False


def _mpi_worker_loop(config: "SystemConfig", my_vues: "List[VirtualUE]") -> None:
    """
    MPI worker loop for ranks > 0.  Two phases:

    Phase 0 (once):
        Compute initial columns for local VUEs using a thread pool, then
        gather results to rank 0.  Threads are safe here because prob.solve()
        calls CBC/HiGHS as an external subprocess — subprocess.wait() releases
        the GIL, allowing true concurrency without fork+MPI danger.

    Phase 1 (per CG iteration):
        1. Receive broadcast dict {"pi": …, "mu_dual": …} from rank 0.
        2. Solve pricing for all local VUEs concurrently (ThreadPoolExecutor).
        3. Gather {vue_id → Column | None} back to rank 0.

    Termination:
        Rank 0 broadcasts None when CG has converged; workers break and return.
    """
    my_valid_asns: Dict[int, List[Tuple]] = {
        v.virtual_id: _get_valid_assignments(v, config) for v in my_vues
    }
    n_local   = len(my_vues)
    n_threads = min(n_local, os.cpu_count() or 1)

    logging.info(
        f"[rank {_MPI_RANK}] pricing worker ready — "
        f"{n_local} VUEs: {[v.virtual_id for v in my_vues]}"
    )

    # ── Phase 0: initial columns for local VUEs ───────────────────────────────
    # Threads release the GIL during CBC subprocess — safe with MPI.
    phase0_results: Dict[int, Tuple] = {}
    init_args = [(vue, config, my_valid_asns[vue.virtual_id]) for vue in my_vues]

    if n_threads > 1:
        with ThreadPoolExecutor(max_workers=n_threads) as tp:
            for vid, col, tb, ts, tt in tp.map(_init_worker, init_args):
                phase0_results[vid] = (col, tb, ts, tt)
    else:
        for args in init_args:
            vid, col, tb, ts, tt = _init_worker(args)
            phase0_results[vid] = (col, tb, ts, tt)

    # Synchronise with rank 0's _generate_initial_columns gather call.
    _COMM.gather(phase0_results, root=0)

    # ── Phase 1: pricing loop ─────────────────────────────────────────────────
    tpool = ThreadPoolExecutor(max_workers=n_threads) if n_threads > 1 else None
    try:
        while True:
            msg = _COMM.bcast(None, root=0)
            if msg is None:
                logging.info(f"[rank {_MPI_RANK}] received termination signal — exiting.")
                break

            pi: Dict[Tuple[int, int], float] = msg["pi"]
            mu_dual: Dict[int, float]        = msg["mu_dual"]

            pricing_args = [
                (vue, config, pi, mu_dual.get(vue.virtual_id, 0.0),
                 my_valid_asns[vue.virtual_id])
                for vue in my_vues
            ]
            results: Dict[int, Optional[Column]] = {}

            if tpool is not None:
                for vid, new_col, elapsed in tpool.map(_pricing_worker, pricing_args):
                    results[vid] = new_col
                    logging.debug(
                        f"[rank {_MPI_RANK}] VUE {vid}: "
                        f"{'new col' if new_col else 'no col'}  [{elapsed:.3f}s]"
                    )
            else:
                for args in pricing_args:
                    vid, new_col, elapsed = _pricing_worker(args)
                    results[vid] = new_col
                    logging.debug(
                        f"[rank {_MPI_RANK}] VUE {vid}: "
                        f"{'new col' if new_col else 'no col'}  [{elapsed:.3f}s]"
                    )

            _COMM.gather(results, root=0)
    finally:
        if tpool is not None:
            tpool.shutdown(wait=True)


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
    vue:                VirtualUE,
    config:             SystemConfig,
    name_prefix:        str  = "single_ue",
    pin_prbs:           bool = True,
    valid_asns_override: Optional[List[Tuple]] = None,
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
        (17) Throughput SLA floor: Σ w·A ≥ n_prb  (always)
             With pin_prbs=True  : also Σ w·A ≤ n_prb  (used for CG phases 0/1)
             With pin_prbs=False : no ceiling          (used for Phase 3 gap-fill)

    pin_prbs=True  pins exactly to n_prb for CG phases 0 & 1.  This is necessary
    so that Phase 0 does not immediately generate a "grab all rows" column that
    monopolises the LP, and so the pricing subproblem only adds columns that the
    master can use fairly.

    pin_prbs=False removes the ceiling so the gap-fill phase can find columns
    that use all available free spectrum.

    valid_asns_override: if provided, skip the internal _get_valid_assignments
        call and use this pre-computed list directly.  Callers that cache
        valid_asns in self._valid_asns should always pass it here to avoid
        recomputing the same list on every call.

    Latency (16) and boundary constraints are pre-filtered into valid_asns.
    The objective is set by the caller.

    Returns: (prob, A_vars, valid_asns)
    """
    K          = config.K
    valid_asns = (
        valid_asns_override
        if valid_asns_override is not None
        else _get_valid_assignments(vue, config)
    )

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

    # (17) Throughput SLA — floor always enforced; ceiling only when pin_prbs=True.
    if valid_asns:
        prb_sum = pulp.lpSum(
            w * A[(k, t_val, f_val, mu, w)]
            for k in range(K)
            for (t_val, f_val, mu, w) in valid_asns
        )
        prob += (prb_sum >= vue.n_prb, "c17_min_prb")
        if pin_prbs:
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


def _columns_guard_band_conflict(c1: Column, c2: Column, config: SystemConfig) -> bool:
    """
    Return True if selecting both columns would violate the guard band requirement.

    A guard band conflict requires ALL THREE of:
        (a) At least one pair of slots (k1 from c1, k2 from c2) that overlap in time,
        (b) Those two slots use DIFFERENT numerologies (INI would occur), AND
        (c) Their frequency ranges are within G_guard rows of each other.

    Same-numerology adjacent allocations are always fine — no guard band needed.
    Slots that don't overlap in time never cause INI regardless of frequency.

    This check is O(K²) per column pair.  With ≤ 20 cols/VUE and 6 VUEs the total
    work for the integer master is about 15 × 20² = 6 000 pair checks — negligible.
    """
    G      = config.G_guard
    mu_max = config.mu_max
    K      = config.K

    for k1 in range(K):
        if c1.w[k1] == 0 or c1.mu[k1] is None or c1.t[k1] < 0:
            continue
        mu1 = c1.mu[k1]
        t1  = c1.t[k1];  t1_end = t1 + E(mu1, mu_max)
        f1  = c1.f[k1];  f1_end = f1 + c1.w[k1] * G_rows(mu1)

        for k2 in range(K):
            if c2.w[k2] == 0 or c2.mu[k2] is None or c2.t[k2] < 0:
                continue
            mu2 = c2.mu[k2]
            if mu1 == mu2:
                continue   # same numerology → no INI → no guard band needed

            t2  = c2.t[k2];  t2_end = t2 + E(mu2, mu_max)
            if t1_end <= t2 or t2_end <= t1:
                continue   # no time overlap → no INI regardless of frequency

            f2  = c2.f[k2];  f2_end = f2 + c2.w[k2] * G_rows(mu2)
            # Violation: frequency ranges are within G rows of each other.
            # Safe iff: f2 >= f1_end + G  (c2 is above c1 with enough gap)
            #        OR f1 >= f2_end + G  (c1 is above c2 with enough gap)
            if f2 < f1_end + G and f1 < f2_end + G:
                return True

    return False


# ============================================================================
# Slot Conflict Check — Asymmetric Guard Band Model
# ============================================================================

def _slot_conflicts(
    tc: int, fc: int, muc: int, wc: int,   # candidate slot
    te: int, fe: int, mue: int, we: int,   # existing slot
    config: SystemConfig,
) -> bool:
    """
    Return True if the candidate slot conflicts with the existing slot.

    Conflict conditions (both must hold — time AND frequency):

    Time:  the two slots overlap in time (their time intervals share ≥1 column).

    Frequency (numerology-aware):
        Same numerology (muc == mue):
            Raw frequency overlap — same subcarrier spacing, no guard needed.
        Different numerology (muc ≠ mue):
            Frequency separation < G_guard rows — INI would occur.
            Safe iff:  fc ≥ fe_end + G  (candidate is above existing with gap)
                    OR fc_end + G ≤ fe  (candidate is below existing with gap)

    This is the DIRECTIONAL guard band model.  The old symmetric ±G model
    treated the guard as a margin owned by each allocation individually,
    which doubled the required separation between cross-numerology BWPs
    (2G rows wasted instead of G).  The correct model treats the guard as a
    property of the BOUNDARY between two allocations — exactly G rows between
    the raw end of one and the raw start of the other.

    Note: this function is used in Phase 3 gap-fill only.  The Phase 1 master
    problem uses a separate guard-band pairwise cut mechanism (_columns_guard_band_conflict)
    which already implements the correct directional check.
    """
    dur_c = E(muc, config.mu_max)
    dur_e = E(mue, config.mu_max)

    # Time overlap check — intervals [tc, tc+dur_c) and [te, te+dur_e)
    if tc + dur_c <= te or te + dur_e <= tc:
        return False   # no time overlap → no conflict regardless of frequency

    fc_end = fc + wc * G_rows(muc)
    fe_end = fe + we * G_rows(mue)

    if muc == mue:
        # Same numerology: only raw frequency overlap matters (no guard needed)
        return not (fc >= fe_end or fe >= fc_end)
    else:
        # Different numerology: need G_guard rows of clear space between them
        G = config.G_guard
        return not (fc >= fe_end + G or fc_end + G <= fe)


# ============================================================================
# Phase 0 Init Worker  (module-level — must be picklable for multiprocessing)
# ============================================================================

def _init_worker(
    args: Tuple,
) -> Tuple[int, Optional["Column"], float, float, float]:
    """
    Standalone Phase 0 initialisation worker for ProcessPoolExecutor.

    Solves the single-UE MIP that seeds the column pool.  The objective is
    to maximise total PRBs (subject to the SLA floor from _build_single_ue_prob).

    Args:
        args : (vue, config, valid_asns)
            vue        — VirtualUE to initialise
            config     — SystemConfig (picklable dataclass)
            valid_asns — pre-computed list of (t,f,mu,w) tuples

    Returns:
        (vue_id, column_or_None, t_build, t_solve, t_total)
        column_or_None : the best Column found, or None if infeasible.
        t_build / t_solve / t_total : timing breakdowns for logging.
    """
    vue, config, valid_asns = args
    t_start = time.perf_counter()

    t_build_start = time.perf_counter()
    prob, A, valid_asns = _build_single_ue_prob(
        vue, config, "init",
        valid_asns_override=valid_asns,
    )
    t_build = time.perf_counter() - t_build_start

    if not A:
        return vue.virtual_id, None, t_build, 0.0, time.perf_counter() - t_start

    prob += pulp.lpSum(
        w * A[(k, t_val, f_val, mu, w)]
        for k in range(config.K)
        for (t_val, f_val, mu, w) in valid_asns
    )

    t_solve_start = time.perf_counter()
    prob.solve(_make_solver(threads=1, time_limit=60))
    t_solve = time.perf_counter() - t_solve_start
    t_total = time.perf_counter() - t_start

    if pulp.value(prob.objective) is None:
        return vue.virtual_id, None, t_build, t_solve, t_total

    return (
        vue.virtual_id,
        _extract_column(vue, config, A, valid_asns),
        t_build, t_solve, t_total,
    )


# ============================================================================
# Parallel Pricing Worker  (module-level — must be picklable for multiprocessing)
# ============================================================================

def _pricing_worker(
    args: Tuple,
) -> Tuple[int, Optional["Column"], float]:
    """
    Standalone pricing subproblem worker for ProcessPoolExecutor.

    Must be defined at module level (not as a class method) so that Python's
    multiprocessing can pickle it when forking worker processes.

    Args:
        args : (vue, config, pi, mu_dual, valid_asns)
            vue        — VirtualUE to price
            config     — SystemConfig (picklable dataclass)
            pi         — {resource_cell: dual_price} from the LP master
            mu_dual    — dual price of this VUE's convexity constraint
            valid_asns — pre-computed list of (t,f,mu,w) tuples for this VUE.
                         Cached at __init__ and passed here to avoid recomputing
                         the triple-nested loop inside the worker.

    NOTE: cells_raw is intentionally NOT passed here. Pickling 13,044 frozensets
    per VUE × 6 VUEs × N iterations costs ~0.54s/iter in IPC overhead, which
    exceeds the time saved by avoiding _cells_of_assignment recomputation (~0.1s
    of total pricing time). The cells cache is only used in Phase 3 (sequential,
    no IPC cost). Workers call _cells_of_assignment directly instead.

    Returns:
        (vue_id, new_Column_or_None, elapsed_seconds)
    """
    vue, config, pi, mu_dual, valid_asns = args
    t_start = time.perf_counter()

    prob, A, valid_asns = _build_single_ue_prob(
        vue, config, "pricing",
        valid_asns_override=valid_asns,
    )
    if not A:
        return vue.virtual_id, None, time.perf_counter() - t_start

    # Shadow cost — call _cells_of_assignment directly (module-level, available
    # in worker after fork). Cheaper than pickling cells_raw across the IPC boundary.
    shadow: Dict[Tuple, float] = {
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
    prob.solve(_make_solver(threads=1, time_limit=30))

    pricing_obj = pulp.value(prob.objective)
    elapsed     = time.perf_counter() - t_start

    if pricing_obj is None or (pricing_obj - mu_dual) <= 1e-6:
        return vue.virtual_id, None, elapsed

    return vue.virtual_id, _extract_column(vue, config, A, valid_asns), elapsed


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
        config:         SystemConfig,
        vues:           List[VirtualUE],
        alpha:          Dict[str, float] = None,
        max_iter:       int = 200,
        n_workers:      int = 0,
        solver_threads: int = 0,
    ):
        """
        Args:
            config         : SystemConfig instance.
            vues           : list of VirtualUE instances.
            alpha          : per-slice priority weights (stored, currently unused — Issue 1).
            max_iter       : maximum Phase 1 CG loop iterations.
            n_workers      : number of parallel worker processes for pricing subproblems.
                             0 (default) → auto: min(len(vues), os.cpu_count()).
                             1           → sequential (no multiprocessing overhead).
                             N > 1       → use N worker processes.
                             Ideal value is len(vues) if you have enough cores, since
                             all pricing subproblems are fully independent per iteration.
            solver_threads : threads given to HiGHS (or CBC) for the LP master and
                             integer master solves.  These two solves run sequentially
                             (one at a time) so all available cores can be handed to
                             the solver.
                             0 (default) → auto: os.cpu_count().
                             1           → single-threaded (same as old CBC behaviour).
        """
        self.config    = config
        self.vues      = vues
        self.alpha     = alpha or {"urllc": 2.0, "embb": 1.0}
        self.max_iter  = max_iter
        self.n_workers = (
            min(len(vues), os.cpu_count() or 1)
            if n_workers == 0
            else n_workers
        )
        self._solver_threads = os.cpu_count() or 1 if solver_threads == 0 else solver_threads

        # MPI VUE assignment — which VUEs rank 0 prices locally.
        # Round-robin: rank r handles vues[r], vues[r+P], vues[r+2P], …
        # Worker ranks compute the same assignment independently from their
        # own _MPI_RANK, so no startup communication is needed.
        if _USE_MPI:
            self._mpi_local_vues: List[VirtualUE] = [
                v for idx, v in enumerate(vues) if idx % _MPI_SIZE == 0
            ]
            logging.info(
                f"[rank 0] MPI enabled — {_MPI_SIZE} ranks total.  "
                f"Rank 0 local VUEs: {[v.virtual_id for v in self._mpi_local_vues]}"
            )

        # column pool: {vue_id → [Column, …]}
        self.columns: Dict[int, List[Column]] = {v.virtual_id: [] for v in vues}

        # ── Incremental master-problem caches ─────────────────────────────────
        #
        # _cell_users[cell] = [(vue_id, col_idx), …]
        #   Maps every raw resource cell to the (VUE, column) pairs that
        #   occupy it.  Updated in O(|col.cells|) by _append_column so that
        #   _build_master_problem no longer needs the O(|cells|×N×C) scan it
        #   previously ran on every LP iteration.
        #
        # _gb_conflicts = {((vi_id, ci), (vj_id, cj)), …}  (vi_id < vj_id)
        #   Pre-computed guard-band-conflict pairs across all columns.
        #   Updated in O(N×C_other) by _append_column; replaces the
        #   O(N²×C²×K²) detection loop that ran once in _solve_integer_master.
        self._cell_users: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self._gb_conflicts: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()

        # ── Pre-computed caches (built once, reused across all phases) ─────────
        #
        # _valid_asns: the set of feasible (t, f, mu, w) positions for each VUE.
        # This is deterministic and depends only on (vue, config).  Previously it
        # was recomputed inside every call to _build_single_ue_prob — once in
        # Phase 0, once per pricing call in Phase 1 (N × n_iters times), and
        # once per VUE per gap-fill iteration in Phase 3.  Caching it here
        # eliminates all of those redundant Python-level triple-nested loops.
        t_cache = time.perf_counter()
        self._valid_asns: Dict[int, List[Tuple]] = {
            v.virtual_id: _get_valid_assignments(v, config) for v in vues
        }

        # _cells_raw / _cells_guarded: the grid-cell footprint of each
        # (t, f, mu, w) assignment, without and with the G_guard buffer.
        # In Phase 3 the ban loop calls _cells_of_assignment for every
        # (k, asn) pair — K × |valid_asns| calls per VUE per iteration
        # (e.g. 3 × 13,044 = 39,132 calls per eMBB VUE).  Replacing each
        # call with a dict lookup reduces this to O(1) per lookup.
        #
        # The cache is keyed by (t, f, mu, w); config parameters (mu_max,
        # n_freq_rows, G_guard) are fixed for the lifetime of the solver.
        all_asns: set = set()
        for asns in self._valid_asns.values():
            all_asns.update(asns)

        self._cells_raw: Dict[Tuple, frozenset] = {
            asn: _cells_of_assignment(*asn, config, with_guard=False)
            for asn in all_asns
        }
        self._cells_guarded: Dict[Tuple, frozenset] = {
            asn: _cells_of_assignment(*asn, config, with_guard=True)
            for asn in all_asns
        }
        t_cache = time.perf_counter() - t_cache
        logging.debug(
            f"Cache built: {len(self._valid_asns)} VUEs  "
            f"{len(all_asns)} unique assignments  "
            f"[{t_cache:.3f}s]"
        )

    # ── Incremental column registration ──────────────────────────────────────

    def _append_column(self, vue_id: int, col: Column) -> None:
        """
        Append col to self.columns[vue_id] and update both master caches.

        _cell_users update  — O(|col.cells|):
            For each raw cell occupied by col, record (vue_id, col_idx) so
            _build_master_problem can build resource constraints in one pass
            over _cell_users instead of the old O(|cells|×N×C) nested scan.

        _gb_conflicts update — O(N × C_other):
            Check col against every existing column of every OTHER VUE and
            record conflicting pairs.  Replaces the O(N²×C²×K²) detection
            loop that ran in _solve_integer_master after the pool was fixed.

        Called at most N times per CG iteration (once per new column found by
        pricing), so the per-iteration overhead is O(N × (|cells| + N×C)).
        """
        c_idx = len(self.columns[vue_id])
        self.columns[vue_id].append(col)

        # Update cell → users index
        for cell in col.cells:
            entry = self._cell_users.get(cell)
            if entry is None:
                self._cell_users[cell] = [(vue_id, c_idx)]
            else:
                entry.append((vue_id, c_idx))

        # Update guard band conflict set
        for other_vue in self.vues:
            ov_id = other_vue.virtual_id
            if ov_id == vue_id:
                continue
            for oc_idx, other_col in enumerate(self.columns[ov_id]):
                if _columns_guard_band_conflict(col, other_col, self.config):
                    # Canonical order: smaller vue_id first to avoid duplicates
                    if vue_id < ov_id:
                        self._gb_conflicts.add(((vue_id, c_idx), (ov_id, oc_idx)))
                    else:
                        self._gb_conflicts.add(((ov_id, oc_idx), (vue_id, c_idx)))

    # ── Phase 0 : initial columns ────────────────────────────────────────────

    def _generate_initial_columns(self):
        """
        Seed each VUE with a null column and (if feasible) an initial column.

        The null column (total_prbs=0, cells=∅) guarantees the master problem
        is always feasible: if every real column for some VUE is blocked by
        resource conflicts, the null column can still be selected.

        Phase 0 MIPs are fully independent (no inter-VUE coupling), so they
        are submitted in parallel using the same ProcessPoolExecutor as Phase 1.
        With n_workers ≥ 3 the three eMBB MIPs (~1.1s each) run concurrently,
        cutting Phase 0 from ~4.3s to ~1.4s.
        """
        t_phase0_start = time.perf_counter()
        logging.info(
            "── Phase 0: Initial columns (single-UE MIPs, no interaction) ───"
        )

        # Build null columns for every VUE up front (trivial, always needed).
        null_cols: Dict[int, Column] = {
            vue.virtual_id: Column(
                vue_id=vue.virtual_id,
                t=[-1]*self.config.K, f=[-1]*self.config.K,
                mu=[None]*self.config.K, w=[0]*self.config.K,
                total_prbs=0, cells=frozenset(),
            )
            for vue in self.vues
        }

        parallel = self.n_workers > 1

        if _USE_MPI:
            # ── MPI path: each rank computes its local VUEs in parallel ──────
            # Workers do the same in _mpi_worker_loop and gather here.
            # Threads are safe: prob.solve() releases the GIL via subprocess.
            n_threads  = min(len(self._mpi_local_vues), os.cpu_count() or 1)
            init_args  = [
                (vue, self.config, self._valid_asns[vue.virtual_id])
                for vue in self._mpi_local_vues
            ]
            local_results: Dict[int, Tuple] = {}

            if n_threads > 1:
                with ThreadPoolExecutor(max_workers=n_threads) as tp:
                    for vid, col, tb, ts, tt in tp.map(_init_worker, init_args):
                        local_results[vid] = (col, tb, ts, tt)
            else:
                for args in init_args:
                    vid, col, tb, ts, tt = _init_worker(args)
                    local_results[vid] = (col, tb, ts, tt)

            # Gather from all ranks (workers already called _COMM.gather).
            all_rank_results = _COMM.gather(local_results, root=0)
            merged: Dict[int, Tuple] = {}
            for rank_dict in all_rank_results:
                merged.update(rank_dict)

            for vue in self.vues:
                i = vue.virtual_id
                col, t_build, t_solve, t_total = merged[i]
                if col is None:
                    logging.warning(
                        f"  VUE {i:3d} ({vue.group_id} {vue.sla_slice_id:5s}): "
                        f"no feasible init MIP — null column only.  "
                        f"[build={t_build:.3f}s  solve={t_solve:.3f}s  total={t_total:.3f}s]"
                    )
                    self._append_column(i, null_cols[i])
                else:
                    logging.info(
                        f"  VUE {i:3d} ({vue.group_id} {vue.sla_slice_id:5s}):  "
                        f"{col.total_prbs:3d} PRBs  {len(col.cells):3d} cells  "
                        f"|asns|={len(self._valid_asns[i])}  "
                        f"[build={t_build:.3f}s  solve={t_solve:.3f}s  total={t_total:.3f}s]"
                    )
                    self._append_column(i, null_cols[i])
                    self._append_column(i, col)

        elif parallel:
            work = [
                (vue, self.config, self._valid_asns[vue.virtual_id])
                for vue in self.vues
            ]
            futures = {}
            with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                for w in work:
                    futures[pool.submit(_init_worker, w)] = w[0]
                results: Dict[int, Tuple] = {}
                for fut in as_completed(futures):
                    vid, col, t_build, t_solve, t_total = fut.result()
                    results[vid] = (col, t_build, t_solve, t_total)

            # Log and assign in VUE order (deterministic output)
            for vue in self.vues:
                i = vue.virtual_id
                col, t_build, t_solve, t_total = results[i]
                if col is None:
                    logging.warning(
                        f"  VUE {i:3d} ({vue.group_id} {vue.sla_slice_id:5s}): "
                        f"no feasible init MIP — null column only.  "
                        f"[build={t_build:.3f}s  solve={t_solve:.3f}s  total={t_total:.3f}s]"
                    )
                    self._append_column(i, null_cols[i])
                else:
                    logging.info(
                        f"  VUE {i:3d} ({vue.group_id} {vue.sla_slice_id:5s}):  "
                        f"{col.total_prbs:3d} PRBs  {len(col.cells):3d} cells  "
                        f"|asns|={len(self._valid_asns[i])}  "
                        f"[build={t_build:.3f}s  solve={t_solve:.3f}s  total={t_total:.3f}s]"
                    )
                    self._append_column(i, null_cols[i])
                    self._append_column(i, col)

        else:
            # Sequential path — identical to original, preserved for debugging.
            for vue in self.vues:
                i = vue.virtual_id
                t_vue_start = time.perf_counter()

                t_build_start = time.perf_counter()
                prob, A, valid_asns = _build_single_ue_prob(
                    vue, self.config, "init",
                    valid_asns_override=self._valid_asns[i],
                )
                t_build = time.perf_counter() - t_build_start

                if not A:
                    logging.warning(
                        f"  VUE {i:3d} ({vue.group_id} {vue.sla_slice_id:5s}): "
                        f"no valid assignments — null column only."
                    )
                    self._append_column(i, null_cols[i])
                    continue

                prob += pulp.lpSum(
                    w * A[(k, t_val, f_val, mu, w)]
                    for k in range(self.config.K)
                    for (t_val, f_val, mu, w) in valid_asns
                )

                t_solve_start = time.perf_counter()
                prob.solve(_make_solver(threads=1, time_limit=60))
                t_solve = time.perf_counter() - t_solve_start
                t_vue   = time.perf_counter() - t_vue_start

                if pulp.value(prob.objective) is None:
                    logging.warning(
                        f"  VUE {i:3d} ({vue.group_id} {vue.sla_slice_id:5s}): "
                        f"infeasible single-UE MIP — null column only.  "
                        f"[build={t_build:.3f}s  solve={t_solve:.3f}s  total={t_vue:.3f}s]"
                    )
                    self._append_column(i, null_cols[i])
                else:
                    col = _extract_column(vue, self.config, A, valid_asns)
                    logging.info(
                        f"  VUE {i:3d} ({vue.group_id} {vue.sla_slice_id:5s}):  "
                        f"{col.total_prbs:3d} PRBs  {len(col.cells):3d} cells  "
                        f"|asns|={len(valid_asns)}  "
                        f"[build={t_build:.3f}s  solve={t_solve:.3f}s  total={t_vue:.3f}s]"
                    )
                    self._append_column(i, null_cols[i])
                    self._append_column(i, col)

        t_phase0 = time.perf_counter() - t_phase0_start
        logging.info(f"  Phase 0 total: {t_phase0:.3f}s")

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
            max  Σ_{i,c}  total_prbs(c) · λ_{i,c}
            s.t. Σ_c λ_{i,c} = 1                   ∀i  (convexity)
                 Σ_{i,c: r ∈ c.cells} λ_{i,c} ≤ 1  ∀r  (resource non-conflict)
                 λ ∈ [0,1]  (LP)  or  {0,1}  (IP)

        The objective is uncapped total PRBs: convexity constraints already force
        exactly one column per VUE, and resource non-conflict prevents any two
        selected columns from sharing cells, so there is no monopoly risk.
        The old cap min(total_prbs, n_prb_i) was removed because it gave zero
        gradient above the SLA floor, which caused the pricing subproblem to never
        generate larger columns and left unused spectrum as gaps in the grid.

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

        # Objective: capped per-VUE PRB contribution.
        # The cap min(total_prbs, n_prb_i) is essential for fairness during CG:
        # without it a single VUE grabbing all rows scores higher than the fair
        # distribution, so the LP picks the monopoly and the dual prices never
        # incentivise the pricing subproblem to generate diverse columns.
        # Gap-filling beyond the SLA floor is handled by _gap_fill() (Phase 3),
        # which runs after the integer master with fixed inter-VUE occupancy.
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

        # Resource non-conflict: use the pre-built incremental index.
        # _cell_users[cell] = [(vue_id, col_idx), …] for every (VUE, column)
        # pair that occupies that raw cell.  Maintained in O(|col.cells|) by
        # _append_column; replaces the old O(|cells|×N×C) nested scan.
        res_cname: Dict[Tuple[int, int], str] = {}
        for cell, users in self._cell_users.items():
            if len(users) <= 1:
                continue
            cname = f"res_{cell[0]:04d}_{cell[1]:04d}"
            prob += (
                pulp.lpSum(lam[(i, c_idx)] for (i, c_idx) in users) <= 1,
                cname,
            )
            res_cname[cell] = cname

        # Guard band pairwise cuts — INTEGER master only.
        # The raw-cell resource constraints above cannot catch guard band violations:
        # two columns from different VUEs may share no raw cells yet still violate
        # the G_guard row separation required between time-overlapping cross-numerology
        # slots.  We detect these pairs and add λ[i,c1] + λ[j,c2] ≤ 1 cuts.
        #
        # Only added for the integer master (not the LP) because:
        #   (a) The LP uses fractional λ; guard band enforcement via shadow costs
        #       is sufficient there to steer column generation.
        #   (b) The integer master has a small, fixed column pool (≤ ~20 cols/VUE)
        #       so the number of pairs is bounded (~6000 for 6 VUEs × 20 cols).
        if integer:
            # Guard band pairwise cuts from the pre-built incremental set.
            # _gb_conflicts was populated by _append_column in O(N×C_other) per
            # new column; replaces the old O(N²×C²×K²) detection loop here.
            t_gb_start = time.perf_counter()
            gb_cuts = 0
            for (vi_id, ci), (vj_id, cj) in self._gb_conflicts:
                cname = f"gb_{vi_id}_{ci}_{vj_id}_{cj}"
                prob += (
                    lam[(vi_id, ci)] + lam[(vj_id, cj)] <= 1,
                    cname,
                )
                gb_cuts += 1
            t_gb = time.perf_counter() - t_gb_start
            logging.info(
                f"  GB cuts: {gb_cuts} pairwise cuts added  [detection={t_gb:.3f}s]"
            )

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
        prob.solve(_make_solver(threads=self._solver_threads, time_limit=120))

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
        prob, A, valid_asns = _build_single_ue_prob(
            vue, self.config, "pricing",
            valid_asns_override=self._valid_asns[vue.virtual_id],
        )
        if not A:
            return None

        # Shadow cost per assignment — use cached raw cell footprints instead
        # of recomputing _cells_of_assignment on every call.
        shadow: Dict[Tuple, float] = {
            asn: sum(pi.get(cell, 0.0) for cell in self._cells_raw[asn])
            for asn in valid_asns
        }

        prob += pulp.lpSum(
            (w - shadow[(t_val, f_val, mu, w)]) * A[(k, t_val, f_val, mu, w)]
            for k in range(self.config.K)
            for (t_val, f_val, mu, w) in valid_asns
        )
        prob.solve(_make_solver(threads=1, time_limit=30))

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

        Pricing execution modes (chosen in priority order):

        1. MPI inter-node (_USE_MPI=True, _MPI_SIZE > 1):
               Rank 0 broadcasts (pi, mu_dual) to all worker ranks each
               iteration.  Each rank solves its assigned VUEs sequentially,
               then gathers results back to rank 0.  After the loop, rank 0
               broadcasts None to terminate the worker loops.

               No ProcessPoolExecutor is used inside MPI workers (fork+MPI is
               unsafe on most HPC clusters).

        2. Intra-node parallel (_USE_MPI=False, n_workers > 1):
               ProcessPoolExecutor with n_workers processes, one per VUE.

        3. Sequential (_USE_MPI=False, n_workers == 1):
               Retained for single-node debugging.
        """
        t_phase1_start = time.perf_counter()
        parallel = self.n_workers > 1 and not _USE_MPI
        mode_str = (
            f"MPI ({_MPI_SIZE} ranks)" if _USE_MPI
            else (f"intra-node parallel ({self.n_workers} workers)" if parallel
                  else "sequential")
        )
        logging.info(
            f"── Phase 1: Column Generation loop (max {self.max_iter} iters)  "
            f"[{mode_str}] ──"
        )

        # Pool is created once and reused across all iterations.
        # Fork on Linux is fast; the pool is idle between iterations.
        pool_ctx = (
            ProcessPoolExecutor(max_workers=self.n_workers)
            if parallel
            else None
        )

        try:
            for iteration in range(self.max_iter):
                t_iter_start = time.perf_counter()

                t_lp_start = time.perf_counter()
                lp_obj, pi, mu_dual = self._solve_lp_master()
                t_lp = time.perf_counter() - t_lp_start

                t_pricing_start = time.perf_counter()
                new_cols_found = 0
                pricing_times: Dict[int, float] = {}  # vue_id → elapsed (local only in MPI mode)

                if _USE_MPI:
                    # ── MPI inter-node pricing ────────────────────────────────
                    # Broadcast dual prices to all worker ranks.
                    # pi is a Dict[Tuple[int,int], float] — at most
                    # n_time_cols × n_freq_rows entries (~648 for 5 MHz / 3 ms).
                    # Pickled broadcast size is O(10 KB) — negligible latency.
                    _COMM.bcast({"pi": pi, "mu_dual": mu_dual}, root=0)

                    # Solve pricing for rank 0's local VUEs (thread pool —
                    # GIL released during CBC/HiGHS subprocess).
                    local_results: Dict[int, Optional[Column]] = {}
                    _n_local = len(self._mpi_local_vues)
                    _n_th    = min(_n_local, os.cpu_count() or 1)
                    _p_args  = [
                        (vue, self.config, pi,
                         mu_dual.get(vue.virtual_id, 0.0),
                         self._valid_asns[vue.virtual_id])
                        for vue in self._mpi_local_vues
                    ]
                    if _n_th > 1:
                        with ThreadPoolExecutor(max_workers=_n_th) as _tp:
                            for vid, new_col, elapsed in _tp.map(_pricing_worker, _p_args):
                                pricing_times[vid]  = elapsed
                                local_results[vid]  = new_col
                    else:
                        for vue, args in zip(self._mpi_local_vues, _p_args):
                            t_p = time.perf_counter()
                            vid, new_col, _ = _pricing_worker(args)
                            pricing_times[vid] = time.perf_counter() - t_p
                            local_results[vid] = new_col

                    # Gather results from all ranks.
                    # all_results[r] is the dict sent by rank r.
                    all_results = _COMM.gather(local_results, root=0)

                    # Register new columns in VUE order (deterministic).
                    merged: Dict[int, Optional[Column]] = {}
                    for rank_dict in all_results:
                        merged.update(rank_dict)
                    for vue in self.vues:
                        new_col = merged.get(vue.virtual_id)
                        if new_col is not None:
                            self._append_column(vue.virtual_id, new_col)
                            new_cols_found += 1

                elif parallel:
                    # ── Intra-node parallel pricing ───────────────────────────
                    # Pass valid_asns per VUE (cheap — list of tuples).
                    # cells_raw is NOT passed: pickling 13,044 frozensets per VUE
                    # costs ~0.54s/iter in IPC overhead, more than the time saved.
                    # Workers call _cells_of_assignment directly instead.
                    work = [
                        (
                            vue,
                            self.config,
                            pi,
                            mu_dual.get(vue.virtual_id, 0.0),
                            self._valid_asns[vue.virtual_id],
                        )
                        for vue in self.vues
                    ]
                    futures = {
                        pool_ctx.submit(_pricing_worker, w): w[0].virtual_id
                        for w in work
                    }
                    results: Dict[int, Optional[Column]] = {}
                    for fut in as_completed(futures):
                        vid, new_col, elapsed = fut.result()
                        results[vid]       = new_col
                        pricing_times[vid] = elapsed

                    # Add columns in VUE order (deterministic).
                    for vue in self.vues:
                        new_col = results.get(vue.virtual_id)
                        if new_col is not None:
                            self._append_column(vue.virtual_id, new_col)
                            new_cols_found += 1

                else:
                    # ── Sequential pricing (debug / single-node) ──────────────
                    for vue in self.vues:
                        t_p = time.perf_counter()
                        new_col = self._solve_pricing(
                            vue, pi, mu_dual.get(vue.virtual_id, 0.0)
                        )
                        pricing_times[vue.virtual_id] = time.perf_counter() - t_p
                        if new_col is not None:
                            self._append_column(vue.virtual_id, new_col)
                            new_cols_found += 1

                t_pricing = time.perf_counter() - t_pricing_start
                total_cols = sum(len(v) for v in self.columns.values())
                t_iter     = time.perf_counter() - t_iter_start

                if pricing_times:
                    times = list(pricing_times.values())
                    timing_detail = (
                        f"(local min={min(times):.2f}s "
                        f"avg={sum(times)/len(times):.2f}s "
                        f"max={max(times):.2f}s)"
                    )
                else:
                    timing_detail = "(no local VUEs)"
                logging.info(
                    f"  Iter {iteration:3d}: LP obj = {lp_obj:8.3f}  "
                    f"new cols = {new_cols_found:3d}  total cols = {total_cols}  "
                    f"[lp={t_lp:.2f}s  pricing={t_pricing:.2f}s {timing_detail}  "
                    f"iter={t_iter:.2f}s]"
                )

                if new_cols_found == 0:
                    logging.info(f"  LP optimal reached at iteration {iteration}.")
                    break

        finally:
            # Ensure the ProcessPoolExecutor is always shut down cleanly.
            if pool_ctx is not None:
                pool_ctx.shutdown(wait=True)
            # Signal MPI workers to exit their pricing loop.
            # bcast(None) is the termination sentinel; workers break on it.
            # This runs whether the loop converged normally or raised.
            if _USE_MPI:
                _COMM.bcast(None, root=0)

        t_phase1 = time.perf_counter() - t_phase1_start
        logging.info(f"  Phase 1 total: {t_phase1:.3f}s")

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
        t_phase2_start = time.perf_counter()
        total_cols = sum(len(v) for v in self.columns.values())
        logging.info(
            f"── Phase 2: Integer master (Set Partitioning)  "
            f"—  {total_cols} columns ──"
        )

        t_build_start = time.perf_counter()
        prob, lam, conv_cname, res_cname = self._build_master_problem(integer=True)
        t_build = time.perf_counter() - t_build_start

        t_solve_start = time.perf_counter()
        prob.solve(_make_solver(threads=self._solver_threads, time_limit=300))
        t_solve = time.perf_counter() - t_solve_start

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

        t_phase2 = time.perf_counter() - t_phase2_start
        logging.info(
            f"  Phase 2 total: {t_phase2:.3f}s  "
            f"[master_build={t_build:.3f}s  master_solve={t_solve:.3f}s]"
        )

        return int_obj, selected

    # ── Phase 3 : gap-fill expansion ─────────────────────────────────────────

    def _gap_fill(
        self,
        selected: Dict[int, Optional[Column]],
        max_iters: int = 20,
    ) -> Dict[int, Optional[Column]]:
        """
        Phase 3: Coordinate-search expansion that fills unused spectrum.

        CG phases 0-2 find a feasible allocation where every VUE exactly meets
        its SLA floor (n_prb PRBs) — the objective's per-VUE cap means there is
        no gradient to go higher.  This phase relaxes that cap and iteratively
        re-solves each VUE's single-UE subproblem with the free spectrum visible,
        keeping improvements until convergence.

        Algorithm (mirrors coord_search.py Phase 2):
          For each iteration:
            For each VUE i (in order):
              1. Compute occupied cells = cells used by all OTHER VUEs' current columns.
              2. Solve single-UE MIP (pin_prbs=False) maximising total PRBs subject
                 to (a) ≥ n_prb floor, (b) no overlap with occupied cells.
              3. If new allocation is better, update selected[i].
          Stop when no VUE improved in a full pass.

        Bug that was here previously: a "rebuild" pass called _build_single_ue_prob
        a second time and then only excluded conflicting assignments from the
        *objective*, leaving them as free binary variables with zero coefficient.
        The MIP solver could still set them to 1 (e.g. to satisfy c17_min_prb or
        as part of cascade/ordering constraints) causing real cell overlaps.

        Fix: build once, then ADD explicit A[...] == 0 constraints for every
        assignment whose raw cells intersect the occupied set.  This hard-bans
        conflicting positions from the feasible region entirely.

        Returns the updated selected dict.
        """
        t_phase3_start = time.perf_counter()
        logging.info(
            f"── Phase 3: Gap-fill expansion (max {max_iters} iters) ──────────"
        )

        def _active_slots_of_others(exclude_id: int) -> List[Tuple[int,int,int,int]]:
            """
            Return list of (t,f,mu,w) active slots from all VUEs except exclude_id.

            Returning slots (not a cell set) allows the ban loop to use the
            directional _slot_conflicts check, which applies the guard band only
            between different-numerology, time-overlapping pairs.  The old
            frozenset-union approach added ±G symmetrically, wasting 1–2 extra
            rows per cross-numerology boundary.
            """
            slots: List[Tuple[int,int,int,int]] = []
            for vid, col in selected.items():
                if vid == exclude_id or col is None or col.total_prbs == 0:
                    continue
                for k in range(self.config.K):
                    if col.w[k] > 0 and col.t[k] >= 0:
                        slots.append((col.t[k], col.f[k], col.mu[k], col.w[k]))
            return slots

        # Parallel Jacobi gap-fill.  At the start of each iteration we take
        # a snapshot of `selected`.  All VUEs are solved concurrently against
        # this snapshot (no sequential dependency).  Improvements are then
        # committed greedily in VUE order with a cell-conflict check.
        #
        # Why threads are safe here:
        #   • prob.solve() calls CBC/HiGHS as an external subprocess, releasing
        #     the GIL while the subprocess runs.  True parallelism despite Python.
        #   • Each problem is named "gapfill_vue{id}", so temp files are unique.
        #   • `snap` and `_valid_asns` are read-only inside the worker closure.
        #
        # Quality vs sequential (Gauss-Seidel):
        #   The sequential version lets VUE i+1 see VUE i's expanded cells.
        #   Jacobi uses a fixed snapshot, so two VUEs may propose using the
        #   same free cells.  The greedy commit (VUE order) resolves conflicts —
        #   the later VUE's improvement is dropped if it overlaps with an earlier
        #   committed improvement.  In practice this causes at most one extra
        #   iteration and yields identical or near-identical final PRB counts.
        n_phase3_workers = min(len(self.vues), os.cpu_count() or 1)

        vues_to_run = set(v.virtual_id for v in self.vues)

        for iteration in range(max_iters):
            t_iter_start = time.perf_counter()

            # ── Snapshot occupancy (shared read-only by all threads) ──────────
            snap: Dict[int, Optional["Column"]] = dict(selected)

            def _active_slots_snap(exclude_id: int) -> List[Tuple]:
                slots: List[Tuple] = []
                for vid, col in snap.items():
                    if vid == exclude_id or col is None or col.total_prbs == 0:
                        continue
                    for k in range(self.config.K):
                        if col.w[k] > 0 and col.t[k] >= 0:
                            slots.append((col.t[k], col.f[k], col.mu[k], col.w[k]))
                return slots

            # ── Per-VUE gap-fill worker (runs in thread pool) ─────────────────
            def _solve_one(vue: "VirtualUE"):
                i           = vue.virtual_id
                old_prbs    = snap[i].total_prbs if snap[i] else 0
                valid_asns  = self._valid_asns[i]

                t_occ_start  = time.perf_counter()
                other_slots  = _active_slots_snap(i)
                t_occ        = time.perf_counter() - t_occ_start

                t_build_start = time.perf_counter()
                prob, A, _    = _build_single_ue_prob(
                    vue, self.config, "gapfill", pin_prbs=False,
                    valid_asns_override=valid_asns,
                )
                if not A:
                    return (i, None, old_prbs, 0, 0.0, 0.0, t_occ, 0)

                n_banned = 0
                for k in range(self.config.K):
                    for (tc, fc, muc, wc) in valid_asns:
                        if any(
                            _slot_conflicts(tc, fc, muc, wc, te, fe, mue, we, self.config)
                            for (te, fe, mue, we) in other_slots
                        ):
                            key = (k, tc, fc, muc, wc)
                            if key in A:
                                prob += (A[key] == 0, f"ban_k{k}_t{tc}_f{fc}_m{muc}_w{wc}")
                                n_banned += 1

                free_asns = [
                    (tc, fc, muc, wc) for (tc, fc, muc, wc) in valid_asns
                    if not any(
                        _slot_conflicts(tc, fc, muc, wc, te, fe, mue, we, self.config)
                        for (te, fe, mue, we) in other_slots
                    )
                ]
                t_build = time.perf_counter() - t_build_start

                if not free_asns:
                    return (i, None, old_prbs, 0, t_build, 0.0, t_occ, n_banned)

                prob += pulp.lpSum(
                    w * A[(k, t_val, f_val, mu, w)]
                    for k in range(self.config.K)
                    for (t_val, f_val, mu, w) in free_asns
                    if (k, t_val, f_val, mu, w) in A
                )

                t_solve_start = time.perf_counter()
                prob.solve(_make_solver(threads=1, time_limit=30))
                t_solve = time.perf_counter() - t_solve_start

                if pulp.value(prob.objective) is None:
                    return (i, None, old_prbs, 0, t_build, t_solve, t_occ, n_banned)

                new_col  = _extract_column(vue, self.config, A, valid_asns)
                new_prbs = new_col.total_prbs
                if new_prbs > old_prbs:
                    return (i, new_col, old_prbs, new_prbs, t_build, t_solve, t_occ, n_banned)
                return (i, None, old_prbs, 0, t_build, t_solve, t_occ, n_banned)

            # ── Solve all VUEs in parallel ────────────────────────────────────
            vues_list = [v for v in self.vues if v.virtual_id in vues_to_run]
            proposals: Dict[int, Tuple] = {}

            if n_phase3_workers > 1:
                with ThreadPoolExecutor(max_workers=n_phase3_workers) as tp:
                    for result in tp.map(_solve_one, vues_list):
                        proposals[result[0]] = result
            else:
                for vue in vues_list:
                    result = _solve_one(vue)
                    proposals[result[0]] = result

            # ── Commit improvements greedily in VUE order ─────────────────────
            # VUEs not in proposals keep their current assignment.
            # If two VUEs proposed overlapping cells, the first one (lower VUE
            # id) wins; later conflicting proposals are skipped this iteration.
            committed_new: Set[Tuple[int, int]] = set()  # cells from NEW improvements
            improved_this_iter: set = set()

            for vue in self.vues:
                i = vue.virtual_id
                if i not in proposals:
                    continue
                i2, new_col, old_prbs, new_prbs, t_build, t_solve, t_occ, n_banned = proposals[i]
                if new_col is None or new_prbs <= old_prbs:
                    logging.debug(
                        f"  Iter {iteration:2d}  VUE {i:3d} "
                        f"({vue.group_id} {vue.sla_slice_id:5s}): no gain  "
                        f"[occ={t_occ:.3f}s  build+ban={t_build:.3f}s  "
                        f"solve={t_solve:.3f}s]"
                    )
                    continue
                # Reject if new cells conflict with an earlier committed improvement.
                new_cells = new_col.cells - (snap[i].cells if snap[i] else frozenset())
                if new_cells & committed_new:
                    logging.debug(
                        f"  Iter {iteration:2d}  VUE {i:3d}: cell conflict with "
                        f"earlier improvement — skipped (will retry next iter)"
                    )
                    continue
                selected[i] = new_col
                committed_new |= new_cells
                improved_this_iter.add(i)
                logging.info(
                    f"  Iter {iteration:2d}  VUE {i:3d} "
                    f"({vue.group_id} {vue.sla_slice_id:5s}): "
                    f"{old_prbs} → {new_prbs} PRBs  "
                    f"banned={n_banned}  "
                    f"[occ={t_occ:.3f}s  build+ban={t_build:.3f}s  "
                    f"solve={t_solve:.3f}s]"
                )

            total_prbs = sum(col.total_prbs for col in selected.values() if col)
            t_iter = time.perf_counter() - t_iter_start

            if not improved_this_iter:
                logging.info(
                    f"  Iter {iteration:2d}: no improvement — converged.  "
                    f"Total PRBs = {total_prbs}  [iter={t_iter:.3f}s]"
                )
                break

            logging.info(
                f"  Iter {iteration:2d}: total PRBs = {total_prbs}  "
                f"[iter={t_iter:.3f}s  improved={sorted(improved_this_iter)}]"
            )

            # Re-run all VUEs next iteration (any VUE could benefit if a
            # skipped-due-to-conflict VUE retries with updated occupancy).
            vues_to_run = set(v.virtual_id for v in self.vues)

        t_phase3 = time.perf_counter() - t_phase3_start
        logging.info(f"  Phase 3 total: {t_phase3:.3f}s")

        return selected

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
        Run all phases and write the solution to output_path.

        Phase 0  _generate_initial_columns() — single-UE MIPs, no interaction
        Phase 1  _run_cg_loop()              — LP master → dual prices → pricing
        Phase 2  _solve_integer_master()     — Set Partitioning MIP (SLA-exact)
        Phase 3  _gap_fill()                 — coordinate-search expansion into
                                               free spectrum beyond SLA floors

        Args:
            output_path : full path to the .txt output file.
        """
        solver_name = "HiGHS" if _USE_HIGHS else "CBC"
        logging.info(
            f"Starting ColumnGeneration solver.  "
            f"Grid: {self.config.n_time_cols}t × {self.config.n_freq_rows}f  |  "
            f"VUEs: {len(self.vues)}  |  K={self.config.K}  |  "
            f"max_iter={self.max_iter}  |  workers={self.n_workers}  |  "
            f"solver={solver_name}  solver_threads={self._solver_threads}  |  "
            f"cached_asns={sum(len(v) for v in self._valid_asns.values())}  "
            f"cached_cells={len(self._cells_raw)}"
        )

        self._generate_initial_columns()
        self._run_cg_loop()
        _, selected = self._solve_integer_master()

        # Phase 3: fill the gaps left by the SLA-exact integer solution
        selected = self._gap_fill(selected)

        final_obj = sum(col.total_prbs for col in selected.values() if col)
        logging.info(f"Final objective (after gap-fill): {final_obj} PRBs")
        self._write_output(output_path, selected, float(final_obj))


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
        "--n-workers", type=int, default=0,
        help="Parallel worker processes for pricing subproblems. "
             "0 = auto (min(n_vues, cpu_count)). 1 = sequential.",
    )
    parser.add_argument(
        "--solver-threads", type=int, default=0,
        help="Threads given to HiGHS (or CBC) for the LP master and integer master. "
             "0 = auto (os.cpu_count()). 1 = single-threaded (old CBC behaviour).",
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
    parser.add_argument(
        "--job-id", default="",
        help="SLURM job ID appended to output filenames (e.g. '321'). "
             "Empty string = no suffix.",
    )
    args = parser.parse_args()

    # Prefix log lines with MPI rank so interleaved mpirun output is readable.
    log_fmt = (
        f"%(asctime)s [rank {_MPI_RANK}/{_MPI_SIZE}] %(message)s"
        if _USE_MPI else "%(asctime)s %(message)s"
    )
    logging.basicConfig(
        level  = getattr(logging, args.log_level),
        format = log_fmt,
    )

    # ── MPI worker path ───────────────────────────────────────────────────────
    # All ranks read the same config file (shared filesystem).
    # VUE assignment is the same deterministic formula on every rank, so no
    # startup communication is needed to distribute the VUE list.
    if _MPI_RANK > 0:
        if _USE_MPI:
            cfg, vues = load_and_build(args.config)
            my_vues = [v for idx, v in enumerate(vues) if idx % _MPI_SIZE == _MPI_RANK]
            _mpi_worker_loop(cfg, my_vues)
        else:
            # mpi4py not installed but srun launched multiple tasks — worker
            # ranks have nothing to do without MPI communication.
            logging.error(
                f"SLURM rank {_MPI_RANK}: mpi4py is not installed. "
                "Worker ranks cannot participate without MPI. "
                "Fix: pip install mpi4py  in your venv."
            )
        sys.exit(0)

    # ── Master / single-node path ─────────────────────────────────────────────
    start_time = time.time()

    cfg, vues = load_and_build(args.config)
    alpha     = {"urllc": args.alpha_urllc, "embb": args.alpha_embb}

    solver = ColumnGenerationSolver(
        config         = cfg,
        vues           = vues,
        alpha          = alpha,
        max_iter       = args.max_iterations,
        n_workers      = args.n_workers,
        solver_threads = args.solver_threads,
    )

    os.makedirs(args.output, exist_ok=True)
    filename    = os.path.splitext(os.path.basename(args.config))[0]
    job_suffix  = f"_job{args.job_id}" if args.job_id else ""
    output_file = os.path.join(args.output, f"{filename}{job_suffix}.txt")

    solver.solve(output_file)

    print(f"Solution written to: {output_file}")
    print(f"Total solve time: {time.time() - start_time:.2f} seconds")