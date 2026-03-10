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
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

try:
    import pulp
except ImportError:
    print("ERROR: PuLP not installed.  Run: pip install pulp --break-system-packages")
    sys.exit(1)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SliceSLA:
    """
    Service Level Agreement for a single network slice.

    Attributes:
        slice_id        : unique label, e.g. "embb" | "urllc" | "mmtc"
        throughput_mbps : minimum throughput required over Delta_T (Mbps)
        latency_ms      : maximum latency deadline (ms)
                          every PRB for this slice must be scheduled
                          before this time within the Delta_T window
    """
    slice_id:        str
    throughput_mbps: float
    latency_ms:      float


@dataclass
class PhysicalUE:
    """
    A physical user equipment with one or more slice attachments.

    Attributes:
        ue_id  : unique string identifier (e.g. "UE1")
        mcs    : Modulation and Coding Scheme index 0-28
                 (3GPP TS 38.214 Table 5.1.3.1)
                 determines bits per resource element → drives N_PRB_i
        slices : list of SliceSLA — one per attached network slice
                 a physical UE with S slices generates S virtual UEs
    """
    ue_id:  str
    mcs:    int
    slices: List[SliceSLA] = field(default_factory=list)


@dataclass
class VirtualUE:
    """
    A virtual UE — logical representation of one slice on one physical UE.

    One PhysicalUE with S slices expands into S VirtualUEs.
    VirtualUEs sharing the same physical UE form a group; constraints
    (18) and (19) enforce that group members cannot use different
    numerologies when their time slots overlap (one radio frontend per
    physical device — cannot run two FFT sizes simultaneously).

    Attributes:
        virtual_id    : unique 0-based integer index; used in variable names
        physical_ue   : parent PhysicalUE reference
        sla           : the SliceSLA this virtual UE represents
        group_id      : equals physical_ue.ue_id — group membership key
        n_prb         : min PRBs to satisfy throughput SLA
                        computed by compute_n_prb()
        latency_slots : latency deadline in integer grid column units
                        = round(sla.latency_ms / delta_T)
                        constraint (16): every slot must end before this
    """
    virtual_id:    int
    physical_ue:   PhysicalUE
    sla:           SliceSLA
    group_id:      str
    n_prb:         int = 0
    latency_slots: int = 0


@dataclass
class SystemConfig:
    """
    System-level parameters defining the resource grid.

    Input fields:
        bandwidth_hz  : carrier bandwidth in Hz (e.g. 5_000_000 for 5 MHz)
        delta_t_ms    : scheduling window length in ms (e.g. 3.0)
        mu_max        : maximum numerology in the system, 0-4
        K             : max time-slot assignments allowed per virtual UE
        numerologies  : list of numerologies M ⊆ {0,1,2,3,4} to consider
        G_guard       : guard band width in grid rows between BWPs of different
                        numerology (INI avoidance). Default = 1 row = δF = 180 kHz.

    Derived fields (computed in __post_init__):
        delta_T       : minimum time unit = 2^(-mu_max) ms  (1 grid column)
        delta_F       : minimum freq  unit = 180 kHz        (1 grid row)
                        = G(mu=0) = 12 * 15 kHz
                        NOTE: the paper writes δF = 12·15·2^µmax Hz which
                        gives non-integer G(µ)/δF for µ < µmax.  The
                        correct atomic unit is G(µ=0) so that G(µ)/δF = 2^µ
                        is always an integer.  This also matches Fig. 5's
                        y-axis label "unit is related to numerology 0".
        n_time_cols   : grid columns = delta_t_ms / delta_T  (integer)
        n_freq_rows   : grid rows    = bandwidth_hz / delta_F (integer)
        Omega         : global upper bound on ω (= n_freq_rows)
    """
    bandwidth_hz: float
    delta_t_ms:   float
    mu_max:       int
    K:            int
    numerologies: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    G_guard:      int   = 1       # guard band rows (default 1 row = 180 kHz)

    # Derived — populated by __post_init__
    delta_T:     float = field(init=False)
    delta_F:     float = field(init=False)
    n_time_cols: int   = field(init=False)
    n_freq_rows: int   = field(init=False)
    Omega:       int   = field(init=False)

    def __post_init__(self):
        self.delta_T     = 2 ** (-self.mu_max)          # ms
        self.delta_F     = 12 * 15 * 1000               # Hz  (G at µ=0 = 180 kHz)
        self.n_time_cols = round(self.delta_t_ms / self.delta_T)
        self.n_freq_rows = int(self.bandwidth_hz / self.delta_F)
        self.Omega       = self.n_freq_rows


# ============================================================================
# N_PRB_i Computation
# ============================================================================

# 3GPP TS 38.214 Table 5.1.3.1 — MCS index → (modulation order Q, code rate r)
# Q = bits/symbol: QPSK=2, 16QAM=4, 64QAM=6, 256QAM=8
# r = code rate (standard stores r*1024; here stored as float)
_MCS_TABLE: Dict[int, Tuple[int, float]] = {
    0:  (2, 0.1172),
    1:  (2, 0.1533),
    2:  (2, 0.1885),
    3:  (2, 0.2451),
    4:  (2, 0.3008),
    5:  (2, 0.3770),
    6:  (2, 0.4492),
    7:  (2, 0.5547),
    8:  (2, 0.6016),
    9:  (2, 0.6504),
    10: (4, 0.3672),   # 16-QAM
    11: (4, 0.4238),
    12: (4, 0.4785),
    13: (4, 0.5352),
    14: (4, 0.6016),
    15: (4, 0.6426),
    16: (4, 0.6953),
    17: (6, 0.4385),   # 64-QAM
    18: (6, 0.4951),
    19: (6, 0.5537),
    20: (6, 0.6016),
    21: (6, 0.6504),
    22: (6, 0.7021),
    23: (6, 0.7539),
    24: (8, 0.5527),   # 256-QAM
    25: (8, 0.6016),
    26: (8, 0.6504),
    27: (8, 0.7021),
    28: (8, 0.7539),
}

# Resource elements per PRB per slot = 12 subcarriers × 14 OFDM symbols
_RE_PER_PRB: int = 168


def compute_n_prb(throughput_mbps: float, delta_t_ms: float, mcs: int) -> int:
    """
    Compute the minimum number of PRBs required to satisfy a throughput SLA.

    Derivation:
        TBS_per_PRB  = RE_per_PRB * Q * r
                     = 168 * modulation_order * code_rate     bits/PRB

        total_bits   = throughput_mbps * delta_t_ms * 1000
                       (1 Mbps * 1 ms = 1000 bits)

        N_PRB_i      = ⌈ total_bits / TBS_per_PRB ⌉

    Args:
        throughput_mbps : required throughput in Mbps
        delta_t_ms      : scheduling window length in ms
        mcs             : MCS index 0-28

    Returns:
        Minimum integer PRB count (at least 1).

    Raises:
        ValueError if mcs not in 0-28.
    """
    if mcs not in _MCS_TABLE:
        raise ValueError(
            f"MCS {mcs} is not supported (valid range 0-28; "
            f"MCS 29-31 are reserved per 3GPP TS 38.214)."
        )
    Q, r         = _MCS_TABLE[mcs]
    tbs_per_prb  = _RE_PER_PRB * Q * r
    total_bits   = throughput_mbps * delta_t_ms * 1000
    return max(1, math.ceil(total_bits / tbs_per_prb))


# ============================================================================
# Grid Utility Functions
# ============================================================================

def E(mu: int, mu_max: int) -> int:
    """
    Slot duration of numerology µ in integer grid column units.

    E(µ) = 2^(µmax - µ)

    With µmax = 3:
        µ=0 → 8 cols   (1 ms    slot, 8 * 0.125 ms cols)
        µ=1 → 4 cols   (0.5 ms  slot)
        µ=2 → 2 cols   (0.25 ms slot)
        µ=3 → 1 col    (0.125 ms slot)
    """
    return 2 ** (mu_max - mu)


def G_rows(mu: int) -> int:
    """
    Frequency width of ONE PRB at numerology µ in integer grid row units.

    G_rows(µ) = G_hz(µ) / delta_F = 2^µ

    With delta_F = 180 kHz (= G at µ=0):
        µ=0 → 1 row    (180 kHz  PRB / 180 kHz row)
        µ=1 → 2 rows   (360 kHz  PRB)
        µ=2 → 4 rows   (720 kHz  PRB)
        µ=3 → 8 rows   (1440 kHz PRB)

    Higher numerology → wider subcarrier spacing → wider PRB → more rows.
    ω contiguous PRBs at µ occupy ω * 2^µ grid rows total.
    """
    return 2 ** mu


def max_omega(mu: int, n_freq_rows: int) -> int:
    """
    Upper bound on ω (contiguous PRBs) at numerology µ.

    From constraint (7): F^k_i + ω * G_rows(µ) ≤ n_freq_rows
    With F^k_i ≥ 0:   ω ≤ n_freq_rows // G_rows(µ)

    Lower µ → narrower PRBs → more PRBs fit per row → larger max_omega.
    """
    return n_freq_rows // G_rows(mu)


# ============================================================================
# Config Loader
# ============================================================================

def load_config(path: str) -> Tuple[SystemConfig, List[PhysicalUE]]:
    """
    Parse a JSON config file into SystemConfig + list of PhysicalUE.

    Expected JSON structure:
    {
        "system": {
            "bandwidth_hz":  5000000,
            "delta_t_ms":    3.0,
            "mu_max":        3,
            "K":             10,
            "numerologies":  [0, 1, 2, 3]
        },
        "ues": [
            {
                "ue_id":  "UE1",
                "mcs":    26,
                "slices": [
                    { "slice_id": "embb",  "throughput_mbps": 4.0, "latency_ms": 3.0 },
                    { "slice_id": "urllc", "throughput_mbps": 1.0, "latency_ms": 0.5 }
                ]
            }
        ]
    }

    Returns:
        (SystemConfig, list[PhysicalUE])

    Raises:
        FileNotFoundError : config file missing
        KeyError          : required field absent
        ValueError        : invalid parameter value
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")

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

    if config.mu_max not in range(5):
        raise ValueError(f"mu_max must be in {{0..4}}, got {config.mu_max}")
    if config.n_time_cols <= 0:
        raise ValueError("Grid has 0 time columns — check delta_t_ms and mu_max.")
    if config.n_freq_rows <= 0:
        raise ValueError("Grid has 0 frequency rows — check bandwidth_hz.")

    ues: List[PhysicalUE] = []
    for u in raw["ues"]:
        slices = [
            SliceSLA(
                slice_id        = str(sl["slice_id"]),
                throughput_mbps = float(sl["throughput_mbps"]),
                latency_ms      = float(sl["latency_ms"]),
            )
            for sl in u["slices"]
        ]
        ues.append(PhysicalUE(ue_id=str(u["ue_id"]), mcs=int(u["mcs"]), slices=slices))

    return config, ues


# ============================================================================
# Virtual UE Builder
# ============================================================================

def build_virtual_ues(
    physical_ues: List[PhysicalUE],
    config: SystemConfig,
) -> Tuple[List[VirtualUE], Dict[str, List[int]]]:
    """
    Expand each physical UE into one VirtualUE per attached slice.

    For each VirtualUE:
        n_prb         ← compute_n_prb(throughput_mbps, delta_t_ms, mcs)
        latency_slots ← round(latency_ms / delta_T)

    Groups are built for physical UEs with ≥ 2 slices.
    Single-slice UEs have no intra-group constraints.

    Returns:
        (list[VirtualUE], groups)
        groups : { physical_ue_id : [virtual_id, ...] }
    """
    vues:   List[VirtualUE]         = []
    groups: Dict[str, List[int]]    = {}
    vid = 0

    for pue in physical_ues:
        members: List[int] = []
        for sla in pue.slices:
            n_prb = compute_n_prb(sla.throughput_mbps, config.delta_t_ms, pue.mcs)
            latency_slots = round(sla.latency_ms / config.delta_T)
            vues.append(VirtualUE(
                virtual_id    = vid,
                physical_ue   = pue,
                sla           = sla,
                group_id      = pue.ue_id,
                n_prb         = n_prb,
                latency_slots = latency_slots,
            ))
            members.append(vid)
            vid += 1

        if len(members) > 1:
            groups[pue.ue_id] = members

    return vues, groups


# ============================================================================
# MILP Class
# ============================================================================

class RadioResourceMILP:
    """
    MILP formulation for 5G NR radio resource management with network slicing.

    Maximises total PRBs allocated (objective 20) subject to constraints
    (3)---(19) from Boutiba et al. 2022.

    Typical usage::

        config, ues = load_config("config.json")
        milp = RadioResourceMILP(config, ues)
        milp.build()
        milp.write()          # writes .lp / .mps beside config
        milp.solve()          # optional

    Decision variables (paper notation → internal dict key):
        T^k_i               → self.T[i][k]             integer
        F^k_i               → self.F[i][k]             integer
        X^{k,µ}_{i,ω}       → self.X[i][k][mu][w]     binary decision variable for slot assignment for UE i at slot k 
                                                      with numerology µ and ω contiguous PRBs
        Y^{k,l}_{i,j}       → self.Y[(i,j)][(k,l)]    binary time overlap between slot k of UE i and slot l of UE j
        Z^{k,l}_{i,j}       → self.Z[(i,j)][(k,l)]    binary time direction selector for overlapping slots of UE i and UE j,
                                                        Z^{k,l}_{i,j} = 1 implies l-th slot of UE j finishes before k-th slot of UE i
                                                        Z^{k,l}_{i,j} = 0 implies k-th slot of UE i finishes before l-th slot of UE j
        W^{k,l}_{i,j}       → self.W[(i,j)][(k,l)]    binary frequency direction selector for overlapping slots of UE i and UE j
                                                        W^{k,l}_{i,j} = 1 implies slot k of UE i is at lower frequency than slot l of UE j
                                                        W^{k,l}_{i,j} = 0 implies slot l of UE j is at lower frequency than slot k of UE i
        I^{k,l,µ}_{i,j}     → self.I_gb[(i,j)][(k,l)][mu]  binary  guard band
    """

    def __init__(self, config: SystemConfig, physical_ues: List[PhysicalUE]):
        """
        Initialise the MILP instance.

        Builds virtual UEs and groups; derives big-M constants.
        Does NOT build the model — call build() for that.

        Big-M constants:
            PHI_TIME = n_time_cols   upper bound on any T^k_i
            PHI_FREQ = n_freq_rows   upper bound on any F^k_i
        """
        self.config       = config
        self.physical_ues = physical_ues

        self.virtual_ues, self.groups = build_virtual_ues(physical_ues, config)

        self.PHI_TIME: int = config.n_time_cols + 2
        self.PHI_FREQ: int = config.n_freq_rows + 2 * config.G_guard + 2

        # PuLP problem object — set by _create_problem()
        self.problem: Optional[pulp.LpProblem] = None

        # Decision variable dicts — populated by _create_variables()
        self.T:    Dict = {}   # T[i][k]
        self.F:    Dict = {}   # F[i][k]
        self.X:    Dict = {}   # X[i][k][mu][w]
        self.Y:    Dict = {}   # Y[(i,j)][(k,l)]
        self.Z:    Dict = {}   # Z[(i,j)][(k,l)]
        self.W:    Dict = {}   # W[(i,j)][(k,l)]
        self.I_gb: Dict = {}   # I_gb[(i,j)][(k,l)][mu]

    # ----------------------------------------------------------------
    # Public Interface
    # ----------------------------------------------------------------

    def build(self) -> "RadioResourceMILP":
        """
        Build the complete MILP model.

        Steps:
            1. _create_problem()    — instantiate PuLP LpProblem
            2. _create_variables()  — declare all decision variables
            3. _add_constraints()   — add constraints (3)–(19)   [STUB]
            4. _set_objective()     — set objective (20)

        Returns self for method chaining.
        """
        self._create_problem()
        self._create_variables()
        self._add_constraints()
        self._set_objective()
        return self

    def solve(
        self,
        solver: Optional[pulp.LpSolver] = None,
        time_limit_s: int = 50_000,
    ) -> int:
        """
        Solve the built MILP.

        Args:
            solver       : PuLP solver instance.
                           Defaults to CBC (bundled with PuLP).
                           To use Gurobi pass: pulp.GUROBI(timeLimit=time_limit_s)
            time_limit_s : wall-clock time limit in seconds (default 50 000)

        Returns:
            PuLP integer status (e.g. pulp.constants.LpStatusOptimal == 1)

        Raises:
            RuntimeError if build() has not been called.
        """
        if self.problem is None:
            raise RuntimeError("Call build() before solve().")

        if solver is None:
            solver = pulp.PULP_CBC_CMD(timeLimit=time_limit_s, gapRel=1e-8, msg=1)

        return self.problem.solve(solver)

    def write(
        self,
        output_dir: Optional[str] = None,
        base_name: str|None = None,
    ) -> Dict[str, str]:
        """
        Write the model to .lp and .mps files.

        Args:
            output_dir : destination directory.
                         Defaults to current working directory.
            base_name  : filename stem (no extension)

        Returns:
            {"lp": absolute_path, "mps": absolute_path}

        Raises:
            RuntimeError if build() has not been called.
        """
        if self.problem is None:
            raise RuntimeError("Call build() before write().")

        # include model parameters in base_name for easier identification of multiple instances
        if not base_name:
            c = self.config
            base_name = (
            f"milp_bw_{int(c.bandwidth_hz/1e6)}_MHz_T_{c.delta_t_ms}_ms_mu_max_{c.mu_max}_K_{c.K}_"
            f"ues_{len(self.physical_ues)}_vues_{len(self.virtual_ues)}"
            )
        out = output_dir or os.getcwd()
        os.makedirs(out, exist_ok=True)

        lp_path  = os.path.join(out, f"{base_name}.lp")
        mps_path = os.path.join(out, f"{base_name}.mps")

        self.problem.writeLP(lp_path)
        self.problem.writeMPS(mps_path)

        return {"lp": os.path.abspath(lp_path), "mps": os.path.abspath(mps_path)}

    def summary(self) -> str:
        """Return a human-readable summary of the problem instance."""
        c = self.config
        lines = [
            "=" * 66,
            "5G NR Radio Resource MILP — Problem Summary",
            "=" * 66,
            f"  Bandwidth        : {c.bandwidth_hz/1e6:.2f} MHz",
            f"  Sched. window    : {c.delta_t_ms} ms",
            f"  µmax             : {c.mu_max}",
            f"  δT               : {c.delta_T} ms  (1 grid column)",
            f"  δF               : {c.delta_F/1e3:.0f} kHz (1 grid row = G at µ=0)",
            f"  Grid             : {c.n_time_cols} cols × {c.n_freq_rows} rows",
            f"  K (max slots)    : {c.K}",
            f"  Numerologies     : {c.numerologies}",
            f"  Guard band (G)   : {c.G_guard} row(s) = {c.G_guard * c.delta_F/1e3:.0f} kHz",
            f"  PHI_TIME (big-M) : {self.PHI_TIME}",
            f"  PHI_FREQ (big-M) : {self.PHI_FREQ}",
            "",
            f"  Physical UEs     : {len(self.physical_ues)}",
            f"  Virtual UEs      : {len(self.virtual_ues)}",
            f"  Groups           : {len(self.groups)}",
            "",
            f"  {'VID':>3}  {'PUE':>5}  {'Slice':>6}  {'MCS':>3}  "
            f"{'R(Mbps)':>7}  {'D(ms)':>6}  {'D(slots)':>8}  {'N_PRB':>5}",
            "  " + "─" * 58,
        ]
        for v in self.virtual_ues:
            lines.append(
                f"  {v.virtual_id:>3}  {v.physical_ue.ue_id:>5}  "
                f"{v.sla.slice_id:>6}  {v.physical_ue.mcs:>3}  "
                f"{v.sla.throughput_mbps:>7.2f}  {v.sla.latency_ms:>6.3f}  "
                f"{v.latency_slots:>8}  {v.n_prb:>5}"
            )
        if self.groups:
            lines += ["", "  Groups (constraints 18 & 19 apply within each):"]
            for gid, members in self.groups.items():
                lines.append(f"    '{gid}'  →  virtual UEs {members}")
        lines.append("=" * 66)
        return "\n".join(lines)

    # ----------------------------------------------------------------
    # Private — Problem Creation
    # ----------------------------------------------------------------

    def _create_problem(self):
        """Instantiate a PuLP maximisation problem (objective 20)."""
        self.problem = pulp.LpProblem(
            name  = "5GNR_RadioResourceMILP",
            sense = pulp.LpMaximize,
        )

    # ----------------------------------------------------------------
    # Private — Variable Creation
    # ----------------------------------------------------------------

    def _create_variables(self):
        """
        Declare all MILP decision variables with correct domains and bounds.

        T[i][k]          : integer, in [-1, n_time_cols - 1]
        F[i][k]          : integer, in [-1, n_freq_rows  - 1]
        X[i][k][mu][w]   : binary,  w ∈ {1 … max_omega(µ)}
        Y/Z/W[(i,j)][(k,l)]        : binary, for all ordered pairs i≠j
        I_gb[(i,j)][(k,l)][mu]     : binary, for all ordered pairs * numerology

        Variable count grows as O(N²·K²·|M|) for Y/Z/W/I — this is
        the source of the exponential solve time noted in the paper.
        """
        cfg  = self.config
        K    = cfg.K
        vids = [v.virtual_id for v in self.virtual_ues]

        # T^k_i and F^k_i
        # use tqdm progress bar for variable creation since this can be a bottleneck for large instances
        for vue in tqdm(self.virtual_ues, desc="[1/3] Creating T and F variables"):
            i = vue.virtual_id
            self.T[i] = {}
            self.F[i] = {}
            for k in range(K):
                self.T[i][k] = pulp.LpVariable(
                    f"T_{i}_{k}",
                    lowBound=-1, upBound=cfg.n_time_cols - 1,
                    cat=pulp.constants.LpInteger,
                )
                self.F[i][k] = pulp.LpVariable(
                    f"F_{i}_{k}",
                    lowBound=-1, upBound=cfg.n_freq_rows - 1,
                    cat=pulp.constants.LpInteger,
                )

        # X^{k,µ}_{i,ω}
        for vue in tqdm(self.virtual_ues, desc="[2/3] Creating X variables"):
            i = vue.virtual_id
            self.X[i] = {}
            for k in range(K):
                self.X[i][k] = {}
                for mu in cfg.numerologies:
                    self.X[i][k][mu] = {}
                    for w in range(1, max_omega(mu, cfg.n_freq_rows) + 1):
                        self.X[i][k][mu][w] = pulp.LpVariable(
                            f"X_{i}_{k}_{mu}_{w}", cat=pulp.constants.LpBinary
                        )

        # Y, Z, W, I_gb — for every ordered pair of distinct virtual UEs
        for i in tqdm(vids, desc="[3/3] Creating Y, Z, W, I_gb variables"):
            for j in vids:
                if i == j:
                    continue
                self.Y[(i, j)]    = {}
                self.Z[(i, j)]    = {}
                self.W[(i, j)]    = {}
                self.I_gb[(i, j)] = {}
                for k in range(K):
                    for l in range(K):
                        self.Y[(i, j)][(k, l)] = pulp.LpVariable(
                            f"Y_{i}_{j}_{k}_{l}", cat=pulp.constants.LpBinary
                        )
                        self.Z[(i, j)][(k, l)] = pulp.LpVariable(
                            f"Z_{i}_{j}_{k}_{l}", cat=pulp.constants.LpBinary
                        )
                        self.W[(i, j)][(k, l)] = pulp.LpVariable(
                            f"W_{i}_{j}_{k}_{l}", cat=pulp.constants.LpBinary
                        )
                        self.I_gb[(i, j)][(k, l)] = {}
                        for mu in cfg.numerologies:
                            self.I_gb[(i, j)][(k, l)][mu] = pulp.LpVariable(
                                f"I_{i}_{j}_{k}_{l}_{mu}", cat=pulp.constants.LpBinary
                            )

        # Print total variable count for debugging / insight into model size
        def _count_lp_vars(d) -> int:
            if isinstance(d, pulp.LpVariable):
                return 1
            return sum(_count_lp_vars(v) for v in d.values())

        total_vars = sum(_count_lp_vars(d) for d in [self.T, self.F, self.X, self.Y, self.Z, self.W, self.I_gb])
        print(f"Total decision variables created: {total_vars}")

    # ----------------------------------------------------------------
    # Private — Constraints  *** STUB ***
    # ----------------------------------------------------------------

    def _add_constraints(self):
        """
        Add constraints (3)–(19) to self.problem.

        *** STUB — constraints will be wired in a later session ***

        Quick-reference for the implementation session:
        ─────────────────────────────────────────────────────────────
        Let i index virtual UEs, k index their slot assignments (0..K-1),
        all time/freq values in integer grid units.

        Helper symbols used below:
            sum_X(i,k) = Σ_{µ,ω} X[i][k][µ][ω]   (active indicator for slot k)
            dur(i,k)   = Σ_{µ,ω} E(µ,µmax) · X[i][k][µ][ω]   (slot duration cols)
            width(i,k) = Σ_{µ,ω} ω · G_rows(µ) · X[i][k][µ][ω] (slot freq rows)

        ─────────────────────────────────────────────────────────────
        (3)  T[i][k] >= -1                            [lower bound — enforced by var decl]
        (4)  F[i][k] >= -1                            [lower bound — enforced by var decl]

        (5)  sum_X(i,k) <= 1
             One (µ,ω) pair per UE per slot.
        
        (6)  T[i][k] + dur(i,k) <= n_time_cols
             Slot rectangle fits within time window.

        (7)  F[i][k] + width(i,k) <= n_freq_rows
             Slot rectangle fits within bandwidth.

        (8)  T[i][k] <= -1 + PHI_TIME · sum_X(i,k)
             Sentinel: T = -1 when slot unassigned.
             With (3): T[i][k] = -1 iff sum_X = 0.

        (9)  F[i][k] <= -1 + PHI_FREQ · sum_X(i,k)
             Sentinel: F = -1 when slot unassigned.

        (10) T[i][k] + dur(i,k) <= T[i][k+1] + PHI_TIME·(1 - sum_X(i,k+1))
             UE's own slots are non-overlapping and time-ordered.

        (11) sum_X(i,k+1) <= sum_X(i,k)
             Slots filled from k=0 upward; once unassigned, stays unassigned.

        (13) Four big-M inequalities defining Y[(i,j)][(k,l)]:
             Y = 1  iff  slot k of UE i overlaps slot l of UE j in time.
             Uses auxiliary Z[(i,j)][(k,l)] as direction selector.
             See paper eq. (13) for exact form.

        (14) Two big-M inequalities enforcing frequency separation when Y=1:
             Uses W[(i,j)][(k,l)] (direction) and
             I_gb[(i,j)][(k,l)][µ] (guard band indicator from (15)).
             See paper eq. (14).

        (15) I_gb[(i,j)][(k,l)][µ] >= Σ_ω X[j][l][µ][ω] - Σ_ω X[i][k][µ][ω]
             I_gb[(i,j)][(k,l)][µ] >= Σ_ω X[i][k][µ][ω] - Σ_ω X[j][l][µ][ω]
             I_gb = 1 iff exactly one of {i,j} uses numerology µ
             → numerologies differ → INI → guard band G required.

        (16) T[i][k] + dur(i,k) < vue.latency_slots
             (implemented as: T[i][k] + dur(i,k) <= latency_slots - 1)
             Every slot for UE i completes before its latency deadline.

        (17) Σ_{k,µ,ω} ω · X[i][k][µ][ω] >= vue.n_prb
             Cumulative PRBs across all slots satisfies throughput SLA.

        (18) For each group g, each pair (i,j) in g, each (k,l), each µ:
             Σ_ω X[j][l][µ][ω] - Σ_ω X[i][k][µ][ω] <= 1 - Y[(i,j)][(k,l)]
             Y[(i,j)][(k,l)] - 1 <= Σ_ω X[j][l][µ][ω] - Σ_ω X[i][k][µ][ω]
             Same-group VUEs use same numerology when time slots overlap.

        (19) For each group g, each pair (i,j) in g, each (k,l):
             T[i][k] - T[j][l] <= PHI_TIME · (1 - Y[(i,j)][(k,l)])
             PHI_TIME · (Y[(i,j)][(k,l)] - 1) <= T[i][k] - T[j][l]
             Same-group VUEs start at the same time when slots overlap.
        ─────────────────────────────────────────────────────────────
        """

        cfg    = self.config
        K      = cfg.K

        # Convenience: sum of all X[i][k][mu][w] for a given (i, k)
        # Reused by constraints (5), (6), (7), (8), (9), (10), (11)
        def sum_X(i, k):
            """Σ_{µ,ω} X[i][k][µ][ω]  — 1 if slot k of UE i is assigned, 0 otherwise."""
            return pulp.lpSum(
                self.X[i][k][mu][w]
                for mu in cfg.numerologies
                for w  in range(1, max_omega(mu, cfg.n_freq_rows) + 1)
            )

        def dur(i, k):
            """Σ_{µ,ω} E(µ,µmax)·X[i][k][µ][ω]  — slot duration in grid columns."""
            return pulp.lpSum(
                E(mu, cfg.mu_max) * self.X[i][k][mu][w]
                for mu in cfg.numerologies
                for w  in range(1, max_omega(mu, cfg.n_freq_rows) + 1)
            )

        def width(i, k):
            """Σ_{µ,ω} ω·G_rows(µ)·X[i][k][µ][ω]  — slot width in grid rows."""
            return pulp.lpSum(
                w * G_rows(mu) * self.X[i][k][mu][w]
                for mu in cfg.numerologies
                for w  in range(1, max_omega(mu, cfg.n_freq_rows) + 1)
            )

        # ───────────────────────── Constraint (5) ─────────────────────────────
        # ∀i ∈ N, ∀k ∈ {0..K-1}:  Σ_{µ,ω} X[i][k][µ][ω] ≤ 1
        #
        # At most one (µ, ω) pair may be active per virtual UE per slot.
        # sum_X = 0 → slot k is unassigned (T and F will be forced to -1 by (8),(9))
        # sum_X = 1 → slot k uses exactly one numerology with exactly one PRB width
        #
        # This is a set-packing constraint over the binary X variables.
        # It naturally prevents a UE from using two different numerologies
        # simultaneously (which is physically impossible — one radio frontend,
        # one FFT size per time slot).
        c5_count = 0
        for vue in self.virtual_ues:
            i = vue.virtual_id
            for k in range(K):
                self.problem += (
                    sum_X(i, k) <= 1,
                    f"c5_one_numerology_per_slot_vue{i}_slot{k}"
                )
                c5_count += 1
        # print(f"Total constraints added for (5): sum_X(i,k) <= 1 :  {c5_count}")

        # Constraint (6): slot must not exceed the right edge of the time grid
        # T[i][k] + dur(i,k) <= n_time_cols
        # Unassigned: T=-1, dur=0 -> LHS=-1, trivially OK
        # Assigned:   enforces slot fits within the Delta_T scheduling window
        # All terms in integer grid column units (multiples of delta_T ms)
        c6_count = 0
        for vue in self.virtual_ues:
            i = vue.virtual_id
            for k in range(K):
                self.problem += (
                    self.T[i][k] + dur(i, k) <= cfg.n_time_cols,
                    f"c6_time_boundary_vue{i}_slot{k}"
                )
                c6_count += 1
        # print(f"Total constraints added for (6): T[i][k] + dur(i,k) <= n_time_cols :  {c6_count}")

        # Constraint (7): slot must not exceed the top edge of the frequency grid
        # F[i][k] + width(i,k) <= n_freq_rows
        # Unassigned: F=-1, width=0 -> LHS=-1, trivially OK
        # Assigned:   enforces slot fits within the bandwidth of the gNB
        # All terms in integer grid row units (multiples of delta_F kHz)
        c7_count = 0
        for vue in self.virtual_ues:
            i = vue.virtual_id
            for k in range(K):
                self.problem += (
                    self.F[i][k] + width(i, k) <= cfg.n_freq_rows,
                    f"c7_freq_boundary_vue{i}_slot{k}"
                )
                c7_count += 1
        # print(f"Total constraints added for (7): F[i][k] + width(i,k) <= n_freq_rows :  {c7_count}")

        # ───────────────────────── Constraint (8) ─────────────────────────────
        # ∀i ∈ N, ∀k ∈ {0..K-1}:  T[i][k] ≤ -1 + Φ · Σ_{µ,ω} X[i][k][µ][ω]
        #
        # Big-M sentinel: forces T[i][k] = -1 when slot k is unassigned.
        #
        # When sum_X = 0 (unassigned): RHS = -1 + Φ·0 = -1
        #   → T[i][k] ≤ -1, and T[i][k] ≥ -1 from (3) → T[i][k] = -1  ✓
        #
        # When sum_X = 1 (assigned):   RHS = -1 + Φ·1 = Φ-1 = n_time_cols-1
        #   → T[i][k] ≤ n_time_cols-1, which is non-binding since T is already
        #     bounded above by upBound=n_time_cols-1 from variable declaration.
        #     The solver is free to choose any valid T[i][k] ≥ 0.
        #
        # PHI_TIME = n_time_cols (set in __init__) is the tightest valid big-M:
        # large enough to never bind when assigned, small enough to avoid
        # numerical issues in the LP relaxation.
        c8_count = 0
        for vue in self.virtual_ues:
            i = vue.virtual_id
            for k in range(K):
                self.problem += (
                    self.T[i][k] <= -1 + self.PHI_TIME * sum_X(i, k),
                    f"c8_sentinel_T_vue{i}_slot{k}"
                )
                c8_count += 1
        # print(f"Total constraints added for (8): T[i][k] <= -1 + PHI_TIME * sum_X(i,k) :  {c8_count}")

        #  ─────── Constraint (9) is analogous to (8) for F[i][k]:─────────────────────────────
        # F[i][k] ≤ -1 + PHI_FREQ · sum_X(i, k)
        c9_count = 0
        for vue in self.virtual_ues:
            i = vue.virtual_id
            for k in range(K):
                self.problem += (
                    self.F[i][k] <= -1 + self.PHI_FREQ * sum_X(i, k),
                    f"c9_sentinel_F_vue{i}_slot{k}"
                )
                c9_count += 1
        # print(f"Total constraints added for (9): F[i][k] <= -1 + PHI_FREQ * sum_X(i,k) :  {c9_count}")

        # ── Constraint (10) ───────────────────────────────────────────────────
        # ∀i ∈ N, ∀k ∈ {0..K-2}:
        #   T[i][k] + dur(i,k) ≤ T[i][k+1] + Φ·(1 − sum_X(i,k+1))
        #
        # Two purposes in one constraint:
        #
        # (A) Non-overlap of a UE's own slots:
        #     When slot k+1 is assigned (sum_X(i,k+1)=1):  RHS = T[i][k+1]
        #       → T[i][k] + dur(i,k) ≤ T[i][k+1]
        #       → slot k must finish before slot k+1 starts  ✓
        #
        # (B) Relaxation when slot k+1 is unassigned (sum_X(i,k+1)=0):
        #     RHS = T[i][k+1] + Φ = -1 + Φ = n_time_cols - 1
        #       → T[i][k] + dur(i,k) ≤ n_time_cols - 1, which is always
        #         satisfied given constraint (6)  ✓
        #
        # Together with (11), this enforces that assigned slots are packed
        # contiguously from k=0 upward with strictly increasing start times.
        c10_count = 0
        for vue in self.virtual_ues:
            i = vue.virtual_id
            for k in range(K - 1):
                self.problem += (
                    self.T[i][k] + dur(i, k) <= self.T[i][k+1] + self.PHI_TIME * (1 - sum_X(i, k+1)),
                    f"c10_slot_ordering_vue{i}_slot{k}"
                )
                c10_count += 1
        # print(f"Total constraints added for (10): T[i][k] + dur(i,k) <= T[i][k+1] + PHI_TIME * (1 - sum_X(i,k+1)) :  {c10_count}")

        # ── Constraint (11) ───────────────────────────────────────────────────
        # ∀i ∈ N, ∀k ∈ {0..K-2}:  sum_X(i,k+1) ≤ sum_X(i,k)
        #
        # Slots are filled left-to-right: slot k+1 can only be assigned if
        # slot k is also assigned. Equivalently, once a slot is unassigned
        # all subsequent slots must also be unassigned.
        #
        # This "cascade" ordering is what makes the sentinel logic in (8)/(9)
        # well-defined and prevents the solver from leaving gaps in the slot
        # sequence, which would make the non-overlap ordering in (10) ambiguous.
        #
        # Note: sum_X(i,k) ∈ {0,1} from constraint (5), so this is simply
        # a binary dominance constraint.
        c11_count = 0
        for vue in self.virtual_ues:
            i = vue.virtual_id
            for k in range(K - 1):
                self.problem += (
                    sum_X(i, k+1) <= sum_X(i, k),
                    f"c11_slot_cascade_vue{i}_slot{k}"
                )
                c11_count += 1
        # print(f"Total constraints added for (11): sum_X(i,k+1) <= sum_X(i,k) :  {c11_count}")

        # ── Constraint (13) ───────────────────────────────────────────────────
        # ∀i,j ∈ N, i≠j, ∀k,l ∈ {0..K-1}:
        # Y[(i,j)][(k,l)] = 1  iff  slot l of UE j overlaps slot k of UE i in time
        #
        # Four big-M inequalities using auxiliary binary Z[(i,j)][(k,l)]:
        #
        # (13a)  T_j^l ≤ T_i^k + Φ·(1 − Y)
        #   When Y=1: T_j^l ≤ T_i^k  →  j's slot starts no later than i's start
        #   When Y=0: trivially relaxed by Φ
        #
        # (13b)  T_i^k ≤ T_j^l + dur(j,l) + Φ·(1 − Y)
        #   When Y=1: T_i^k ≤ T_j^l + dur(j,l)  →  i's start is before j's slot ends
        #   Together with (13a): T_j^l ≤ T_i^k ≤ T_j^l+dur(j,l)  → overlap confirmed ✓
        #
        # (13c)  T_i^k + dur(i,k) ≤ T_j^l + Φ·(Y + Z) − 1
        #   When Y=1, Z=1: T_i^k + dur(i,k) ≤ T_j^l + 2Φ − 1  (relaxed)
        #   When Y=1, Z=0: T_i^k + dur(i,k) ≤ T_j^l + Φ − 1   (relaxed)
        #   When Y=0, Z=1: T_i^k + dur(i,k) ≤ T_j^l + Φ − 1   (relaxed)
        #   When Y=0, Z=0: T_i^k + dur(i,k) ≤ T_j^l − 1  →  i finishes before j starts
        #
        # (13d)  T_j^l + dur(j,l) ≤ T_i^k + Φ·(Y + 1 − Z) − 1
        #   When Y=0, Z=1: T_j^l + dur(j,l) ≤ T_i^k − 1  →  j finishes before i starts
        #   Exactly one of (13c)/(13d) is active when Y=0, selected by Z:
        #     Z=0 → (13c) active: i ends before j starts
        #     Z=1 → (13d) active: j ends before i starts
        #
        # Strict < is implemented as ≤ − 1 since all T values are integers.
        # PHI = PHI_TIME throughout; a single big-M suffices for both dims.
        vids = [v.virtual_id for v in self.virtual_ues]
        PHI  = self.PHI_TIME

        c13_count = 0
        for i in vids:
            for j in vids:
                if i == j:
                    continue
                for k in range(K):
                    for l in range(K):
                        Y = self.Y[(i, j)][(k, l)]
                        Z = self.Z[(i, j)][(k, l)]

                        # (13a): T_j^l ≤ T_i^k + Φ·(1 − Y)
                        self.problem += (
                            self.T[j][l] <= self.T[i][k] + PHI * (1 - Y),
                            f"c13a_overlap_lb_vue{i}_{j}_slot{k}_{l}"
                        )
                        
                        # (13b): T_i^k ≤ T_j^l + dur(j,l) + Φ·(1 − Y)
                        self.problem += (
                            self.T[i][k] <= self.T[j][l] + dur(j, l) + PHI * (1 - Y),
                            f"c13b_overlap_ub_vue{i}_{j}_slot{k}_{l}"
                        )
                        # (13c): T_i^k + dur(i,k) < T_j^l + Φ·(Y + Z)
                        #        → ≤ T_j^l + Φ·(Y + Z) − 1  (integer strict →  ≤ − 1)
                        # self.problem += (
                        #     self.T[i][k] + dur(i, k) <= self.T[j][l] + PHI * (Y + Z) - 1,
                        #     f"c13c_nonoverlap_ij_vue{i}_{j}_slot{k}_{l}"
                        # )
                        # use non strict
                        self.problem += (
                            self.T[i][k] + dur(i, k) <= self.T[j][l] + PHI * (Y + Z) ,
                            f"c13c_nonoverlap_ij_vue{i}_{j}_slot{k}_{l}"
                        )
                        # (13d): T_j^l + dur(j,l) < T_i^k + Φ·(Y + 1 − Z)
                        #        → ≤ T_i^k + Φ·(Y + 1 − Z) − 1
                        self.problem += (
                            self.T[j][l] + dur(j, l) <= self.T[i][k] + PHI * (Y + 1 - Z) - 1,
                            f"c13d_nonoverlap_ji_vue{i}_{j}_slot{k}_{l}"
                        )
                        c13_count += 4
        # print(f"Total constraints added for (13): big-M constraints defining Y[(i,j)][(k,l)] = 1  iff  slot l of UE j overlaps slot k of UE i in time :  {c13_count}")

        # ── Constraint (14) ───────────────────────────────────────────────────
        # ∀i,j ∈ N, i≠j, ∀k,l ∈ {0..K-1}:
        # When Y[(i,j)][(k,l)]=1 (slots overlap in time), the two BWPs must
        # not share the same frequency rows.  Guard band G is added between
        # BWPs whose numerologies differ (INI avoidance, from constraint 15).
        #
        # Σ_µ I_{i,j}^{k,l,µ} × G  is the total guard band required:
        # I_gb = 1 for the numerology used by exactly one of {i,j} → mixed
        # numerology → INI → guard band needed. At most one µ is active per
        # slot (constraint 5), so this sum is at most G rows.
        #
        # Two inequalities, direction selected by W[(i,j)][(k,l)]:
        #
        # (14a)  F_j^l + guard(i,j,k,l) + width(j,l) < F_i^k + Φ·(1 − Y + W)
        #        → ≤ F_i^k + Φ·(1 − Y + W) − 1
        #   W=1: relaxed (Φ term dominates)
        #   W=0, Y=1: F_j^l + guard + width(j,l) ≤ F_i^k − 1
        #             → j's block (+ guard) is strictly below i's block  ✓
        #
        # (14b)  F_i^k + guard(i,j,k,l) + width(i,k) < F_j^l + Φ·(2 − Y − W)
        #        → ≤ F_j^l + Φ·(2 − Y − W) − 1
        #   W=0: relaxed
        #   W=1, Y=1: F_i^k + guard + width(i,k) ≤ F_j^l − 1
        #             → i's block (+ guard) is strictly below j's block  ✓
        #
        # Exactly one of (14a)/(14b) is binding when Y=1, chosen by W:
        #   W=0 → (14a) active: j is below i in frequency
        #   W=1 → (14b) active: i is below j in frequency
        #
        # Strict < → ≤ − 1 (integer frequency rows).
        G = cfg.G_guard

        c14_count = 0
        for i in vids:
            for j in vids:
                if i == j:
                    continue
                for k in range(K):
                    for l in range(K):
                        Y = self.Y[(i, j)][(k, l)]
                        W = self.W[(i, j)][(k, l)]

                        # Guard band term: Σ_µ I_{i,j}^{k,l,µ} × G
                        guard = pulp.lpSum(
                            self.I_gb[(i, j)][(k, l)][mu] * G
                            for mu in cfg.numerologies
                        )

                        # (14a): F_j^l + guard + width(j,l) < F_i^k + Φ·(1 − Y + W)
                        self.problem += (
                            self.F[j][l] + guard + width(j, l) <= self.F[i][k] + self.PHI_FREQ * (1 - Y + W) - 1,
                            f"c14a_freq_sep_ji_vue{i}_{j}_slot{k}_{l}"
                        )
                        # (14b): F_i^k + guard + width(i,k) < F_j^l + Φ·(2 − Y − W)
                        self.problem += (
                            self.F[i][k] + guard + width(i, k) <= self.F[j][l] + self.PHI_FREQ * (2 - Y - W) - 1,
                            f"c14b_freq_sep_ij_vue{i}_{j}_slot{k}_{l}"
                        )
                        c14_count += 2
        # print(f"Total constraints added for (14): Inter UE frequency separation with guard bands :  {c14_count}")

        # ── Constraint (15) ───────────────────────────────────────────────────
        # ∀i,j ∈ N, i≠j, ∀k,l ∈ {0..K-1}, ∀µ ∈ M:
        #   I_{i,j}^{k,l,µ} ≥ Σ_ω X_j^{l,µ} - Σ_ω X_i^{k,µ}
        #   I_{i,j}^{k,l,µ} ≥ Σ_ω X_i^{k,µ} - Σ_ω X_j^{l,µ}
        #
        # I_gb is the guard band indicator used in constraint (14).
        # It equals 1 when exactly one of {i,j} uses numerology µ at their
        # respective slots, i.e. the two slots have *different* numerologies.
        # When numerologies differ, subcarriers are no longer orthogonal →
        # Inter-Numerology Interference (INI) → a guard band G must be
        # inserted between the two BWPs in frequency (enforced by (14)).
        #
        # Let s_i(µ) = Σ_ω X_i^{k,µ}_{ω}  and  s_j(µ) = Σ_ω X_j^{l,µ}_{ω}
        # Both are in {0,1} from constraint (5).
        #
        # The two inequalities together force:
        #   I_gb ≥ |s_i(µ) − s_j(µ)|
        # Since I_gb is binary and minimised by the objective (it wastes
        # frequency rows via the guard band in (14)), the solver will set:
        #   I_gb = 1  iff  s_i(µ) ≠ s_j(µ)   (one uses µ, the other doesn't)
        #   I_gb = 0  iff  s_i(µ) = s_j(µ)   (both use µ, or neither does)
        #
        # Note: when both use µ simultaneously they would share the same
        # numerology → no INI → no guard band needed → I_gb = 0 correct ✓
        # When neither uses µ → trivially 0 ✓

        c15_count = 0
        for i in vids:
            for j in vids:
                if i == j:
                    continue
                for k in range(K):
                    for l in range(K):
                        for mu in cfg.numerologies:
                            sum_X_i_mu = pulp.lpSum(
                                self.X[i][k][mu][w]
                                for w in range(1, max_omega(mu, cfg.n_freq_rows) + 1)
                            )
                            sum_X_j_mu = pulp.lpSum(
                                self.X[j][l][mu][w]
                                for w in range(1, max_omega(mu, cfg.n_freq_rows) + 1)
                            )
                            I = self.I_gb[(i, j)][(k, l)][mu]
                            # I ≥ s_j(µ) − s_i(µ)
                            self.problem += (
                                I >= sum_X_j_mu - sum_X_i_mu,
                                f"c15a_guardband_vue{i}_{j}_slot{k}_{l}_mu{mu}"
                            )
                            # I ≥ s_i(µ) − s_j(µ)
                            self.problem += (
                                I >= sum_X_i_mu - sum_X_j_mu,
                                f"c15b_guardband_vue{i}_{j}_slot{k}_{l}_mu{mu}"
                            )
                            c15_count += 2

        # print(f"Total constraints added for (15): Guard band constraints for Inter-Numerology Interference :  {c15_count}")

        # ── Constraint (16) ───────────────────────────────────────────────────
        # ∀i ∈ N, ∀k ∈ {0..K-1}:
        #   T[i][k] + Σ_{µ,ω} E(µ)·X[i][k][µ][ω]  <  latency_slots_i
        #
        # Every PRB block assigned to virtual UE i must finish strictly before
        # its latency deadline. Strict < on integers → implemented as ≤ − 1.
        #
        # latency_slots_i = round(SLA.latency_ms / δT) — precomputed in
        # VirtualUE.latency_slots during build_virtual_ues().
        #
        # When slot k is unassigned: T=-1, dur=0 → LHS=-1 ≤ deadline-1 ✓
        # When assigned: T[i][k] + E(µ) ≤ latency_slots_i − 1
        #   → slot finishes at or before column (latency_slots_i − 1), i.e.
        #     strictly inside the deadline window.
        c16_count = 0
        for vue in self.virtual_ues:
            i   = vue.virtual_id
            dl  = vue.latency_slots   # deadline in grid columns
            for k in range(K):
                self.problem += (
                    self.T[i][k] + dur(i, k) <= dl ,
                    f"c16_latency_vue{i}_slot{k}"
                )
                c16_count += 1
        # print(f"Total constraints added for (16): T[i][k] + dur(i,k) < latency_slots_i :  {c16_count}")

        # ── Constraint (17) ───────────────────────────────────────────────────
        # ∀i ∈ N:  Σ_{k,µ,ω} ω · X[i][k][µ][ω]  ≥  N_PRB_i
        #
        # The cumulative PRB count across all K slots must meet the minimum
        # required to satisfy the throughput SLA, precomputed as vue.n_prb
        # via compute_n_prb(throughput_mbps, delta_t_ms, mcs).
        #
        # Note: this is a single constraint per virtual UE (not per slot),
        # summing over all k. The solver may distribute the PRBs freely
        # across slots and numerologies as long as the total ≥ n_prb.
        c17_count = 0
        for vue in self.virtual_ues:
            i = vue.virtual_id
            self.problem += (
                pulp.lpSum(
                    w * self.X[i][k][mu][w]
                    for k   in range(K)
                    for mu  in cfg.numerologies
                    for w   in range(1, max_omega(mu, cfg.n_freq_rows) + 1)
                ) >= vue.n_prb,
                f"c17_throughput_vue{i}"
            )
            c17_count += 1
        # print(f"Total constraints added for (17): sum_(k,mu,w) w · X[i][k][mu][w]  ≥  N_PRB_i :  {c17_count}")

        # ── Constraint (18) ───────────────────────────────────────────────────
        # ∀g ∈ G, ∀i,j ∈ g, i≠j, ∀k,l ∈ {0..K-1}, ∀µ ∈ M:
        #   Σ_ω X_j^{l,µ} − Σ_ω X_i^{k,µ}  ≤  1 − Y[(i,j)][(k,l)]
        #   Y[(i,j)][(k,l)] − 1              ≤  Σ_ω X_j^{l,µ} − Σ_ω X_i^{k,µ}
        #
        # Purpose: virtual UEs in the same group (= same physical UE) must
        # use the same numerology whenever their time slots overlap.
        # One physical device has a single radio frontend — it cannot run
        # two different FFT sizes at the same time.
        #
        # Let d(µ) = s_j^l(µ) − s_i^k(µ)  where s(µ) ∈ {0,1} is whether
        # that VUE uses numerology µ at that slot.
        #
        # When Y=1 (slots overlap): both inequalities combine to d(µ) = 0
        #   → s_j^l(µ) = s_i^k(µ) for every µ
        #   → since at most one µ is active per slot (constraint 5), both
        #     VUEs must use the same (or no) numerology simultaneously ✓
        #
        # When Y=0 (no time overlap): RHS range is [−1, +1], which is
        #   always satisfied since d(µ) ∈ {−1, 0, +1} ✓
        c18_count = 0
        for gid, members in self.groups.items():
            for i in members:
                for j in members:
                    if i == j:
                        continue
                    for k in range(K):
                        for l in range(K):
                            Y = self.Y[(i, j)][(k, l)]
                            for mu in cfg.numerologies:
                                sum_X_i_mu = pulp.lpSum(
                                    self.X[i][k][mu][w]
                                    for w in range(1, max_omega(mu, cfg.n_freq_rows) + 1)
                                )
                                sum_X_j_mu = pulp.lpSum(
                                    self.X[j][l][mu][w]
                                    for w in range(1, max_omega(mu, cfg.n_freq_rows) + 1)
                                )
                                diff = sum_X_j_mu - sum_X_i_mu  # d(µ)
                                # (18a): d(µ) ≤ 1 − Y
                                self.problem += (
                                    diff <= 1 - Y,
                                    f"c18a_same_numer_g{gid}_vue{i}_{j}_slot{k}_{l}_mu{mu}"
                                )
                                # (18b): Y − 1 ≤ d(µ)
                                self.problem += (
                                    Y - 1 <= diff,
                                    f"c18b_same_numer_g{gid}_vue{i}_{j}_slot{k}_{l}_mu{mu}"
                                )
                                c18_count += 2
        # print(f"Total constraints added for (18): virtual UEs in the same group must use the same numerology whenever their time slots overlap :  {c18_count}")

        # ── Constraint (19) ───────────────────────────────────────────────────
        # ∀g ∈ G, ∀i,j ∈ g, i≠j, ∀k,l ∈ {0..K-1}:
        #   T_i^k − T_j^l  ≤  Φ · (1 − Y[(i,j)][(k,l)])
        #   Φ · (Y[(i,j)][(k,l)] − 1)  ≤  T_i^k − T_j^l
        #
        # Purpose: same-group VUEs whose time slots overlap must start at
        # the same grid column. Together with (18), this ensures they
        # occupy the same time slot, just different frequency bands.
        #
        # When Y=1 (overlap):
        #   (19a): T_i^k − T_j^l ≤ 0  →  T_i^k ≤ T_j^l
        #   (19b): 0 ≤ T_i^k − T_j^l  →  T_j^l ≤ T_i^k
        #   Together: T_i^k = T_j^l  ✓
        #
        # When Y=0 (no overlap): both constraints relax via Φ ✓
        c19_count = 0
        for gid, members in self.groups.items():
            for idx_a, i in enumerate(members):
                for j in members[idx_a + 1:]:
                    for k in range(K):
                        for l in range(K):
                            Y   = self.Y[(i, j)][(k, l)]
                            PHI = self.PHI_TIME
                            # (19a): T_i^k − T_j^l ≤ Φ·(1 − Y)
                            self.problem += (
                                self.T[i][k] - self.T[j][l] <= PHI * (1 - Y),
                                f"c19a_same_start_g{gid}_vue{i}_{j}_slot{k}_{l}"
                            )
                            # (19b): Φ·(Y − 1) ≤ T_i^k − T_j^l
                            self.problem += (
                                PHI * (Y - 1) <= self.T[i][k] - self.T[j][l],
                                f"c19b_same_start_g{gid}_vue{i}_{j}_slot{k}_{l}"
                            )
                            c19_count += 2
        # print(f"Total constraints added for (19): same-group VUEs whose time slots overlap must start at the same grid column :  {c19_count}")
        # print(f"Total constraints added: {c5_count + c6_count + c7_count + c8_count + c9_count + c10_count + c11_count + c13_count + c14_count + c15_count + c16_count + c17_count + c18_count + c19_count}")

        # Constraint (8b) — lower sentinel: T[i][k] ≥ sum_X(i,k) - 1
        # When sum_X=1: T ≥ 0  → blocks the -1 sentinel when slot is active
        # When sum_X=0: T ≥ -1 → already satisfied by variable lower bound (no-op)
        for vue in self.virtual_ues:
            i = vue.virtual_id
            for k in range(K):
                self.problem += (
                    self.T[i][k] >= sum_X(i, k) - 1,
                    f"c8b_sentinel_T_lower_vue{i}_slot{k}"
                )
        # Constraint (9b) — lower sentinel: F[i][k] ≥ sum_X(i,k) - 1
        # Same logic as (8b) but for frequency coordinate
        for vue in self.virtual_ues:
            i = vue.virtual_id
            for k in range(K):
                self.problem += (
                    self.F[i][k] >= sum_X(i, k) - 1,
                    f"c9b_sentinel_F_lower_vue{i}_slot{k}"
                )

    def model_report(self) -> str:
        """
        Return a human-readable report of the built MILP using PuLP's
        built-in inspection API.

        PuLP built-ins used
        ───────────────────
        prob.numVariables()   → total variable count
        prob.numConstraints() → total constraint count
        prob.variables()      → list[LpVariable]  (each has .cat, .lowBound, .upBound)
        prob.objective        → the LpAffineExpression objective
        prob.constraints      → dict[name, LpConstraint]
        pulp.LpStatus[status] → human-readable solver status string

        Raises RuntimeError if build() has not been called yet.
        """
        if self.problem is None:
            raise RuntimeError("Call build() before model_report().")

        import re
        from collections import Counter

        prob      = self.problem
        variables = prob.variables()          # built-in: sorted list of LpVariable

        # ── Variable breakdown ────────────────────────────────────────────────
        # IMPORTANT PuLP internals note:
        #   LpBinary variables are stored internally as cat='Integer' with
        #   lowBound=0, upBound=1.  They are NOT tagged as cat='Binary'.
        #   Use the combination (isInteger + lb=0 + ub=1) to identify binaries.
        n_binary     = sum(1 for v in variables
                           if v.isInteger() and v.lowBound == 0 and v.upBound == 1)
        n_true_int   = sum(1 for v in variables
                           if v.isInteger() and not (v.lowBound == 0 and v.upBound == 1))
        n_continuous = sum(1 for v in variables if not v.isInteger())

        # ── Constraint breakdown by sense ─────────────────────────────────────
        # prob.constraints is a dict {name: LpConstraint}
        # LpConstraint.sense: -1 (>=), 0 (==), 1 (<=)
        sense_map    = {-1: ">=", 0: "==", 1: "<="}
        sense_counts = Counter(
            sense_map.get(c.sense, "?")
            for c in prob.constraints.values()
        )

        # ── Group constraint counts by paper equation number ──────────────────
        # Constraint names follow the pattern "c<N>[a|b|…]_<description>…"
        # e.g. "c13a_overlap_lb_vue0_1_slot2_3" → group under "c13"
        prefix_counts: dict = {}
        for name in prob.constraints:
            prefix = name.split("_")[0]                         # "c13a"
            key    = re.sub(r'[a-z]+$', '', prefix)            # "c13a" → "c13"
            prefix_counts[key] = prefix_counts.get(key, 0) + 1

        W = 66
        lines = [
            "=" * W,
            "PuLP Model Report  —  built-in API summary",
            "=" * W,
            f"  Problem name      : {prob.name}",
            f"  Sense             : {'Maximise' if prob.sense == pulp.LpMaximize else 'Minimise'}",
            f"  Solver status     : {pulp.LpStatus[prob.status]}",
            "",
            "  ── Variables  (prob.numVariables() / prob.variables()) ─────",
            f"  Total             : {prob.numVariables():>8}",
            f"    Binary (0/1 int) : {n_binary:>8}   (X, Y, Z, W, I_gb — stored as Integer lb=0 ub=1)",
            f"    General integer  : {n_true_int:>8}   (T, F — unbounded integer)",
            f"    Continuous       : {n_continuous:>8}",
            "",
            "  ── Constraints  (prob.numConstraints()) ────────────────────",
            f"  Total             : {prob.numConstraints():>8}",
            f"    (<= ) sense     : {sense_counts.get('<=', 0):>8}",
            f"    (>= ) sense     : {sense_counts.get('>=', 0):>8}",
            f"    ( == ) sense    : {sense_counts.get('==',  0):>8}",
            "",
            "  ── Per-equation breakdown  (paper numbering) ───────────────",
            f"  {'Eq.':>5}  {'Count':>8}",
            "  " + "─" * 18,
        ]
        for key in sorted(prefix_counts,
                          key=lambda x: int(re.sub(r'\D', '', x) or 0)):
            lines.append(f"  {key:>5}  {prefix_counts[key]:>8}")

        n_obj_terms = len(prob.objective) if prob.objective else 0
        lines += [
            "",
            "  ── Objective  (prob.objective) ─────────────────────────────",
            f"  Non-zero terms    : {n_obj_terms:>8}",
            "=" * W,
        ]
        return "\n".join(lines)

    # ----------------------------------------------------------------
    # Private — Objective
    # ----------------------------------------------------------------

    def _set_objective(self):
        """
        Objective (20): maximise total PRBs allocated.

        max  Σ_i  Σ_k  Σ_µ  Σ_ω  ω · X[i][k][µ][ω]

        No UE priorities — all weighted equally.
        To add priority weights, multiply each term by a per-UE weight w_i.
        """
        cfg = self.config
        self.problem += (
            pulp.lpSum(
                w * self.X[vue.virtual_id][k][mu][w]
                for vue in self.virtual_ues
                for k   in range(cfg.K)
                for mu  in cfg.numerologies
                for w   in range(1, max_omega(mu, cfg.n_freq_rows) + 1)
            ),
            "maximise_bandwidth_utilisation"
        )

    @classmethod
    def get_bwp_allocation_schedule(cls, config_path: str, solution_path: str, bwp_output: str = None) -> str:
        """
        Extracts BWP configurations and switching times for each Physical UE 
        from a saved solution file.
        """
        # 1. Load context from config file
        config, physical_ues = load_config(config_path)
        virtual_ues, _ = build_virtual_ues(physical_ues, config)

        # 2. Parse the solution file into a dictionary
        sol: Dict[str, float] = {}
        with open(solution_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        sol[parts[0]] = float(parts[1])
                    except ValueError: pass

        schedule = {}

        # 3. Iterate through Physical UEs to aggregate slice allocations
        for pue in physical_ues:
            pue_slots = []
            # Map physical UE to its associated virtual IDs
            associated_vids = [v.virtual_id for v in virtual_ues if v.group_id == pue.ue_id]
            
            for k in range(config.K):
                active_mu = None
                f_min = float('inf')
                f_max = 0
                t_start = -1
                
                # Check if any virtual slice for this UE is active in slot k
                for vid in associated_vids:
                    for mu in config.numerologies:
                        # Find the active X variable for this slot
                        for w in range(1, max_omega(mu, config.n_freq_rows) + 1):
                            var_name = f"X_{vid}_{k}_{mu}_{w}"
                            if abs(sol.get(var_name, 0.0) - 1.0) < 1e-4:
                                active_mu = mu
                                f_start = int(round(sol.get(f"F_{vid}_{k}", 0.0)))
                                t_start = int(round(sol.get(f"T_{vid}_{k}", 0.0)))
                                current_w = w * G_rows(mu)
                                
                                # Aggregate BWP boundaries
                                f_min = min(f_min, f_start)
                                f_max = max(f_max, f_start + current_w)
                
                if active_mu is not None:
                    duration_cols = E(active_mu, config.mu_max)
                    pue_slots.append({
                        "slot_index": k,
                        "t_start_ms": t_start * config.delta_T,
                        "duration_ms": duration_cols * config.delta_T,
                        "mu": active_mu,
                        "bwp_start_row": f_min,
                        "bwp_width_rows": f_max - f_min,
                        "bwp_center_freq_hz": (f_min + (f_max - f_min)/2) * config.delta_F
                    })
            
            schedule[pue.ue_id] = sorted(pue_slots, key=lambda x: x['t_start_ms'])

        # 4. Export to JSON
        out_json = solution_path.replace(".txt", "_bwp_schedule.json")
        # Create output directory if it doesn't exist
        if bwp_output and not os.path.exists(bwp_output):
            os.makedirs(bwp_output)
        if bwp_output:
            out_json = os.path.join(bwp_output, os.path.basename(out_json))
        with open(out_json, "w") as f:
            json.dump(schedule, f, indent=4)
            
        return out_json
    # ----------------------------------------------------------------
    # Plot the resource grid allocation for visual verification
    # ----------------------------------------------------------------

    @classmethod
    def plot_solution(
        cls,
        config_path: str,
        solution_path: str,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> str:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import matplotlib.patheffects as patheffects
            import matplotlib.cm as cm
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "matplotlib and numpy are required for plot_solution."
            ) from exc

        # -- Load config --
        config, physical_ues = load_config(config_path)
        virtual_ues, _ = build_virtual_ues(physical_ues, config)

        c        = config
        mu_max   = c.mu_max
        K        = c.K
        N_COLS   = c.n_time_cols
        N_ROWS   = c.n_freq_rows

        # -- [FIX] Dynamic Palette Setup --
        # Use the modern colormaps API to resolve the MatplotlibDeprecationWarning
        n_vues = len(virtual_ues)
        colormap = matplotlib.colormaps.get_cmap('turbo').resampled(n_vues)

        # Create a persistent mapping for color and labels
        vue_colour = {v.virtual_id: colormap(idx) for idx, v in enumerate(virtual_ues)}
        vue_label  = {v.virtual_id: f"{v.physical_ue.ue_id} ({v.sla.slice_id})" for v in virtual_ues}

        # -- Parse solution file --
        sol: Dict[str, float] = {}
        with open(solution_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        sol[parts[0]] = float(parts[1])
                    except ValueError: pass

        def _X_val(i, k):
            for key, val in sol.items():
                if key.startswith(f"X_{i}_{k}_") and abs(val - 1.0) < 1e-4:
                    parts = key.split("_")
                    return int(parts[3]), int(parts[4])
            return None

        # -- Figure Creation --
        fig_w = max(12, N_COLS * 0.4)
        fig_h = max(8, N_ROWS * 0.35)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor='#F8F9FA')
        ax.set_facecolor('#FFFFFF')
        ax.set_xlim(0, N_COLS)
        ax.set_ylim(0, N_ROWS)

        # -- Draw Allocations with 3D Effect --
        used_vues = set()
        for vue in virtual_ues:
            i = vue.virtual_id
            for k in range(K):
                res = _X_val(i, k)
                if res is None: continue
                mu, omega = res
                t_start = int(round(sol.get(f"T_{i}_{k}", 0.0)))
                f_start = int(round(sol.get(f"F_{i}_{k}", 0.0)))
                w_cols  = E(mu, mu_max)
                prb_height = G_rows(mu)

                for p in range(omega):
                    prb_f_start = f_start + (p * prb_height)
                    
                    # 1. Main PRB Rectangle with compatible Shadow
                    # SimplePatchShadow only reliably takes offset, alpha, and rho
                    rect = mpatches.Rectangle(
                        (t_start, prb_f_start), w_cols, prb_height,
                        facecolor=vue_colour[i], edgecolor="#333333", 
                        linewidth=0.8, alpha=0.4, zorder=4
                    )
                    # rect.set_path_effects([
                    #     patheffects.SimplePatchShadow(offset=(1.5, -1.5), alpha=0.4)
                    # ])
                    ax.add_patch(rect)
                    
                    # 2. 3D "Gloss" Effect (Light overlay on top half)
                    gloss = mpatches.Rectangle(
                        (t_start, prb_f_start + prb_height*0.15), w_cols, prb_height*0.5,
                        facecolor=vue_colour[i], edgecolor='none', alpha=0.05, zorder=5
                    )
                    ax.add_patch(gloss)
                
                # Allocation Labels
                total_height = omega * prb_height
                
                # Get MCS data for the parent physical UE
                pue = vue.physical_ue
                Q, r = _MCS_TABLE[pue.mcs]
                mbits_per_prb = (168 * Q * r) / 1e6

                # Update the label to show total Mbits for the entire allocation in that slot
                total_mbits = mbits_per_prb * omega 

                txt = ax.text(
                    t_start + w_cols/2, f_start + total_height/2,
                    fr"$\mu$={mu}, $\omega$={omega}" + "\n" + f"{total_mbits:.3f} Mb",
                    color="white", weight="bold", ha="center", va="center", 
                    fontsize=9, zorder=10
                )

                txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground='black', alpha=0.7)])
                used_vues.add(i)

        # -- Aesthetics --
        ax.set_xlabel(fr"Grid Columns (Unit: $\delta T$ = {c.delta_T}ms)", fontsize=11, labelpad=10)
        ax.set_ylabel(fr"Grid Rows (Unit: $\delta F$ = 180kHz)", fontsize=11, labelpad=10)
        
        # Double X-Axis
        secax = ax.secondary_xaxis('top', functions=(lambda x: x * c.delta_T, lambda x: x / c.delta_T))
        secax.set_xlabel('Time (ms)', fontsize=12, fontweight='bold', labelpad=15)

        # Clean Grid
        ax.grid(True, which='major', color='#E0E0E0', linestyle='-', linewidth=0.7, zorder=1)
        ax.set_xticks(np.arange(0, N_COLS + 1, 1))
        ax.set_yticks(np.arange(0, N_ROWS + 1, 1))
        ax.tick_params(axis='both', which='major', labelsize=9)

        # -- [FIXED LEGEND] --
        # Create handles that explicitly match the facecolor, alpha, and edgecolor of the PRBs
        handles = [
            mpatches.Patch(
                facecolor=vue_colour[vid], 
                label=vue_label[vid], 
                edgecolor='#333333', 
                linewidth=0.8,
                alpha=0.4  # Matches the alpha used in the grid rectangles
            ) 
            for vid in sorted(used_vues)
        ]
        
        ax.legend(
            handles=handles, 
            loc="upper left", 
            bbox_to_anchor=(1.02, 1), 
            title="UE Allocations", 
            title_fontsize='11', 
            frameon=True, 
            shadow=True
        )

        plt.suptitle(title or "5G NR Resource Allocation Grid", fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])

        if output_path is None:
            # Handle potential directory vs file path from main()
            output_path = os.path.splitext(solution_path)[0] + "_professional_grid.png"
        else:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            base = os.path.basename(solution_path).replace(".txt", "_professional_grid.png")
            output_path = os.path.join(output_path, base)

        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return output_path
    # ----------------------------------------------------------------

    def solve(self,mps_path: str|None = None, time_limit: int = 50_000):
        pass 
    
    def verify_solution(self,mps_path: str|None = None, sol_path: str|None = None) -> bool:
        pass

    def verify_constraints(self) -> Dict[str, bool]:
        pass
    
    def verify_objective(self) -> bool:
        pass    
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