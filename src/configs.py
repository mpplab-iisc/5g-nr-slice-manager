from dataclasses import dataclass, field
from typing import Dict, List, Tuple


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

