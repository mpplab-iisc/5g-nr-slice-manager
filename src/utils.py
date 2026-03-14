
from typing import Dict, List, Tuple
import math
import os
import json

from .configs import _MCS_TABLE, _RE_PER_PRB, SystemConfig, PhysicalUE, SliceSLA, VirtualUE



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
