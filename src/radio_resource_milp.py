import sys
import os
import tqdm
import json
from typing import Any, Dict, List, Optional

from .utils import load_config , build_virtual_ues, max_omega, G_rows, E
from .configs import SystemConfig, PhysicalUE, _MCS_TABLE

try:
    import pulp
except ImportError:
    print("ERROR: PuLP not installed.  Run: pip install pulp --break-system-packages")
    sys.exit(1)

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

        # FIX: PuLP's writeMPS() writes '*SENSE:Maximize' as a comment (line
        # starting with *), which HiGHS ignores — it then defaults to MINIMIZE.
        # Inject a proper 'OBJSENSE / MAX' section after the NAME line so that
        # HiGHS correctly maximises the objective.
        if self.problem.sense == -1:  # pulp.LpMaximize == -1
            with open(mps_path, 'r') as f:
                mps_lines = f.read().split('\n')
            patched = []
            for line in mps_lines:
                patched.append(line)
                if line.startswith('NAME'):
                    patched.append('OBJSENSE')
                    patched.append('    MAX')
            with open(mps_path, 'w') as f:
                f.write('\n'.join(patched))

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
                            self.T[i][k] <= self.T[j][l] + dur(j, l) + PHI * (1 - Y) - 1,
                            f"c13b_overlap_ub_vue{i}_{j}_slot{k}_{l}"
                        )
                        # (13c): T_i^k + dur(i,k) < T_j^l + Φ·(Y + Z)
                        #        → ≤ T_j^l + Φ·(Y + Z) − 1  (integer strict →  ≤ − 1)
                        # self.problem += (
                        #     self.T[i][k] + dur(i, k) <= self.T[j][l] + PHI * (Y + Z) - 1,
                        #     f"c13c_nonoverlap_ij_vue{i}_{j}_slot{k}_{l}"
                        # )
                        self.problem += (
                            self.T[i][k] + dur(i, k) <= self.T[j][l] + PHI * (Y + Z) ,
                            f"c13c_nonoverlap_ij_vue{i}_{j}_slot{k}_{l}"
                        )
                        # (13d): T_j^l + dur(j,l) < T_i^k + Φ·(Y + 1 − Z)
                        #        → ≤ T_i^k + Φ·(Y + 1 − Z) − 1
                        self.problem += (
                            self.T[j][l] + dur(j, l) <= self.T[i][k] + PHI * (Y + 1 - Z)  ,
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
                        # Use non strict since all F values are integers
                        self.problem += (
                            self.F[j][l] + guard + width(j, l) <= self.F[i][k] + self.PHI_FREQ * (1 - Y + W) ,
                            f"c14a_freq_sep_ji_vue{i}_{j}_slot{k}_{l}"
                        )
                        # (14b): F_i^k + guard + width(i,k) < F_j^l + Φ·(2 − Y − W)
                        self.problem += (
                            self.F[i][k] + guard + width(i, k) <= self.F[j][l] + self.PHI_FREQ * (2 - Y - W) ,
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

        import colorsys
        # Assign one base colour per physical UE; embb uses the base colour at
        # full saturation while urllc uses a lighter, desaturated tint so that
        # both slices of the same UE are visually related in the legend.
        phy_ue_list = list(dict.fromkeys(v.physical_ue.ue_id for v in virtual_ues))
        n_phy       = len(phy_ue_list)
        base_cmap   = matplotlib.colormaps.get_cmap('tab10')
        vue_colour: Dict = {}
        for v in virtual_ues:
            p_idx = phy_ue_list.index(v.physical_ue.ue_id)
            r, g, b, a = base_cmap(p_idx / max(1, n_phy))
            if v.sla.slice_id != 'urllc':
                vue_colour[v.virtual_id] = (r, g, b, a)
            else:
                h, l, s = colorsys.rgb_to_hls(r, g, b)
                r2, g2, b2 = colorsys.hls_to_rgb(h, 0.80, s * 0.55)
                vue_colour[v.virtual_id] = (r2, g2, b2, a)
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

    @classmethod
    def plot_cg_solution(
        cls,
        config_path:   str,
        solution_path: str,
        columns_path:  Optional[str] = None,
        output_path:   Optional[str] = None,
        title:         Optional[str] = None,
    ) -> str:
        """
        Plot the resource-grid for a CG master solution produced by an external
        solver (Gurobi, HiGHS, …).

        Parameters
        ----------
        config_path   : path to the JSON config used to build the CG master MPS
        solution_path : .sol or .txt file from the external solver (Gurobi .sol
                        format or HiGHS column solution format).  Lines with
                        ``lam_<i>_<c>  1`` (value ≈ 1) identify the selected
                        column for each VUE.
        columns_path  : path to the ``_columns.json`` registry written by
                        ``solve_integer_master`` alongside the MPS file.
                        If omitted, auto-detected by replacing the solution
                        file's extension with ``_columns.json`` *and* by
                        looking for ``*_columns.json`` next to the MPS in
                        --cg-output directories.
        output_path   : directory or file path for the output PNG.
                        Defaults to same directory as solution_path.
        title         : plot title (optional).

        Returns
        -------
        Path to the saved PNG file.
        """
        import re
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import matplotlib.patheffects as patheffects
            import numpy as np
        except ImportError as exc:
            raise ImportError("matplotlib and numpy are required for plot_cg_solution.") from exc

        # -- Load config --
        config, physical_ues = load_config(config_path)
        virtual_ues, _ = build_virtual_ues(physical_ues, config)

        c      = config
        mu_max = c.mu_max
        K      = c.K
        N_COLS = c.n_time_cols
        N_ROWS = c.n_freq_rows

        import colorsys
        # Assign one base colour per physical UE; embb uses the base colour at
        # full saturation while urllc uses a lighter, desaturated tint so that
        # both slices of the same UE are visually related in the legend.
        phy_ue_list = list(dict.fromkeys(v.physical_ue.ue_id for v in virtual_ues))
        n_phy       = len(phy_ue_list)
        base_cmap   = matplotlib.colormaps.get_cmap('tab10')
        vue_colour: Dict = {}
        for v in virtual_ues:
            p_idx = phy_ue_list.index(v.physical_ue.ue_id)
            r, g, b, a = base_cmap(p_idx / max(1, n_phy))
            if v.sla.slice_id != 'urllc':
                vue_colour[v.virtual_id] = (r, g, b, a)
            else:
                # Lighter tint: shift lightness toward 0.80
                h, l, s = colorsys.rgb_to_hls(r, g, b)
                r2, g2, b2 = colorsys.hls_to_rgb(h, 0.80, s * 0.55)
                vue_colour[v.virtual_id] = (r2, g2, b2, a)
        vue_label  = {v.virtual_id: f"{v.physical_ue.ue_id} ({v.sla.slice_id})"
                      for v in virtual_ues}

        # -- Auto-detect columns registry --
        if columns_path is None:
            # Try sibling file: same base name as solution but ending _columns.json
            sol_base = os.path.splitext(solution_path)[0]
            candidate = sol_base + "_columns.json"
            if os.path.isfile(candidate):
                columns_path = candidate
            else:
                # Walk the directory of the solution file looking for *_columns.json
                sol_dir = os.path.dirname(os.path.abspath(solution_path))
                matches = [os.path.join(sol_dir, f) for f in os.listdir(sol_dir)
                           if f.endswith("_columns.json")]
                if len(matches) == 1:
                    columns_path = matches[0]
                elif len(matches) > 1:
                    # Pick the one whose stem most closely matches the solution file
                    sol_stem = os.path.basename(sol_base)
                    # strip trailing suffix differences
                    best = max(matches, key=lambda p: len(
                        os.path.commonprefix([os.path.basename(p), sol_stem])))
                    columns_path = best

        if columns_path is None or not os.path.isfile(columns_path):
            raise FileNotFoundError(
                f"Cannot find _columns.json registry. "
                f"Tried: {columns_path!r}. "
                "Pass --cg-columns <path> explicitly."
            )

        with open(columns_path) as f:
            registry: Dict[str, Dict[str, Any]] = json.load(f)

        # -- Parse solution file for lam_i_c = 1 --
        lam_pat = re.compile(r"^lam_(\d+)_(\d+)\s+(\S+)")
        selected: Dict[int, Optional[Dict[str, Any]]] = {}
        with open(solution_path) as f:
            for line in f:
                m = lam_pat.match(line.strip())
                if m:
                    vue_id  = int(m.group(1))
                    c_idx   = int(m.group(2))
                    try:
                        val = float(m.group(3))
                    except ValueError:
                        continue
                    if abs(val - 1.0) < 1e-4:
                        vue_str = str(vue_id)
                        c_str   = str(c_idx)
                        if vue_str in registry and c_str in registry[vue_str]:
                            selected[vue_id] = registry[vue_str][c_str]

        if not selected:
            raise ValueError(
                f"No lam_i_c = 1 entries found in {solution_path!r}. "
                "Check solver output format or solution feasibility."
            )

        # -- Figure --
        fig_w = max(12, N_COLS * 0.4)
        fig_h = max(8,  N_ROWS * 0.35)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor='#F8F9FA')
        ax.set_facecolor('#FFFFFF')
        ax.set_xlim(0, N_COLS)
        ax.set_ylim(0, N_ROWS)

        used_vues: set = set()
        for vue in virtual_ues:
            i   = vue.virtual_id
            col = selected.get(i)
            if col is None or col["total_prbs"] == 0:
                continue
            for k in range(K):
                w_val  = col["w"][k]
                mu_val = col["mu"][k]
                t_val  = col["t"][k]
                f_val  = col["f"][k]
                if w_val == 0 or mu_val is None or t_val < 0:
                    continue

                w_cols     = E(mu_val, mu_max)
                prb_height = G_rows(mu_val)

                # w_val = omega (number of PRBs stacked in frequency)
                for p in range(w_val):
                    prb_f_start = f_val + p * prb_height
                    rect = mpatches.Rectangle(
                        (t_val, prb_f_start), w_cols, prb_height,
                        facecolor=vue_colour[i], edgecolor="#333333",
                        linewidth=0.8, alpha=0.4, zorder=4,
                    )
                    ax.add_patch(rect)
                    gloss = mpatches.Rectangle(
                        (t_val, prb_f_start + prb_height * 0.15), w_cols, prb_height * 0.5,
                        facecolor=vue_colour[i], edgecolor='none', alpha=0.05, zorder=5,
                    )
                    ax.add_patch(gloss)

                total_height = w_val * prb_height
                pue = vue.physical_ue
                Q, r = _MCS_TABLE[pue.mcs]
                mbits_per_prb = (168 * Q * r) / 1e6
                total_mbits   = mbits_per_prb * w_val

                txt = ax.text(
                    t_val + w_cols / 2, f_val + total_height / 2,
                    fr"$\mu$={mu_val}, $\omega$={w_val}" + "\n" + f"{total_mbits:.3f} Mb",
                    color="white", weight="bold", ha="center", va="center",
                    fontsize=9, zorder=10,
                )
                txt.set_path_effects(
                    [patheffects.withStroke(linewidth=2, foreground='black', alpha=0.7)]
                )
                used_vues.add(i)

        ax.set_xlabel(
            fr"Grid Columns (Unit: $\delta T$ = {c.delta_T}ms)", fontsize=11, labelpad=10
        )
        ax.set_ylabel(fr"Grid Rows (Unit: $\delta F$ = 180kHz)", fontsize=11, labelpad=10)
        secax = ax.secondary_xaxis(
            'top', functions=(lambda x: x * c.delta_T, lambda x: x / c.delta_T)
        )
        secax.set_xlabel('Time (ms)', fontsize=12, fontweight='bold', labelpad=15)

        ax.grid(True, which='major', color='#E0E0E0', linestyle='-', linewidth=0.7, zorder=1)
        ax.set_xticks(np.arange(0, N_COLS + 1, 1))
        ax.set_yticks(np.arange(0, N_ROWS + 1, 1))
        ax.tick_params(axis='both', which='major', labelsize=9)

        handles = [
            mpatches.Patch(
                facecolor=vue_colour[vid], label=vue_label[vid],
                edgecolor='#333333', linewidth=0.8, alpha=0.4,
            )
            for vid in sorted(used_vues)
        ]
        ax.legend(
            handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1),
            title="UE Allocations", title_fontsize='11', frameon=True, shadow=True,
        )

        plt.suptitle(
            title or "5G NR Resource Allocation Grid (Column Generation)",
            fontsize=16, fontweight='bold', y=0.98,
        )
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])

        # -- Save --
        sol_base = os.path.basename(solution_path)
        png_name = os.path.splitext(sol_base)[0] + "_cg_grid.png"
        if output_path is None:
            out = os.path.join(os.path.dirname(os.path.abspath(solution_path)), png_name)
        elif os.path.isdir(output_path) or output_path.endswith(os.sep):
            os.makedirs(output_path, exist_ok=True)
            out = os.path.join(output_path, png_name)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            out = output_path

        fig.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return out

    def solve(self,mps_path: str|None = None, time_limit: int = 50_000):
        pass
    
    def verify_solution(self,mps_path: str|None = None, sol_path: str|None = None) -> bool:
        pass

    def verify_constraints(self) -> Dict[str, bool]:
        pass
    
    def verify_objective(self) -> bool:
        pass    