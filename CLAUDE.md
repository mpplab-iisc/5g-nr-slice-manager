Deep Analysis: HPC Parallelization of column_gen.py

  1. Current Parallel Structure (What Already Runs in Parallel)

  Phase 0  [already parallel, intra-node]:
    ProcessPoolExecutor → N workers → N single-UE MIPs concurrently
    ✓ Embarrassingly parallel — zero inter-VUE coupling

  Phase 1  [PARTIALLY parallel, intra-node]:
    LP master solve  ─── SEQUENTIAL (runs on 1 thread, CBC)
    Pricing          ─── parallel (ProcessPoolExecutor, N workers)
    IPC serialization: pi dict (all dual prices) pickled per worker per
  iteration

  Phase 2  [SEQUENTIAL]:
    Single large Set Partitioning MIP, CBC single-threaded
    Guard band cut detection: O(N² × C²) pure Python loop

  Phase 3  [SEQUENTIAL by design]:
    Coordinate search — each VUE depends on all others' current solution

  The actual runtime breakdown matters here. Read the log format in the code:
  Iter N: LP obj = X  new cols = Y  [lp=Xs  pricing=Xs  iter=Xs]
  Until you profile a real run and see the lp vs pricing time split, speedup
  estimates are just guesses. That said, here's the structural analysis:

  ---
  2. The True Bottlenecks (What the Code Comments Reveal)

  Bottleneck 1: LP master is rebuilt from scratch every iteration
  # CACHE NOTE: _build_master_problem rebuilds the cell→users index from
  # scratch on every call. An incremental dict updated on column append
  # would avoid this O(|cells|×N×C) rebuild every iteration.
  Every CG iteration calls _build_master_problem() which re-scans every cell of
  every column of every VUE. As the column pool grows (say 20 cols × 6 VUEs ×
  100 cells each), this is 12,000 Python dict lookups per iteration, done in a
  pure Python triple loop. This scales badly.

  Bottleneck 2: IPC overhead for pricing workers
  # cells_raw is NOT passed: pickling 13,044 frozensets per VUE
  # costs ~0.54s/iter in IPC overhead, more than the time saved.
  # Workers call _cells_of_assignment directly instead.
  The code already had to back off from passing pre-computed caches across
  process boundaries because pickling cost exceeded the compute benefit. This is
   the single most important constraint on intra-node scaling.

  Bottleneck 3: Guard band cut detection is O(N² × C²)
  for vi_idx in range(len(vue_list)):
      for vj_idx in range(vi_idx + 1, len(vue_list)):
          for ci, col_i in enumerate(cols_i):
              for cj, col_j in enumerate(cols_j):
                  if _columns_guard_band_conflict(col_i, col_j, self.config):
  For 10 VUEs × 20 columns each: 10×9/2 × 400 = 18,000 conflict checks, each
  involving slot-by-slot time overlap detection. Pure Python, no vectorization.

  Bottleneck 4: CBC is single-threaded per subproblem
  Every pricing call uses PULP_CBC_CMD(msg=0, timeLimit=30) — CBC is
  single-threaded. So N pricing workers × 1 thread each = N cores used, but you
  cannot use more cores than VUEs.

  ---
  3. How to Parallelize Across 4 Nodes

  Tier 1: Inter-node — MPI master-worker via mpi4py

  The algorithm maps naturally to MPI master-worker because the communication
  pattern in Phase 1 is:
  - Master (rank 0): solve LP master → broadcast pi and mu_dual → collect new
  columns
  - Workers (ranks 1–3): solve pricing subproblems for their assigned VUEs →
  return new columns

  Node 0 (master rank):     LP master solve (sequential) + pricing for VUEs
  [0..N/4)
  Node 1 (worker rank 1):   pricing for VUEs [N/4..N/2)
  Node 2 (worker rank 2):   pricing for VUEs [N/2..3N/4)
  Node 3 (worker rank 3):   pricing for VUEs [3N/4..N)

  Communication volume (this is what makes it viable):
  - Broadcast pi: n_time_cols × n_freq_rows floats = 24 × 27 = 648 floats ≈ 5 KB
   per iteration
  - Broadcast mu_dual: N floats ≈ negligible
  - Gather new columns: K × 4 ints per column per VUE ≈ negligible

  At 5 KB per iteration, even a 1 GbE link (125 MB/s) takes 40 microseconds for
  the broadcast. This is completely dominated by the compute time. The
  communication overhead is negligible.

  Framework recommendation: mpi4py with MPI.COMM_WORLD.Bcast() /
  MPI.COMM_WORLD.Gather(). Not Dask, not Ray, not Celery. Those add abstraction
  layers that make the synchronization point (every LP iteration) harder to
  express cleanly. Ray is the second choice if you want easier setup.

  Tier 2: Intra-node — Switch from CBC to HiGHS (multi-threaded)

  The current code uses PULP_CBC_CMD (CBC, single-threaded) for everything.
  HiGHS supports parallel simplex and parallel branch & bound natively. Drop-in
  replacement:

  prob.solve(pulp.HiGHS_CMD(msg=0, timeLimit=120, threads=N_CORES_PER_NODE))

  For the LP master (which is the sequential bottleneck), a multi-threaded
  simplex can cut LP solve time by 2–4x depending on the problem size. This is
  free speedup requiring almost zero code changes.

  For pricing subproblems (small MIPs), multi-threaded B&B on a tiny problem
  doesn't help much — better to keep 1 thread per subproblem and run N
  subproblems in parallel (current approach). But for the Phase 2 integer master
   (one large MIP), multi-threaded HiGHS B&B is a direct win.

  Dual-level parallelism architecture:

                     ┌─────────────────────────────┐
                     │  Node 0 (MPI rank 0)         │
                     │  ─ LP master solve           │
                     │    (HiGHS, 8 threads)        │
                     │  ─ Broadcast pi, mu_dual     │
                     │  ─ Pricing: VUEs 0,1         │
                     │    (ProcessPoolExecutor, 2)  │
                     └──────────┬──────────────────┘
            MPI Broadcast pi    │
            ┌────────────┬──────┴──────────┐
            ▼            ▼                 ▼
     ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
     │ Node 1      │ │ Node 2      │ │ Node 3      │
     │ VUEs 2,3    │ │ VUEs 4,5    │ │ VUEs 6,7    │
     │ (2 procs)   │ │ (2 procs)   │ │ (2 procs)   │
     └─────────────┘ └─────────────┘ └─────────────┘
            │            │                 │
            └────────────┴─────────────────┘
                    MPI Gather new columns

  ---
  4. Realistic Speedup — Amdahl's Law Applied Carefully

  Let's define the serial fraction (parts that cannot be parallelized across
  nodes):
  - LP master solve: fully sequential
  - _build_master_problem() cell-index rebuild: sequential
  - Guard band cut detection in Phase 2: sequential
  - Phase 3 coordinate search: sequential

  For a 6-VUE config (3 physical UEs × 2 slices each), say the per-iteration
  timing is:
  LP master solve:    1.0s   (serial — stays on master node)
  Pricing (6 VUEs):   3.0s   (parallelizable — currently 6 workers, wall time =
  max worker)

  Current behavior (intra-node, 6 workers): 1.0 + 3.0/6 = 1.5s/iter (if all
  pricing takes equal time)

  With 4 nodes (distributing 6 VUEs across 4 nodes, ~1.5 VUEs/node):
  LP master:    1.0s  (unchanged, serial)
  Pricing:      3.0s / 4 nodes = 0.75s wall time (if perfect balance, 2
  VUEs/node)
  Per-iteration: 1.0 + 0.75 = 1.75s
  That's 1.75s vs 1.5s — actually slower because the 6-worker intra-node case
  was already hitting the limit imposed by the serial LP master. The LP master
  dominates.

  This illustrates the key insight: with f_serial = 1.0/1.5 = 0.67, Amdahl gives
   max speedup = 1 / 0.67 = 1.5x no matter how many nodes you add. The LP master
   solve is the wall.

  For a larger instance (10 UEs × 2 slices = 20 VUEs), where pricing dominates:
  LP master:    3.0s   (grows with column pool size)
  Pricing:     20.0s   (20 VUEs × 1s each, currently bottlenecked at N_cores)
  Single-node with 8 cores: 3.0 + 20.0/8 = 5.5s/iter
  4 nodes × 8 cores each (32 workers): 3.0 + 20.0/32 = 3.6s/iter → 1.5x speedup
  If you also multi-thread the LP master (HiGHS, 4x): 0.75 + 0.63 = 1.38s/iter →
   4x over baseline

  Summary of realistic speedup with 4 nodes + HiGHS:

  ┌────────────┬────────────────┬─────────────────────┬───────────────────┐
  │   Phase    │    Current     │ With 4-node MPI +   │      Speedup      │
  │            │                │        HiGHS        │                   │
  ├────────────┼────────────────┼─────────────────────┼───────────────────┤
  │ Phase 0    │ Intra-node     │ Scatter across      │ ~4x (if I/O isn't │
  │            │ parallel       │ nodes               │  limiting)        │
  ├────────────┼────────────────┼─────────────────────┼───────────────────┤
  │ Phase 1 LP │ 1 thread CBC   │ HiGHS multi-thread  │ 2–4x              │
  │            │                │ on master           │                   │
  ├────────────┼────────────────┼─────────────────────┼───────────────────┤
  │ Phase 1    │ N intra-node   │ N/4 workers × 4     │ 1.5–3x wall time  │
  │ Pricing    │ workers        │ nodes               │                   │
  ├────────────┼────────────────┼─────────────────────┼───────────────────┤
  │ Phase 2    │ 1 thread CBC   │ HiGHS parallel B&B  │ 2–4x              │
  │ MIP        │                │                     │                   │
  ├────────────┼────────────────┼─────────────────────┼───────────────────┤
  │ Phase 3    │ Sequential     │ Still sequential    │ 1x                │
  ├────────────┼────────────────┼─────────────────────┼───────────────────┤
  │ Overall    │ 1x             │ ~2–3x               │ ~2–3x             │
  └────────────┴────────────────┴─────────────────────┴───────────────────┘

  The honest ceiling is 3–4x end-to-end for large instances, <2x for small
  instances. The LP master is the fundamental serial bottleneck. No amount of
  worker-node scaling changes that.

  ---
  5. C++ / BLACS / cuBLAS — Honest Assessment

  BLACS: Zero benefit. BLACS is distributed dense linear algebra (ScaLAPACK).
  The LP simplex uses sparse factorization (LU on the basis matrix), not dense
  matrix multiplication. BLACS does not participate in LP/MIP solving in any
  existing open-source solver. This is simply the wrong tool.

  cuBLAS: Zero benefit for the same reason. GPU BLAS accelerates dense matrix
  operations (DGEMM, etc.). The LP pricing subproblems here have ~300–1000
  variables each. The GPU kernel launch overhead alone (~5–10μs) would dwarf the
   solve time. cuBLAS is designed for matrices in the millions-of-elements
  range, not 1000-variable sparse MIPs. Even cuOpt (NVIDIA's GPU combinatorial
  optimizer) is designed for vehicle routing with thousands of stops — not this
  structure.

  C++ rewrite generally: The solver calls (prob.solve(CBC_CMD)) already execute
  in C++ — PuLP is just a Python wrapper generating the LP and calling CBC which
   is written in C. The Python code only builds the problem structure and parses
   results. So porting _build_master_problem() to C++ would speed up the problem
   setup by 10–30x, but if setup is 10% of total time and solve is 90%, the net
  speedup is ~11%. Not worth months of engineering.

  Where C++ genuinely would help:

  The pricing subproblem has exploitable structure. Each VUE's pricing problem
  is essentially: place K non-overlapping rectangles in a 2D grid, maximizing
  total width minus shadow costs, subject to time ordering and latency
  constraints. This is structurally a variant of weighted interval scheduling
  with 2D resources, which has known polynomial DP solutions for special cases.
  A hand-written C++ DP solver for the pricing subproblem could be:
  - 50–200x faster than CBC on each pricing call (CBC is overkill for this
  structure)
  - Remove the 30-second time limit entirely (DP is exact and fast)
  - Eliminate all the PuLP overhead (variable creation, model building) per call

  This is the highest-leverage C++ work: replace CBC in pricing with a custom
  DP, keep everything else. That's targeted, high-payoff, and doesn't require
  rewriting the whole system.

  ---
  6. Priority-Ordered Recommendations

  ┌──────────┬───────────────────────────────┬────────┬─────────────────────┐
  │ Priority │            Action             │ Effort │  Expected Speedup   │
  ├──────────┼───────────────────────────────┼────────┼─────────────────────┤
  │ 1        │ Switch LP master + Phase 2    │ Hours  │ 2–4x on those       │
  │          │ MIP to HiGHS with threads=N   │        │ phases              │
  ├──────────┼───────────────────────────────┼────────┼─────────────────────┤
  │ 2        │ Fix incremental master        │ Days   │ Significant for     │
  │          │ rebuild (the CACHE NOTE)      │        │ large column pools  │
  ├──────────┼───────────────────────────────┼────────┼─────────────────────┤
  │ 3        │ mpi4py inter-node pricing     │ Days   │ 1.5–3x on pricing   │
  │          │ distribution                  │        │ phase               │
  ├──────────┼───────────────────────────────┼────────┼─────────────────────┤
  │ 4        │ Custom DP pricing subproblem  │ Weeks  │ 10–50x on pricing   │
  │          │ in Python/Cython              │        │ calls               │
  ├──────────┼───────────────────────────────┼────────┼─────────────────────┤
  │ 5        │ C++ custom DP pricing         │ Months │ 50–200x on pricing  │
  │          │                               │        │ calls               │
  ├──────────┼───────────────────────────────┼────────┼─────────────────────┤
  │          │                               │        │ 0% — wrong          │
  │ AVOID    │ BLACS, cuBLAS                 │ N/A    │ abstraction         │
  │          │                               │        │ entirely            │
  └──────────┴───────────────────────────────┴────────┴─────────────────────┘

  Bottom line: The 4-node cluster gives you 2–3x total speedup with mpi4py +
  HiGHS. C++ is only worth it if you replace CBC in the pricing subproblems with
   a custom DP solver — that's the only place where the Python/LP abstraction is
   genuinely paying a large tax. BLACS and cuBLAS address dense linear algebra
  which is not the computation this code performs.

  Tell me what you want to tackle first.

✻ Cooked for 3m 18s

❯ ok we will leave cpp implmentation for now let's start with Switch LP master  
+ Phase 2    │ Hours  │ 2–4x on those       │                                   
  │          │ MIP to HiGHS with threads=N                                      

● Let me find every CBC call in the file first.

  Searched for 1 pattern (ctrl+o to expand)

● Good. 7 CBC calls total. Let me read the worker functions and the CLI section
  to see the full context before touching anything.

  Read 1 file (ctrl+o to expand)

● I have everything I need. Here's the plan before touching the file:

  - Add _make_solver(threads, time_limit) — module-level helper, detects HiGHS
  once at import, falls back to CBC silently
  - _solve_lp_master and _solve_integer_master: use threads=self._solver_threads
   (the two big solves that benefit from parallelism)
  - All workers + sequential paths + gap-fill: threads=1 (they run as parallel
  processes already; giving each process N threads would over-subscribe the CPU)
  - Add solver_threads to __init__ and --solver-threads CLI arg

  Read 1 file (ctrl+o to expand)

● Now making all changes:
