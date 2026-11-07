# perf/engine: faster single-trajectory steps for the peptoid box

Engine changes on top of `perf/npt-workload`, for one simulation on one GPU (33,438 atoms, CHARMM36, PME 1.0 nm,
HBonds, hydrogen mass 3 amu, 4 fs, LangevinMiddleIntegrator, Monte Carlo barostat every 100 steps, single precision,
as in `npt/run_4fs.py`). None of them change the physics: the neighbor list holds the same pairs, the force kernel
computes the same pairs, and the integrator does the same arithmetic. What changes is how the work is organized on the GPU.

## Result (A40)

| | ns/day (median of 5 × 4,000 steps, each in a fresh process) |
|---|---|
| Before perf/engine | 1,097–1,107 |
| **perf/engine defaults** | **1,416–1,431 (+29%)** |

For a 500 ns run that is 10.9 h before and 8.4 h with the defaults.

At 2 fs with `npt/run.py`'s settings (no hydrogen mass repartitioning, NoseHooverIntegrator, barostat every 25 steps)
the defaults give 612–613 ns/day against 516–517 (+18.5%). The fused integration step only applies to
LangevinMiddleIntegrator, so it does not help there.

Turning off one default at a time (all measured in one session, on the same pod and starting state):

| without | ns/day | | what it does |
|---|---|---|---|
| two atoms per lane | 1,292–1,294 | −9% | in the force kernel, each lane takes two atoms of a block, so half as many warp shuffles |
| fused integration | 1,328 | −6% | one kernel for the whole LangevinMiddle step and its SETTLE and SHAKE constraints |

The other defaults were measured one at a time in an earlier session, before the changes above (when the defaults gave
1,238–1,245):

| without | | what it does |
|---|---|---|
| build prefilter | −2 to −3.5% | test atoms against a block's bounding box before comparing them to all 32 of its atoms |
| split build | −2 to −3% | spread each block's neighbor search over several warps |
| padding 0.12 nm | −2 to −3% | rebuild the list every ~3 steps instead of every ~2 |
| PME sort every 20 | −2.5% | re-sort the atoms for PME every 20 evaluations instead of every 2 |
| reorder every 1000 steps | −1 to −2% | sort the atoms less often (each sort stalls the GPU ~2 ms) |
| warp block bounds | −1 to −2% | one warp per block for bounding boxes instead of one thread |
| list growth | steady state equal | enlarge the list before it overflows; without it the first repeat was 1,051 |

All 14 CUDA test programs pass with the defaults and with the pruned list, and `engine_check.py` finds exactly the
same pairs within the cutoff as the original kernels along an NPT trajectory (counted by a CustomNonbondedForce), forces
equal to 1e-6 of their rms, and the fused integration step on the same trajectory as the separate kernels to within their
run-to-run differences.

## What the time goes to, and why the changes work

Measured with Nsight Systems, a diagnostic per-warp timer in the list build (`OPENMM_CUDA_BUILD_TIMING=1`), runs with
PME moved onto the main stream (`engine_bench.py --serial-pme`), and timing-only variants of kernels. Nsight Compute's
counters are not available on the rented pods, so where a kernel spends its time was measured by removing one part at a
time, with forces evaluated repeatedly at one fixed configuration (stepping with a modified kernel wrecks the trajectory
and with it the timings).

The step now takes ~245 µs under the profiler: ~195 µs on the two thirds of steps that don't rebuild the neighbor list,
~305 µs on the third that do.

- **The force kernel is limited by warp shuffles, then by arithmetic.** Each warp computes a tile of 32 atoms of a block
  against 32 other atoms, rotating the other atoms' positions, parameters and forces through the lanes with 9 shuffles per
  step of the loop. At a fixed configuration (123 µs): removing the pair arithmetic saved 28 µs, removing 3 of the 9
  shuffles saved 12 µs, removing the force atomics saved nothing. The A40 executes one warp shuffle per cycle per SM, so
  the shuffles, not the arithmetic, set the pace. With **two atoms per lane** each half of the warp rotates 16 atoms and
  every lane computes two pairs per shuffle: 123 → 106 µs. The loop is now issue-bound on the arithmetic of the pairs
  (one more iteration costs ~4.5 µs), and the kernel uses 64 registers, the most that keeps 4 work groups per SM.
- **Two thirds of the pairs in a tile are farther apart than the cutoff** (34% of the 19.7M pair slots are within it),
  and the kernel computes them anyway, with the result masked. Computing only the pairs within the cutoff was tried and is
  2× slower (see below).
- **Two streams overlap.** PME runs on its own stream alongside the main one. On steps that don't rebuild the list, PME
  now ends ~5 µs after the force kernel, so it sets their length; on steps that do, it ends long before.
- **The list build is throughput-bound.** With the split build, items last 41 µs on average and 90–100 µs at most, and the
  build ~113 µs. Smaller segments shorten the longest items (63 µs with 128 blocks) but add work, and the build gets
  slower (125 µs): every item pays a fixed ~2 µs plus ~1 µs per candidate block.
  - The **prefilter** skips two thirds of the distance checks: only 34% of atoms in candidate blocks are within the padded
    cutoff of the block's bounding box. The list is unchanged, since the test only rejects atoms farther than the cutoff
    from every atom of the block.
  - The **split build** gives each warp at most 256 of the blocks to search. Each segment stores its full tiles, and the
    warp that finishes a block's last segment packs the atoms all its segments left over (fewer than 32 each) into tiles;
    otherwise the extra partly filled tiles made the force kernel 12% slower. It is off for systems of 256 blocks or fewer.
- **The fused integration step.** LangevinMiddle took 7 kernels per step (velocity step, SETTLE and SHAKE on velocities,
  heat bath and position step, SETTLE and SHAKE on positions, final step): 22.5 µs plus 6 gaps. Constraints only couple
  atoms of one water or one heavy atom and its hydrogens, so one thread can take each cluster, or each unconstrained atom,
  through the whole step: 10 µs. That kernel is limited by memory: it moves ~3.5 MB (positions, velocities, forces and
  random numbers in, positions and velocities out), a third of what the separate kernels moved. It is used when every
  constraint is in a SETTLE or SHAKE cluster (no CCMA) and no cluster has a massless particle. The SETTLE and SHAKE
  arithmetic is shared with the separate kernels (`constraintClusters.cc`), and on OpenCL the trajectory is bitwise
  identical to theirs.
- **Fewer kernel launches.** Small kernels take 2–4 µs each however little they do, so `findBlockBoundsByWarp` also checks
  whether any atom has moved far enough to rebuild the list (which saved launching `checkRebuildNeighborList`), the
  leftover atoms are packed inside the build, and the block sort runs one warp per key over the whole GPU instead of
  9 work groups (21 → 9 µs on rebuild steps). The merged kernels got ~5 µs slower in total, so this saved less than the
  launches cost.
- **The rebuild interval is a staircase in the padding.** With hydrogen mass repartitioning at 4 fs the fastest hydrogens
  move ~0.027 nm per step, and the list is rebuilt once some atom has moved half the padding. 0.08 and 0.10 nm give a
  rebuild every ~2 steps; 0.12 nm one every 2.86. With the current kernels 0.10 nm is 3% slower than 0.12 and 0.14–0.18 nm
  are ~1% slower. The default is 0.12 × the cutoff (upstream: 0.08). At 2 fs the padding hardly matters.
- **Once the main stream is shorter, PME counts.** Re-sorting the atoms for PME only every 20 evaluations removes ~16 µs
  per step from the PME stream without slowing the charge spreading.
- **Idle GPU time** is ~11 µs per step: ~6 µs of ~1 µs gaps between kernels, and ~3 µs from re-sorting the atoms (the
  GPU waits ~2 ms each time while the CPU sorts), which CUDA now does every 1000 steps instead of 250; the neighbor list
  still forces a re-sort when it has grown more than 10% since the last one.
- **List overflows are expensive with the lazy check.** The host only learns of an overflow when a later step's counts
  arrive, and every step queued until then computes all tiles (~3.6 ms each): one overflow cost ~200 ms. The list is now
  enlarged by 1.5× (after one synchronization) as soon as it is 90% full.

## Opt-in experiments that do not help on the A40

These are implemented, correct (tested), and off by default.

- **Reciprocal space every n steps** (`LangevinMiddleIntegrator.setReciprocalSpaceInterval(2)`, all GPU platforms
  and Reference): impulse r-RESPA with the excluded-pair correction moved into the slow part. Matches a CustomIntegrator
  implementation to 3e-12 nm (Reference). +5% on top of the earlier defaults (1,299–1,302 against 1,238–1,245 ns/day).
  It changes the thermodynamics slightly, so it stays opt-in. Two 20 ns runs at `npt/run_4fs.py`'s settings from the
  same equilibrated state (A40, errors from 10 blocks after 1 ns):

  | | every step | reciprocal every 2nd step | difference |
  |---|---|---|---|
  | temperature (K) | 300.939 ± 0.016 | 301.111 ± 0.018 | +0.17 (7σ) |
  | potential energy (kJ/mol) | −546,777 ± 9 | −546,679 ± 16 | +97 (5σ) |
  | density (kg/m³) | 1041.24 ± 0.05 | 1041.69 ± 0.07 | +0.46 (5σ) |
  | mean chain Rg (nm) | 0.922 ± 0.002 | 0.920 ± 0.003 | none detected |
  | inter-chain heavy-atom contacts < 0.5 nm | 176 ± 21 | 199 ± 18 | none detected (not converged) |
  | chain pairs in contact (of 66) | 5.2 ± 0.5 | 5.0 ± 0.3 | none detected (not converged) |
  | water O–O g(r) first peak | 3.210 at 0.2725 nm | 3.212 at 0.2725 nm | none |
  | water O diffusion (10⁻⁵ cm²/s) | 1.80 | 1.80 | none |

  The impulse heats the system by 0.17 K against the thermostat, and that accounts for the potential energy (about
  9,929 waters × ~57 J/mol/K of potential heat capacity × 0.17 K ≈ 95 kJ/mol). It does not account for the density,
  since heating would lower it by ~0.05 kg/m³: the rise is either a real effect of sampling with forces the barostat's
  energies do not match, or a slow fluctuation (for example in how the chains aggregate) that 1.9 ns blocks miss.
  Both runs read ~0.9 K above 300 K, so that part comes from the 4 fs step, not from this option.
- **Pruned neighbor list** (`OPENMM_CUDA_PRUNED_NEIGHBOR_LIST=1`, optional `OPENMM_CUDA_OUTER_PADDING`,
  `OPENMM_CUDA_INNER_PADDING`, `OPENMM_CUDA_ANCHORED_NEIGHBOR_LIST=0`): an outer list with a large padding, rebuilt
  rarely, pruned to the cutoff every step. By default the outer list follows each constrained hydrogen through its heavy
  atom with a margin of 1.001 × the bond length, which made it last 13.3 steps instead of 6.2 on the saved trajectory
  with no pairs ever missed. It is 9–12% slower: pruning 1.04M outer entries × 32 atoms every step costs about as much
  per check as the build does, and packing them is one warp per block again. Its kernels are also split over warps.

## Tried and removed

- **Computing only the pairs within the cutoff** in the force kernel: queue the pairs of a tile within the cutoff and
  compute them 32 at a time, summing forces in shared memory. That needs 10.8 evaluations per tile instead of 32, but the
  kernel got 2× slower (278 µs against 135). Timing-only variants show why: queueing alone, with no pair computed at all,
  took 144 µs, more than the whole original kernel, because the arithmetic it saves is only ~28 µs of it; and on sm_86
  atomic additions of floats (and of 64-bit integers) to shared memory compile to compare-and-swap loops
  (`ATOMS.CAST.SPIN`), which cost another 95 µs (only 32-bit integer additions are native).
- **Skipping the reciprocal-space correction for constrained excluded pairs** (prototype only): the force between two
  atoms whose distance is constrained is removed by the constraints, and its energy is constant, so 31,800 of the 47,706
  corrections can be skipped during integration. The trajectory was unchanged to within run-to-run differences, but the
  bonded kernel only went from 14.2 to 11.6 µs (+1%), not enough for the bookkeeping that keeps reported energies and
  forces exact.
- A **half-ring "balanced" list scan** (earlier session, −18%): it gave half of every block's pairs to that block,
  including the few very large blocks that need the slow per-pair periodic path, so one warp did most of the work and
  the build took 2.4× longer. The split build keeps OpenMM's ownership and splits only the work.
- **Prefetching** the next candidate block's atoms in the build: the kernel is limited to 80 registers by its launch
  bounds, and the prefetch made it spill (16 bytes per thread), so it was slower; the loads it removed hit in L1. Tried
  again on an RTX 3090 without spilling (80 registers): rebuilds took 150 µs instead of 128. What is kept is taking each
  candidate's block index by a warp shuffle from the lane that tested it, instead of reloading it, which takes one load
  out of the chain that leads to loading its atoms: rebuilds 128 → 124 µs (−3%), about +0.5% overall.
- **Testing a per-tile bit mask instead of each atom's index** for padding atoms in the force loop: the loop went from 116
  to 114 instructions per two pairs and the kernel time did not change.
- **Launching PME charge spreading with 4× the threads** (it runs at 23% occupancy): the kernel only went from 46 to
  42 µs, since it is limited by atomic additions in L2 (1.18M per evaluation), and the step got 2–3.5% slower because
  it then competes harder with the force kernel. Even with no atomics at all (timing only) the step would gain 8 µs.
- **The single-pair threshold** (`MAX_BITS_FOR_PAIRS`, upstream's 3 on this GPU): 0 → −8%, 1 → −5%, 2 → −2%, 4 and 5
  → within ±0.5%, 6 → −1%, 8 → −6%.
- **One kernel for the block sort** (one work group): the sort ran on a single SM and took longer than the three
  kernels it replaced.

## Hardware counters, and how far engine work can go

Nsight Compute's counters are blocked in containers (`ERR_NVGPUCTRPERM`), so these were measured on an RTX 3090 rented
as a VM (KVM, where root can read them), with perf/engine at 1,619 ns/day (1,222 before it) on that GPU.

- **Force kernel:** 75% of issue slots busy, 62% of instructions are FP32 multiply/add, 3 special-function
  instructions per pair; ~58 instructions per pair in the loop including shuffles. Of its warp samples, 59% are in the
  pair loop, 14% in the single-pair loop (5 loads and 6 atomics per pair), 13% in tiles with exclusions.
- **PME spreading, interpolation, bonded forces, fused integration:** 55–61% of stalls wait on memory, at 8–23%
  occupancy (every CUDA kernel's grid is capped at 6 blocks of 64 threads per SM, and the integration has one thread per
  constraint cluster).
- **What each part is worth in the concurrent step** (Verlet steps of 1e-9 ps, so the list is never rebuilt; force
  groups left out of the integration):

  | computed | µs per step |
  |---|---|
  | everything | 176 |
  | without PME reciprocal space | 149 |
  | without bonded forces | 173 |
  | without centre-of-mass motion removal | 169 |
  | direct space only | 137.5 |

  Kernel times measured one at a time overstate what speeding up PME and the small kernels would gain: most of their
  time is hidden behind the force kernel. Direct space plus the list build is ~80% of a production step.
- **GROMACS 2025.4 on the same system and GPU** (same topology and coordinates, hydrogen mass 3 amu, 4 fs, PME on
  OpenMM's 60³ grid but order 4, the only order its GPU PME supports, and its `md` integrator, which GPU update needs):
  1,198 ns/day with its default list interval, 1,307 at its best (`-nstlist 100`), against 1,619 for perf/engine. Its
  cluster-pair design is the one an OpenMM rewrite would copy, so a same-settings 2× is not available from engine work.

## Larger time steps (settings for the run scripts, not the engine)

Measured on the RTX 3090 with `settings_bench.py` (each in its own process) and checked with `validate_settings.py`
(5 ns NPT from the same equilibrated state, sampled every ps after 0.5 ns, three seeds each, compared with
`compare_reps.py`):

| settings | ns/day | |
|---|---|---|
| `npt/run_4fs.py` (4 fs, H 3 amu) | 1,609–1,628 | |
| centre-of-mass removal every 100 steps | 1,609 | no gain |
| reciprocal space every 2nd step | 1,697–1,711 | shifts the potential energy (see above) |
| 5 fs | 1,909 | stable |
| 5.5 fs | 2,074 | stable |
| **5.5 fs, water H 3 amu** | **2,260** | **1.40×; matches 4 fs in density, energy and chain Rg** |
| 5.5 fs, water H 3 amu, reciprocal every 2nd step | 2,371 | potential energy +110 to +195 kJ/mol (3–5σ) |
| 5 or 6 fs with H 4 amu | NaN within 1000 steps | a CH₃ carbon is left with ~3 amu |
| 6 fs, H 3 amu (with or without water H 3 amu) | NaN within 7000 steps | |

"Water H 3 amu" moves mass from each SPC/E oxygen to its hydrogens (O 12.0 amu). Water is rigid, so this changes only
how fast it rotates, not its equilibrium, and it slows the fastest atoms, so the neighbor list is rebuilt less often.
Three replicates each:

| | 4 fs (now) | 5.5 fs, water H 3 amu | difference |
|---|---|---|---|
| density (kg/m³) | 1041.16 ± 0.12 | 1041.11 ± 0.04 | −0.4 × its error |
| potential energy (kJ/mol) | −546,775 ± 15 | −546,765 ± 3 | +0.7 × |
| mean chain Rg (nm) | 0.9220 ± 0.0026 | 0.9221 ± 0.0014 | 0.0 × |
| inter-chain heavy-atom contacts < 0.5 nm | 205 ± 33 (runs: 177, 168, 270) | 148 ± 19 | −1.5 × (not converged in 5 ns) |
| kinetic temperature (K) | 300.95 | 301.12 | +0.17 (how kinetic T reads at a larger step) |

## Settings (environment variables, read when a Context is created)

| variable | default | |
|---|---|---|
| `OPENMM_CUDA_TWO_ATOMS_PER_LANE` | 1 | 0 = the original force loop, one atom of the block per lane |
| `OPENMM_FUSED_INTEGRATION` | 1 | 0 = the separate LangevinMiddle and constraint kernels (all GPU platforms) |
| `OPENMM_CUDA_BUILD_PREFILTER` | 1 | 0 = compare every atom of candidate blocks |
| `OPENMM_CUDA_SPLIT_BUILD` | 256 | blocks searched per warp; 0 = one warp per block |
| `OPENMM_CUDA_PADDING` | 0.12 × cutoff | neighbor list padding in nm |
| `OPENMM_REORDER_INTERVAL` | 1000 on CUDA, 250 elsewhere | steps between periodic re-sorts of the atoms |
| `OPENMM_PME_SORT_INTERVAL` | 20 on CUDA, upstream's choice elsewhere | PME evaluations between sorts of the atoms by grid cell |
| `OPENMM_CUDA_GROW_NEIGHBOR_LIST` | 1 | 0 = only enlarge the list after it overflows |
| `OPENMM_CUDA_WARP_BLOCK_BOUNDS` | 1 | 0 = one thread per block for bounding boxes (and a separate rebuild check) |
| `OPENMM_CUDA_BUILD_TIMING` | 0 | 1 = print per-warp timing of the list build (diagnostic) |

## Testing on a GPU machine

From the root of this checkout, with a `for-derrick-q` checkout at REPO:

```
bash perf/engine/build_and_test.sh REPO       # SHORT=1 for fewer tests and a shorter benchmark
```

It builds with CUDA, runs 14 CUDA test programs with the ordinary and the pruned list, then `engine_check.py`
(pairs and forces against the original kernels, along an NPT trajectory, and the fused integration step against the
separate kernels) and `engine_bench.py` (ns/day for each variant in its own process, at production settings). For the
benchmark, start from an equilibrated state (`npt/state_post_npt_equil.xml`), and use single precision, which
`npt/run_4fs.py` uses; mixed precision is 20% slower and spends its time differently.

It builds with a CUDA 12 or 13 toolkit (upstream now uses CUDA 13's names for 16-byte aligned 64-bit vectors in the
kernels it compiles at run time; `common.cu` falls back to the CUDA 12 names).
