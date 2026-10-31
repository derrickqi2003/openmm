"""Correctness checks for the perf/engine changes on CUDA, using the 33,438-atom peptoid box.

usage: python engine_check.py --repo /workspace/for-derrick-q [--steps 2000]

1. Neighbor list: at configurations along an NPT trajectory (neighbor list rebuilds, prunes, atom
   reordering and barostat box changes all happen), a context with the pruned list must find exactly the same
   number of atom pairs within the cutoff as one with the ordinary list, and the same forces.  The pair count
   comes from a CustomNonbondedForce whose energy is 1 for every pair within the cutoff, so missing pairs show up
   even near the cutoff, where their forces are too small to notice.  Copying coordinates between contexts
   changes their rounding, so a few pairs within ~1e-6 nm of the cutoff can flip (about 20 pairs lie that close);
   a list that misses pairs in a shell of width d near the cutoff loses about 2e7*d of them.  The reference context
   uses the original neighbor list settings and kernels, so this also checks the new defaults and the PME sort interval.
   By default the outer list follows hydrogens through their heavy atoms (anchored); it is also checked without that.
2. The same with an outer padding too large for the first guess at the outer list's size, so the list
   overflows and has to be enlarged, and with an inner padding large enough that pruning is skipped on some steps.
   The anchored list is also checked from positions whose constraints are not satisfied, where atoms outside their
   margins must not be anchored.
3. Reciprocal space every 2nd step: runs, stays finite, and leaves forces outside a step unchanged.
4. The fused LangevinMiddleIntegrator step (one kernel for the whole step and its SETTLE and SHAKE constraints) follows
   the same trajectory as the separate kernels, to within the run-to-run differences of the separate kernels.

The reference contexts compute forces with the original interaction loop, so every comparison also checks the loop
that gives each lane two atoms of a block (OPENMM_CUDA_TWO_ATOMS_PER_LANE).
"""
import argparse, os, sys
import numpy as np
import openmm as mm
from openmm import app, unit

p = argparse.ArgumentParser()
p.add_argument("--repo", default="/workspace/for-derrick-q")
p.add_argument("--steps", type=int, default=2000)
p.add_argument("--every", type=int, default=100)
p.add_argument("--platform", default="CUDA", help="another platform only exercises this script; pruning is CUDA only")
p.add_argument("--precision", default="single", help="CUDA precision; npt/run_4fs.py uses the default, single")
p.add_argument("--state", default=None, help="equilibrated state XML to start from (default: npt/state_post_npt_equil.xml in --repo, if present)")
args = p.parse_args()
build = os.path.join(args.repo, "build")
gro = app.GromacsGroFile(os.path.join(build, "final.gro"))
top = app.GromacsTopFile(os.path.join(build, "topol.top"), periodicBoxVectors=gro.getPeriodicBoxVectors())
cuda = mm.Platform.getPlatformByName(args.platform)
props = {"Precision": args.precision} if args.platform in ("CUDA", "OpenCL", "HIP") else {}

def make_system():
    system = top.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometer, constraints=app.HBonds, hydrogenMass=3*unit.amu)
    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    counter = mm.CustomNonbondedForce("1")
    counter.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    counter.setCutoffDistance(1.0*unit.nanometer)
    for i in range(system.getNumParticles()):
        counter.addParticle([])
    for i in range(nb.getNumExceptions()):
        a, b = nb.getExceptionParameters(i)[:2]
        counter.addExclusion(a, b)
    counter.setForceGroup(2)
    system.addForce(counter)
    system.addForce(mm.MonteCarloBarostat(1*unit.bar, 300*unit.kelvin, 25))
    return system

def context(system, integrator, env):
    for k in ("OPENMM_REORDER_INTERVAL", "OPENMM_CUDA_GROW_NEIGHBOR_LIST", "OPENMM_CUDA_PADDING", "OPENMM_CUDA_SPLIT_BUILD", "OPENMM_CUDA_WARP_BLOCK_BOUNDS", "OPENMM_CUDA_BUILD_PREFILTER", "OPENMM_PME_SORT_INTERVAL", "OPENMM_CUDA_PRUNED_NEIGHBOR_LIST", "OPENMM_CUDA_ANCHORED_NEIGHBOR_LIST", "OPENMM_CUDA_OUTER_PADDING", "OPENMM_CUDA_INNER_PADDING",
              "OPENMM_FUSED_INTEGRATION", "OPENMM_CUDA_TWO_ATOMS_PER_LANE"):
        os.environ.pop(k, None)
    os.environ.update(env)
    ctx = mm.Context(system, integrator, cuda, props)
    for k in env:
        os.environ.pop(k)
    return ctx

start = args.state or os.path.join(args.repo, "npt", "state_post_npt_equil.xml")
state0 = mm.XmlSerializer.deserialize(open(start).read()) if os.path.exists(start) else None
def start_context(ctx):
    if state0 is not None:
        ctx.setPeriodicBoxVectors(*state0.getPeriodicBoxVectors())
        ctx.setPositions(state0.getPositions())
    else:
        ctx.setPeriodicBoxVectors(*gro.getPeriodicBoxVectors())
        ctx.setPositions(gro.getPositions())
        mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 200)
    ctx.setVelocitiesToTemperature(300*unit.kelvin, 1)

# The settings of the code before perf/engine.
REFERENCE = {"OPENMM_CUDA_WARP_BLOCK_BOUNDS": "0", "OPENMM_CUDA_BUILD_PREFILTER": "0", "OPENMM_CUDA_SPLIT_BUILD": "0", "OPENMM_CUDA_PADDING": "0.08",
             "OPENMM_REORDER_INTERVAL": "250", "OPENMM_CUDA_GROW_NEIGHBOR_LIST": "0", "OPENMM_PME_SORT_INTERVAL": "2", "OPENMM_FUSED_INTEGRATION": "0",
             "OPENMM_CUDA_TWO_ATOMS_PER_LANE": "0"}

failures = []
def compare(label, test, ref):
    s = test.getState(getPositions=True)
    ref.setPeriodicBoxVectors(*s.getPeriodicBoxVectors())
    ref.setPositions(s.getPositions())
    pairs = [c.getState(getEnergy=True, groups={2}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) for c in (test, ref)]
    f = [c.getState(getForces=True, groups={0}).getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer) for c in (test, ref)]
    e = [c.getState(getEnergy=True, groups={0}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) for c in (test, ref)]
    ferr = np.sqrt(((f[0]-f[1])**2).sum(1)).max()/np.sqrt((f[1]**2).sum(1).mean())
    ok = abs(pairs[0]-pairs[1]) <= 5 and ferr < 2e-3 and abs(e[0]-e[1]) < 1e-5*abs(e[1])+1
    print(f"  {label}: pairs within cutoff {pairs[0]:.0f} vs {pairs[1]:.0f}, max force error {ferr:.1e} (relative to rms), "
          f"energy {e[0]:.1f} vs {e[1]:.1f}  {'ok' if ok else 'MISMATCH'}", flush=True)
    if not ok:
        failures.append(label)

def check_pruning(name, env, unconstrained=False):
    print(f"pruned neighbor list, {name}:", flush=True)
    system = make_system()
    integ = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 4*unit.femtoseconds)
    test = context(system, integ, env)
    ref = context(system, mm.VerletIntegrator(0.001), REFERENCE)
    start_context(test)
    if unconstrained:
        # Move every constrained atom by up to 0.05 nm, so many are outside their anchors' margins.
        pos = test.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        moved = sorted({system.getConstraintParameters(i)[0] for i in range(system.getNumConstraints())})
        rng = np.random.default_rng(1)
        pos[moved] += rng.uniform(-0.03, 0.03, (len(moved), 3))
        test.setPositions(pos)
        compare("constraints not satisfied", test, ref)
        # Stepping straight from these positions blows up with any neighbor list, so relax them first.
        test.applyConstraints(1e-5)
        mm.LocalEnergyMinimizer.minimize(test, 10.0, 200)
        test.setVelocitiesToTemperature(300*unit.kelvin, 1)
    compare("step 0", test, ref)
    for step in range(args.every, args.steps+1, args.every):
        integ.step(args.every)
        compare(f"step {step}", test, ref)
    del test, ref

check_pruning("ordinary list, perf/engine defaults", {})
check_pruning("ordinary list, split build (256 blocks per warp)", {"OPENMM_CUDA_SPLIT_BUILD": "256"})
check_pruning("ordinary list, padding 0.12 nm, split build, reorder every 1000 steps", {"OPENMM_CUDA_PADDING": "0.12", "OPENMM_CUDA_SPLIT_BUILD": "256", "OPENMM_REORDER_INTERVAL": "1000"})
check_pruning("default paddings", {"OPENMM_CUDA_PRUNED_NEIGHBOR_LIST": "1"})
check_pruning("split build of the baseline list", {"OPENMM_CUDA_PRUNED_NEIGHBOR_LIST": "1", "OPENMM_CUDA_ANCHORED_NEIGHBOR_LIST": "0",
              "OPENMM_CUDA_OUTER_PADDING": "0.08", "OPENMM_CUDA_INNER_PADDING": "0.08"})
check_pruning("not anchored", {"OPENMM_CUDA_PRUNED_NEIGHBOR_LIST": "1", "OPENMM_CUDA_ANCHORED_NEIGHBOR_LIST": "0"})
check_pruning("anchored, from unconstrained positions", {"OPENMM_CUDA_PRUNED_NEIGHBOR_LIST": "1"}, unconstrained=True)
check_pruning("outer list overflows at first", {"OPENMM_CUDA_PRUNED_NEIGHBOR_LIST": "1", "OPENMM_CUDA_OUTER_PADDING": "0.5"})
check_pruning("pruning not every step", {"OPENMM_CUDA_PRUNED_NEIGHBOR_LIST": "1", "OPENMM_CUDA_INNER_PADDING": "0.06"})

print("reciprocal space every 2nd step:", flush=True)
system = make_system()
integ = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 4*unit.femtoseconds)
integ.setReciprocalSpaceInterval(2)
test = context(system, integ, {"OPENMM_CUDA_PRUNED_NEIGHBOR_LIST": "1"})
ref = context(system, mm.VerletIntegrator(0.001), REFERENCE)
start_context(test)
integ.step(args.steps)
s = test.getState(getEnergy=True, groups={0})
T = 2*s.getKineticEnergy()/(mm.unit.MOLAR_GAS_CONSTANT_R*(3*system.getNumParticles()-system.getNumConstraints()-3))
finite = np.isfinite(s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
print(f"  {args.steps} steps with pruning too: potential energy {s.getPotentialEnergy()}, temperature {T.value_in_unit(unit.kelvin):.1f} K  {'ok' if finite else 'NOT FINITE'}")
if not finite:
    failures.append("reciprocal space interval")
compare("forces outside a step", test, ref)
del test, ref

print("fused LangevinMiddleIntegrator step:", flush=True)
system = make_system()
def trajectory(fused, steps):
    integ = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 4*unit.femtoseconds)
    integ.setRandomNumberSeed(1234)
    ctx = context(system, integ, {"OPENMM_FUSED_INTEGRATION": "1" if fused else "0"})
    start_context(ctx)
    integ.step(steps)
    x = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    err = max(abs(np.linalg.norm(x[a]-x[b])-d.value_in_unit(unit.nanometer))/d.value_in_unit(unit.nanometer)
              for a, b, d in (system.getConstraintParameters(i) for i in range(0, system.getNumConstraints(), 7)))
    return x, err
for steps in (1, 10):
    sep1, err1 = trajectory(False, steps)
    sep2, err2 = trajectory(False, steps)
    fused, errFused = trajectory(True, steps)
    noise = np.abs(sep2-sep1).max(); diff = np.abs(fused-sep1).max()
    ok = diff <= max(1e-5, 10*noise) and errFused <= 2*max(err1, err2)+1e-6
    print(f"  after {steps} steps: max |dx| separate vs separate {noise:.1e} nm, fused vs separate {diff:.1e} nm; "
          f"constraint error {errFused:.1e} (separate {max(err1, err2):.1e})  {'ok' if ok else 'MISMATCH'}", flush=True)
    if not ok:
        failures.append(f"fused integration after {steps} steps")
print("ALL CHECKS PASSED" if not failures else "FAILED: " + ", ".join(failures))
sys.exit(1 if failures else 0)
