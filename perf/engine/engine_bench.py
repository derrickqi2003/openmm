"""Speed of one simulation of the 33,438-atom peptoid box with the perf/engine options, on CUDA.

usage: python engine_bench.py --repo /workspace/for-derrick-q [--state STATE.xml] [--steps 4000] [--repeats 3]

Production settings from npt/run_4fs.py: PME 1.0 nm, HBonds, hydrogen mass 3 amu, 4 fs, LangevinMiddleIntegrator
1/ps, Monte Carlo barostat every 100 steps, single precision (the CUDA default it uses), no reporters.  Each variant gets a fresh Context.
"""
import argparse, json, os, subprocess, sys, time
import numpy as np
import openmm as mm
from openmm import app, unit

p = argparse.ArgumentParser()
p.add_argument("--repo", default="/workspace/for-derrick-q")
p.add_argument("--state", default=None)
p.add_argument("--steps", type=int, default=4000)
p.add_argument("--warmup", type=int, default=500)
p.add_argument("--repeats", type=int, default=3)
p.add_argument("--platform", default="CUDA")
p.add_argument("--setup", default="run_4fs", choices=["run_4fs", "run"],
               help="run_4fs: npt/run_4fs.py (4 fs, HMR, Langevin, barostat every 100); run: npt/run.py (2 fs, Nose-Hoover, barostat every 25)")
p.add_argument("--precision", default="single", help="CUDA precision; npt/run_4fs.py uses the default, single")
p.add_argument("--serial-pme", action="store_true", help="diagnostic: compute PME on the main stream, so kernels never overlap")
p.add_argument("--only", default=None, help="variant names separated by semicolons")
p.add_argument("--extra", action="append", default=[], help='extra variant "label|KEY=VALUE,KEY=VALUE" (reciprocal interval 1)')
p.add_argument("--same-process", action="store_true", help="run all variants in this process instead of one process each")
p.add_argument("--child", default=None, help=argparse.SUPPRESS)
p.add_argument("--main", action="store_true", help="only the baseline and the defaults, alternating")
args = p.parse_args()
build = os.path.join(args.repo, "build")
gro = app.GromacsGroFile(os.path.join(build, "final.gro"))
top = app.GromacsTopFile(os.path.join(build, "topol.top"), periodicBoxVectors=gro.getPeriodicBoxVectors())
start = args.state or os.path.join(args.repo, "npt", "state_post_npt_equil.xml")
state0 = mm.XmlSerializer.deserialize(open(start).read()) if os.path.exists(start) else None
props = {"Precision": args.precision} if args.platform in ("CUDA", "OpenCL", "HIP") else {}
if args.serial_pme:
    props["DisablePmeStream"] = "true"

# The baseline turns off everything perf/engine changed about the neighbor list and atom reordering, so it matches the
# code before it.  "defaults" is perf/engine as it is, and each "without" variant turns off one of its defaults.
OFF = {"OPENMM_CUDA_WARP_BLOCK_BOUNDS": "0", "OPENMM_CUDA_BUILD_PREFILTER": "0", "OPENMM_CUDA_SPLIT_BUILD": "0",
       "OPENMM_CUDA_PADDING": "0.08", "OPENMM_REORDER_INTERVAL": "250", "OPENMM_CUDA_GROW_NEIGHBOR_LIST": "0",
       "OPENMM_PME_SORT_INTERVAL": "2", "OPENMM_FUSED_INTEGRATION": "0", "OPENMM_CUDA_TWO_ATOMS_PER_LANE": "0"}
variants = [
    ("baseline", 1, OFF),
    ("defaults", 1, {}),
    ("without split build", 1, {"OPENMM_CUDA_SPLIT_BUILD": "0"}),
    ("without padding 0.12 nm", 1, {"OPENMM_CUDA_PADDING": "0.08"}),
    ("without reorder every 1000", 1, {"OPENMM_REORDER_INTERVAL": "250"}),
    ("without build prefilter", 1, {"OPENMM_CUDA_BUILD_PREFILTER": "0"}),
    ("without list growth", 1, {"OPENMM_CUDA_GROW_NEIGHBOR_LIST": "0"}),
    ("without warp block bounds", 1, {"OPENMM_CUDA_WARP_BLOCK_BOUNDS": "0"}),
    ("without PME sort every 20", 1, {"OPENMM_PME_SORT_INTERVAL": "2"}),
    ("without fused integration", 1, {"OPENMM_FUSED_INTEGRATION": "0"}),
    ("without two atoms per lane", 1, {"OPENMM_CUDA_TWO_ATOMS_PER_LANE": "0"}),
    # Opt-in experiments, on top of the defaults.
    ("reciprocal every 2 steps", 2, {}),
    ("pruned, anchored list", 1, {"OPENMM_CUDA_PRUNED_NEIGHBOR_LIST": "1"}),
    ("pruned, plain list 0.24 nm", 1, {"OPENMM_CUDA_PRUNED_NEIGHBOR_LIST": "1", "OPENMM_CUDA_ANCHORED_NEIGHBOR_LIST": "0"}),
]
for spec in args.extra:
    label, _, settings = spec.partition("|")
    variants.append((label, 1, dict(kv.split("=", 1) for kv in settings.split(",") if kv)))
ENV_VARS = ("OPENMM_REORDER_INTERVAL", "OPENMM_CUDA_GROW_NEIGHBOR_LIST", "OPENMM_CUDA_PADDING", "OPENMM_CUDA_SPLIT_BUILD", "OPENMM_CUDA_BUILD_TIMING", "OPENMM_CUDA_WARP_BLOCK_BOUNDS", "OPENMM_CUDA_BUILD_PREFILTER", "OPENMM_PME_SORT_INTERVAL", "OPENMM_CUDA_PRUNED_NEIGHBOR_LIST",
            "OPENMM_CUDA_ANCHORED_NEIGHBOR_LIST", "OPENMM_CUDA_OUTER_PADDING", "OPENMM_CUDA_INNER_PADDING", "OPENMM_FUSED_INTEGRATION",
            "OPENMM_CUDA_TWO_ATOMS_PER_LANE")
if args.main:
    variants = [variants[0], variants[1], variants[0], variants[1]]
if args.only:
    # Run the named variants in this order; a name may appear several times, such as a baseline between the others.
    byName = {v[0]: v for v in variants}
    variants = [byName[name] for name in args.only.split(";")]

DT = 0.004 if args.setup == "run_4fs" else 0.002   # time step in ps

def run(interval, env):
    if args.setup == "run_4fs":
        system = top.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometer, constraints=app.HBonds, hydrogenMass=3*unit.amu)
        system.addForce(mm.MonteCarloBarostat(1*unit.bar, 300*unit.kelvin, 100))
        integ = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 4*unit.femtoseconds)
        integ.setReciprocalSpaceInterval(interval)
    else:
        if interval != 1:
            raise ValueError("the reciprocal space interval needs LangevinMiddleIntegrator")
        system = top.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometer, constraints=app.HBonds)
        system.addForce(mm.MonteCarloBarostat(1*unit.bar, 300*unit.kelvin, 25))
        integ = mm.NoseHooverIntegrator(300*unit.kelvin, 1/unit.picosecond, 2*unit.femtoseconds)
    for k in ENV_VARS:
        os.environ.pop(k, None)
    os.environ.update(env)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName(args.platform), props)
    if state0 is not None:
        ctx.setPeriodicBoxVectors(*state0.getPeriodicBoxVectors())
        ctx.setPositions(state0.getPositions())
    else:
        ctx.setPeriodicBoxVectors(*gro.getPeriodicBoxVectors())
        ctx.setPositions(gro.getPositions())
        mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 200)
    ctx.setVelocitiesToTemperature(300*unit.kelvin, 1)
    integ.step(args.warmup)
    ctx.getState(getEnergy=True)
    rates = []
    for r in range(args.repeats):
        t = time.perf_counter()
        integ.step(args.steps)
        ctx.getState(getEnergy=True)   # waits for the GPU to finish
        rates.append(args.steps*DT*86400/1000/(time.perf_counter()-t))
    return rates

def child_args(name):
    """This script's arguments, with --only replaced by --child name."""
    out, skip = [], False
    for a in sys.argv[1:]:
        if skip:
            skip = False
        elif a == "--only":
            skip = True
        elif not a.startswith("--only="):
            out.append(a)
    return [sys.executable, os.path.abspath(__file__)] + out + ["--child", name]

if args.child is not None:
    # Run one variant and report its rates to the parent process.
    name, interval, env = next(v for v in variants if v[0] == args.child)
    print("RATES " + json.dumps(run(interval, env)), flush=True)
    sys.exit(0)

# By default every variant runs in a fresh process: in one long process, variants after many others sometimes had
# a slow first repeat that did not occur in a fresh process.
base = None
print(f"{'variant':40s} {'median':>8s} {'best':>8s}   median vs first   all repeats (ns/day)")
for name, interval, env in variants:
    if args.same_process:
        rates = run(interval, env)
    else:
        r = subprocess.run(child_args(name), capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("RATES ")]
        if r.returncode != 0 or not line:
            print(f"{name:40s} FAILED: " + (r.stderr.strip().splitlines() or ["?"])[-1], flush=True)
            continue
        rates = json.loads(line[-1][6:])
    med = float(np.median(rates))
    base = base or med
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print(f"{name:40s} {med:8.0f} {max(rates):8.0f}   {med/base:9.3f}x   " + ", ".join(f"{r:.0f}" for r in rates) + f"   [GPU: {gpu}]", flush=True)
