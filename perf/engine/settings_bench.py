"""Speed of the production system under candidate settings, each variant in a fresh process.

usage: python settings_bench.py --repo /workspace/for-derrick-q [--state STATE.xml] [--steps-ps 16] [--repeats 3] VARIANT ...
A variant is "dt=4,h=3,recip=1,com=1,wh=0": time step (fs), hydrogen mass (amu), steps between PME reciprocal space
evaluations, steps between centre-of-mass motion removals (0 = no CMMotionRemover), and water hydrogen mass (0 = unchanged).  Everything else is
npt/run_4fs.py: PME 1.0 nm, HBonds, LangevinMiddleIntegrator 1/ps, Monte Carlo barostat every 100 steps, single precision.
Each repeat simulates the same amount of time (--steps-ps), so ns/day compare directly.
"""
import argparse, json, os, subprocess, sys, time
p = argparse.ArgumentParser()
p.add_argument("variants", nargs="+")
p.add_argument("--steps-ps", type=float, default=16.0)
p.add_argument("--repeats", type=int, default=3)
p.add_argument("--repo", default="/workspace/for-derrick-q")
p.add_argument("--state", default=None, help="equilibrated state XML (default: npt/state_post_npt_equil.xml in --repo)")
p.add_argument("--child", action="store_true")
args = p.parse_args()

def parse(v):
    d = dict(kv.split("=") for kv in v.split(","))
    return float(d.get("dt", 4)), float(d.get("h", 3)), int(d.get("recip", 1)), int(d.get("com", 1)), float(d.get("wh", 0))

if not args.child:
    print(f"{'variant':32s} {'median ns/day':>13s}   repeats")
    for v in args.variants:
        out = subprocess.run([sys.executable, __file__, "--child", "--repo", args.repo] + (["--state", args.state] if args.state else []) +
                             ["--steps-ps", str(args.steps_ps), "--repeats", str(args.repeats), v],
                             capture_output=True, text=True)
        try:
            r = json.loads(out.stdout.strip().split("\n")[-1])
            res = sorted(r["nsday"])
            print(f"{v:32s} {res[len(res)//2]:13.0f}   {', '.join(f'{x:.0f}' for x in r['nsday'])}", flush=True)
        except Exception:
            print(f"{v:32s} FAILED: {out.stderr.strip().splitlines()[-1] if out.stderr.strip() else out.stdout[-300:]}", flush=True)
    sys.exit()

import openmm as mm
from openmm import app, unit
dt, hmass, recip, com, waterH = parse(args.variants[0])
build = os.path.join(args.repo, "build")
gro = app.GromacsGroFile(os.path.join(build, "final.gro"))
top = app.GromacsTopFile(os.path.join(build, "topol.top"), periodicBoxVectors=gro.getPeriodicBoxVectors())
system = top.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometer, constraints=app.HBonds, hydrogenMass=hmass*unit.amu)
for i in reversed(range(system.getNumForces())):
    f = system.getForce(i)
    if isinstance(f, mm.CMMotionRemover):
        if com == 0:
            system.removeForce(i)
        else:
            f.setFrequency(com)
if waterH > 0:
    # Rigid water: moving mass from O to H changes only how fast the molecule rotates, not its equilibrium.
    for r in top.topology.residues():
        if r.name in ("SOL", "HOH", "WAT"):
            idx = [a.index for a in r.atoms()]
            total = sum(system.getParticleMass(i).value_in_unit(unit.amu) for i in idx)
            for i in idx[1:]:
                system.setParticleMass(i, waterH)
            system.setParticleMass(idx[0], total-2*waterH)
system.addForce(mm.MonteCarloBarostat(1*unit.bar, 300*unit.kelvin, 100))
integ = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, dt*unit.femtoseconds)
if recip > 1:
    integ.setReciprocalSpaceInterval(recip)
ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"), {"Precision": "single"})
s = mm.XmlSerializer.deserialize(open(args.state or os.path.join(args.repo, "npt", "state_post_npt_equil.xml")).read())
ctx.setPeriodicBoxVectors(*s.getPeriodicBoxVectors()); ctx.setPositions(s.getPositions())
ctx.setVelocitiesToTemperature(300*unit.kelvin, 1)
nsteps = int(round(args.steps_ps*1000/dt))
integ.step(nsteps//4); ctx.getState(getEnergy=True)
res = []
for r in range(args.repeats):
    t = time.perf_counter(); integ.step(nsteps); ctx.getState(getEnergy=True)
    res.append(args.steps_ps*1e-3/((time.perf_counter()-t)/86400))
print(json.dumps({"nsday": res}))
