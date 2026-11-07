"""Equilibrium check of candidate settings against npt/run_4fs.py's: NPT from the same equilibrated state, sampled every
1 ps after 0.5 ns.
usage: python validate_settings.py --repo /workspace/for-derrick-q [--state STATE.xml] [--seed 1] VARIANT NS OUT.npz
(VARIANT as in settings_bench.py; compare replicates with compare_reps.py)
Records potential energy, kinetic temperature, density, mean chain radius of gyration (heavy atoms) and the number of
heavy-atom contacts (< 0.5 nm) between different chains."""
import argparse, os, time
import numpy as np
import openmm as mm
from openmm import app, unit
p = argparse.ArgumentParser()
p.add_argument("variant"); p.add_argument("ns", type=float); p.add_argument("out")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--repo", default="/workspace/for-derrick-q")
p.add_argument("--state", default=None, help="equilibrated state XML (default: npt/state_post_npt_equil.xml in --repo)")
args = p.parse_args()
variant, ns, out, seed = args.variant, args.ns, args.out, args.seed
d = dict(kv.split("=") for kv in variant.split(","))
dt, hmass, recip, waterH = float(d.get("dt", 4)), float(d.get("h", 3)), int(d.get("recip", 1)), float(d.get("wh", 0))
build = os.path.join(args.repo, "build")
gro = app.GromacsGroFile(os.path.join(build, "final.gro"))
top = app.GromacsTopFile(os.path.join(build, "topol.top"), periodicBoxVectors=gro.getPeriodicBoxVectors())
system = top.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometer, constraints=app.HBonds, hydrogenMass=hmass*unit.amu)
residues = list(top.topology.residues())
if waterH > 0:
    for r in residues:
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
integ.setRandomNumberSeed(seed)
ctx.setVelocitiesToTemperature(300*unit.kelvin, seed)
# The 12 chains: molecules of the first molecule type, heavy atoms only.
chains = [c for c in top.topology.chains()]
mols = [[a.index for a in c.atoms()] for c in chains]
chainAtoms = [m for m in mols if len(m) > 100]
heavy = [[i for i in m if list(top.topology.atoms())[i].element.symbol != "H"] for m in chainAtoms]
print(f"{len(heavy)} chains, {len(heavy[0])} heavy atoms each", flush=True)
mass = sum(system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(system.getNumParticles()))
dof = 3*system.getNumParticles()-system.getNumConstraints()-3
per = int(round(1000/dt))          # steps per ps
nframes = int(ns*1000)
pe, T, rho, rg, contacts = [], [], [], [], []
t0 = time.time()
for k in range(nframes):
    integ.step(per)
    if k < 500:
        continue
    st = ctx.getState(getEnergy=True, getPositions=True)
    x = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    box = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    L = np.diag(box)
    vol = np.prod(L)
    pe.append(st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    T.append((2*st.getKineticEnergy()/(dof*unit.MOLAR_GAS_CONSTANT_R)).value_in_unit(unit.kelvin))
    rho.append(mass*1.66053906660/vol)    # kg/m^3
    g = []
    for h in heavy:
        p = x[h]; g.append(np.sqrt(((p-p.mean(0))**2).sum(1).mean()))
    rg.append(np.mean(g))
    if k % 10 == 0:
        allp = np.concatenate([x[h] for h in heavy]); lab = np.concatenate([[c]*len(h) for c, h in enumerate(heavy)])
        n = 0
        for c in range(len(heavy)):
            a = allp[lab == c]; b = allp[lab > c]
            dlt = a[:, None, :]-b[None, :, :]; dlt -= L*np.round(dlt/L)
            n += int(((dlt**2).sum(2) < 0.25).sum())
        contacts.append(n)
    if not np.isfinite(pe[-1]):
        print("NOT FINITE at", k, flush=True); break
el = time.time()-t0
np.savez(out, pe=pe, T=T, rho=rho, rg=rg, contacts=contacts, ns=ns, seconds=el)
print(f"{variant}: {ns} ns in {el:.0f} s; PE {np.mean(pe):.0f}, T {np.mean(T):.2f}, density {np.mean(rho):.2f}, Rg {np.mean(rg):.4f}, contacts {np.mean(contacts):.0f}", flush=True)
