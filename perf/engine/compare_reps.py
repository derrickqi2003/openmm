"""Compare replicate runs of two settings.  Each run's mean is one sample, so the uncertainty comes from how much
independent runs differ, which is honest for slow observables like chain Rg and inter-chain contacts.
usage: python compare_reps.py "LABEL_A=a1.npz,a2.npz,..." "LABEL_B=b1.npz,..."
"""
import sys
import numpy as np
groups = []
for arg in sys.argv[1:]:
    label, files = arg.split("=", 1)
    groups.append((label, [np.load(f) for f in files.split(",")]))
keys = [("rho", "density (kg/m3)", 2), ("pe", "potential energy (kJ/mol)", 0), ("rg", "chain Rg (nm)", 4),
        ("contacts", "inter-chain contacts", 0), ("T", "kinetic T (K)", 2)]
for k, name, dig in keys:
    print(name)
    means = []
    for label, runs in groups:
        m = np.array([np.mean(r[k]) for r in runs])
        means.append(m)
        print(f"   {label:24s} runs: {', '.join(f'{x:.{dig}f}' for x in m)}   mean {m.mean():.{dig}f} +- {m.std(ddof=1)/np.sqrt(len(m)):.{dig}f}")
    a, b = means[0], means[1]
    se = np.hypot(a.std(ddof=1)/np.sqrt(len(a)), b.std(ddof=1)/np.sqrt(len(b)))
    print(f"   difference {b.mean()-a.mean():+.{dig}f} = {(b.mean()-a.mean())/se:+.1f} x its standard error")
for label, runs in groups:
    times = ", ".join("%.1f ns in %.0f s" % (float(r["ns"]), float(r["seconds"])) for r in runs)
    print(f"{label}: {times}")
