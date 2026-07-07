import glob, os
from tensorboard.backend.event_processing import event_accumulator

TAGS = ["test/reward", "test/avg_violations", "test/avg_energy",
        "test/avg_comfort_mean", "train/avg_violations", "train/avg_energy"]

def load(ev):
    ea = event_accumulator.EventAccumulator(ev, size_guidance={'scalars': 0})
    ea.Reload()
    out = {}
    for t in TAGS:
        if t in ea.Tags().get('scalars', []):
            vals = [s.value for s in ea.Scalars(t)]
            out[t] = vals
    return out

def summ(name, d):
    print(f"\n===== {name} =====")
    for t in TAGS:
        if t in d and d[t]:
            v = d[t]
            print(f"  {t:22s} n={len(v):3d} first={v[0]:.3f} last={v[-1]:.3f} "
                  f"min={min(v):.3f} max={max(v):.3f}")

base = "log_building"
# MLP dir (the 220532 one, not the killed 214530)
mlp_dir = os.path.join(base, "diffusion_mlp_bcfix_clean_SchoolPrimary_Hot_Dry_20260706_220532")
mlp_ev = glob.glob(os.path.join(mlp_dir, "events.out.tfevents.*"))
for ev in mlp_ev:
    summ(f"MLP  [{os.path.basename(ev)}]", load(ev))

# Collided FNO dir: two event files, disambiguate by test/reward last value
fno_dir = os.path.join(base, "diffusion_fno_guided_bcfix_clean_SchoolPrimary_Hot_Dry_20260706_220532")
fno_evs = sorted(glob.glob(os.path.join(fno_dir, "events.out.tfevents.*")))
for ev in fno_evs:
    d = load(ev)
    last_r = d.get("test/reward", [None])[-1]
    # w48 final -6.70, w128 final -6.24
    tag = "w48?" if (last_r is not None and last_r < -6.45) else "w128?"
    summ(f"FNO {tag}  last_reward={last_r}  [{os.path.basename(ev)}]", d)
