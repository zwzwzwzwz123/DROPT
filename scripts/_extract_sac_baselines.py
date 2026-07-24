"""抽 SAC / SAC+MPC 公平协议 3-seed 基线 (末段8点窗均值)。
数据源: log_building/{sacmpc_fair,sac_pure}_{small,medium,school}_1m_s{42,0,1}_*。
每 run 独立 EventAccumulator 防缓存串味; 同前缀多目录时选 test/avg_energy eval 点最多的真身。
违规按 zone 归一化每区率。输出供填 INCREMENT §5 / stage5 基线行。
"""
import glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = "log_building"
W = 8
T_E, T_V, T_C = "test/avg_energy", "test/avg_violations", "test/avg_comfort_mean"
ZONES = {"small": 6, "medium": 18, "school": 25}
SEEDS = ["s42", "s0", "s1"]


def pick_run_dir(prefix):
    """同前缀多目录: 选 T_E eval 点最多的 (真身, 防假启动/半截)。"""
    cands = sorted(set(glob.glob(os.path.join(ROOT, prefix + "_*"))))
    best, best_n = None, -1
    for d in cands:
        ev = sorted(glob.glob(os.path.join(d, "events.out.tfevents.*")))
        if not ev:
            continue
        acc = EventAccumulator(ev[-1], size_guidance={"scalars": 0}); acc.Reload()
        if T_E not in acc.Tags().get("scalars", []):
            continue
        n = len(acc.Scalars(T_E))
        if n > best_n:
            best, best_n = d, n
    return best, best_n


def tail_mean(run_dir, tag, w=W):
    ev = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not ev:
        return None
    acc = EventAccumulator(ev[-1], size_guidance={"scalars": 0}); acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return None
    vals = [s.value for s in acc.Scalars(tag)]
    if not vals:
        return None
    tail = vals[-w:]
    return sum(tail) / len(tail)


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return None, None, 0
    m = sum(xs) / n
    if n < 2:
        return m, 0.0, n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, var ** 0.5, n


def main():
    print(f"{'方法':<10}{'楼':<8}{'seed':<6}{'n_eval':<8}{'energy':<10}{'viol':<9}{'每区率':<9}{'comfort':<9}")
    print("-" * 70)
    summary = []
    for method, mp in [("SAC+MPC", "sacmpc_fair"), ("SAC", "sac_pure")]:
        for bld in ["small", "medium", "school"]:
            z = ZONES[bld]
            e_list, v_list, c_list = [], [], []
            for sd in SEEDS:
                prefix = f"{mp}_{bld}_1m_{sd}"
                d, n = pick_run_dir(prefix)
                if d is None:
                    print(f"{method:<10}{bld:<8}{sd:<6}{'MISSING':<8}")
                    continue
                e = tail_mean(d, T_E); v = tail_mean(d, T_V); c = tail_mean(d, T_C)
                e_list.append(e); v_list.append(v); c_list.append(c)
                rate = v / z * 100 if v is not None else float("nan")
                print(f"{method:<10}{bld:<8}{sd:<6}{n:<8}{e:<10.1f}{v:<9.3f}{rate:<9.1f}{(c if c is not None else float('nan')):<9.3f}")
            em, es, ne = mean_std(e_list); vm, vs, _ = mean_std(v_list); cm, cs, _ = mean_std(c_list)
            if em is not None:
                rate = vm / z * 100
                summary.append((method, bld, ne, em, es, vm, vs, rate, cm, cs))
    print("\n" + "=" * 70)
    print("3-seed 聚合 (末段8点窗 mean±std):")
    print(f"{'方法':<10}{'楼':<8}{'n_seed':<7}{'energy':<16}{'viol':<14}{'每区率':<9}{'comfort':<12}")
    print("-" * 70)
    for method, bld, ne, em, es, vm, vs, rate, cm, cs in summary:
        print(f"{method:<10}{bld:<8}{ne:<7}{em:.1f}±{es:.1f}    {vm:.3f}±{vs:.3f}   {rate:<9.1f}{cm:.3f}±{cs:.3f}")


if __name__ == "__main__":
    main()
