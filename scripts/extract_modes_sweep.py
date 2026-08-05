"""School FNO modes 扫描抽数 (m2/m4/m8 各 3-seed, 末 8 点窗均值 + 组均±std, ddof=1)。

用途: ① 画 §3.1 待补的 3×3 方阵稳健性图; ② 审稿人追问 modes 选择依据时复现。
判决 (2026-08-02, 3seed×3modes 全满): **边界 null**
  - 违规率铁平: 每区率 28.4/28.3/28.1% (跨全部 9 个种子稳) —— 硬结果
  - 能耗弱单调掉进噪声: 6414±43 / 6418±22 / 6449±10, m8-m2=35kWh(0.55%), 差/std比 1.32
  - 写作定论: 别写"截断=省能耗机制"(过度包装); 用**稳健性**——跨 15%->62% 保留率仅摆 0.55%,
    坐实 modes=4 非调参刀尖值 (支柱1 防御点)。详见 docs/HANDOFF_stage5.md §3 + memory。
单变量成立性: 三组只差 --fno-modes, 其余走默认档 (配置漂移已核对, 见 handoff §3)。
"""
import glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = "log_building"; ZONES = 25
T_E, T_V, T_C = "test/avg_energy", "test/avg_violations", "test/avg_comfort_mean"

# 每 mode -> {seed: 目录}。m4 = 默认档 anchor (即三建筑主表的 School Full)。
GROUPS = {
    "m2(15%)": {
        "s42": "school_fno_modes2_default_1m_s42_SchoolPrimary_Hot_Dry_20260725_121926",
        "s0":  "school_fno_modes2_default_1m_s0_SchoolPrimary_Hot_Dry_20260728_095049",
        "s1":  "school_fno_modes2_default_1m_s1_SchoolPrimary_Hot_Dry_20260731_153541",
    },
    "m4(31%)": {
        "s42": "school_guided_1m_s42_SchoolPrimary_Hot_Dry_20260708_160354",
        "s0":  "school_guided_1m_s0_SchoolPrimary_Hot_Dry_20260708_160618",
        "s1":  "school_guided_1m_s1_SchoolPrimary_Hot_Dry_20260708_160630",
    },
    "m8(62%)": {
        "s42": "school_fno_modes8_default_1m_s42_SchoolPrimary_Hot_Dry_20260726_125402",
        "s0":  "school_fno_modes8_default_1m_s0_SchoolPrimary_Hot_Dry_20260729_104813",
        "s1":  "school_fno_modes8_default_1m_s1_SchoolPrimary_Hot_Dry_20260801_153749",
    },
}

def tail(d, tag, w=8):
    ev = sorted(glob.glob(os.path.join(ROOT, d, "events.out.tfevents.*")))
    if not ev: return None
    acc = EventAccumulator(ev[-1], size_guidance={"scalars": 0}); acc.Reload()
    if tag not in acc.Tags().get("scalars", []): return None
    v = [s.value for s in acc.Scalars(tag)]
    return sum(v[-w:]) / len(v[-w:])

def ms(xs):
    n=len(xs); m=sum(xs)/n
    if n<2: return m,0.0
    return m,(sum((x-m)**2 for x in xs)/(n-1))**0.5

means={}
for g,seeds in GROUPS.items():
    es,vs=[],[]
    print(f"\n=== {g} ===")
    for sd,d in seeds.items():
        e=tail(d,T_E); v=tail(d,T_V); c=tail(d,T_C)
        es.append(e); vs.append(v)
        print(f"  {sd}: energy={e:.1f}  每区率={v/ZONES*100:.1f}%  comfort={c:.3f}")
    em,estd=ms(es); vm,vstd=ms(vs)
    means[g]=(em,estd,vm/ZONES*100)
    print(f"  >>> {len(es)}-seed: energy {em:.1f}±{estd:.1f} | 每区率 {vm/ZONES*100:.1f}%")

print("\n"+"="*50)
print("能耗组均排序 (方向判决):")
order=sorted(means,key=lambda k:means[k][0])
print("  "+" < ".join(f"{k}={means[k][0]:.0f}" for k in order))
m2e,m2s=means["m2(15%)"][:2]; m8e,m8s=means["m8(62%)"][:2]
gap=m8e-m2e; pooled=(m2s+m8s)/2 if (m2s+m8s)>0 else 1
print(f"\nm8-m2 组均差 = {gap:.0f} kWh ({gap/m2e*100:.2f}%)")
print(f"m2/m8 组内 std = {m2s:.0f}/{m8s:.0f}, 均 {pooled:.0f}")
print(f"差/std比 = {gap/pooled:.2f}  (远>1 才算方向可信)")
print("单调是否保持(m2<m4<m8): " + ("是" if order==['m2(15%)','m4(31%)','m8(62%)'] else f"否 -> {order}"))
