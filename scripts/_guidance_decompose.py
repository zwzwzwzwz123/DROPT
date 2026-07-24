"""guidance 训练期 vs 推理期贡献分解 (三点分解, 非朴素 η 扫描)。

问题: guidance 的好处来自训练稳定化还是推理期能量倾斜? (stage4 §6.2-5c 开放问题)

三点 (每点 = 从 checkpoint 重建 policy + 复现训练 test 协议):
  NoGuide          : 训练关/推理关  (加载 NoGuide checkpoint, 推理 η=0)   [已有 CSV, 此处复算作 sanity]
  Full-inferη0     : 训练开0.5/推理关 (加载 Full checkpoint, 推理 η=0)     ⬅ 唯一新点
  Full-inferη0.5   : 训练开0.5/推理开0.5 (加载 Full checkpoint, 推理 η=0.5) [≈ CSV Full, sanity]

分解:
  NoGuide → Full-inferη0    : 仅差"训练期是否开 guidance" → guidance 训练期贡献(稳定器)
  Full-inferη0 → Full-inferη0.5: 仅差"推理期是否加梯度" → guidance 推理期贡献(能量倾斜)

⚠️ 首跑验证闸门(必须先看): Full-inferη0.5 应 ≈ CSV Full、NoGuide 应 ≈ CSV NoGuide。
   对不上 = reload 协议/ checkpoint 与末8窗口径不一致, 分解数不可信, 先排查再解读。
⚠️ caveat(写论文交代): Full-inferη0 骨干仍是 guided-trained, "推理贡献"是"已引导骨干上再加推理引导"的增量, 非绝对值。

用法: python scripts/_guidance_decompose.py <small|medium|school> [--episodes 8]
需 GPU (K=6 反扩散采样); RAM 轻(无 1M buffer)。建议 RAM/GPU 空闲时跑。
"""
import argparse, glob, os, pickle
import numpy as np
import torch

import main_building_fno_guided_bcfix_clean as M
from diffusion import Diffusion
from diffusion.model import DoubleCritic
from diffusion.model_fno import DiffFNO
from policy import DiffusionOPT
from tianshou.data import Collector

ROOT = "log_building"
ZONES = {"small": 6, "medium": 18, "school": 25}
CKPT = "policy_final_fno_guided.pth"  # 末态最接近末段窗; sanity check 会暴露与末8窗的差

# 每楼 Full / NoGuide run 目录前缀 (来自 master_metrics_v2.csv 的 runs 列)
RUNS = {
    "small": {
        "Full": ["fno_guided_bcfix_clean_OfficeSmall_Hot_Dry_小_guidancescale=0.5_100万步",
                 "diffusion_fno_guided_bcfix_clean_OfficeSmall_Hot_Dry_20260411_100255__guided_seed0",
                 "diffusion_fno_guided_bcfix_clean_OfficeSmall_Hot_Dry_20260416_110146__guided_seed1"],
        "NoGuide": ["diffusion_fno_guided_bcfix_clean_OfficeSmall_Hot_Dry_20260403_085019_小_无引导100万步",
                    "diffusion_fno_guided_bcfix_clean_OfficeSmall_Hot_Dry_20260412_163742__noguide_seed0",
                    "diffusion_fno_guided_bcfix_clean_OfficeSmall_Hot_Dry_20260417_152156__noguide_seed1"],
    },
    "school": {
        "Full": ["school_guided_1m_s42_SchoolPrimary_Hot_Dry_20260708_160354",
                 "school_guided_1m_s0_SchoolPrimary_Hot_Dry_20260708_160618",
                 "school_guided_1m_s1_SchoolPrimary_Hot_Dry_20260708_160630"],
        "NoGuide": ["school_fno_noguide_1m_s42_SchoolPrimary_Hot_Dry_20260712_103848",
                    "school_fno_noguide_1m_s0_SchoolPrimary_Hot_Dry_20260712_104030",
                    "school_fno_noguide_1m_s1_SchoolPrimary_Hot_Dry_20260712_104047"],
    },
    # medium: NoGuide 未跑完(s42在跑), 待补; Full 用默认档
    "medium": {
        "Full": ["officemedium_fno_default_probe_s42_OfficeMedium_Hot_Dry_20260716_103411",
                 "officemedium_fno_default_1m_s0_OfficeMedium_Hot_Dry_20260717_194958",
                 "officemedium_fno_default_1m_s1_OfficeMedium_Hot_Dry_20260717_195025"],
        "NoGuide": [],  # 待 officemedium_fno_noguide_default_1m_s* 完训后填
    },
}
def load_args(run_dir):
    pk = glob.glob(os.path.join(ROOT, run_dir, "**", "*metadata*.pkl"), recursive=True)
    if not pk:
        return None
    m = pickle.load(open(pk[0], "rb"))
    a = m.get("args", m) if isinstance(m, dict) else m
    return a


def geta(a, k, default=None):
    return (a.get(k) if isinstance(a, dict) else getattr(a, k, default)) if a is not None else default


def build_policy(a, device):
    """按 run 的 args 重建 policy 结构 (与 main 的构建段一致)。"""
    # env 先建以拿 state_dim/action_dim
    env, train_envs, test_envs = M.make_building_env_bcfix_clean(
        building_type=geta(a, "building_type", "OfficeSmall"),
        weather_type=geta(a, "weather_type", "Hot_Dry"),
        training_num=1, test_num=1,
    )
    sd, ad = env.state_dim, env.action_dim
    fno = DiffFNO(state_dim=sd, action_dim=ad,
                  width=geta(a, "fno_width", 64), modes=geta(a, "fno_modes", 4),
                  n_layers=geta(a, "fno_layers", 1), t_dim=16,
                  activation=geta(a, "fno_activation", "mish")).to(device)
    critic = DoubleCritic(state_dim=sd, action_dim=ad,
                          hidden_dim=geta(a, "hidden_dim", 256)).to(device)
    actor = Diffusion(state_dim=sd, action_dim=ad, model=fno, max_action=1.0,
                      beta_schedule=geta(a, "beta_schedule", "vp"),
                      n_timesteps=geta(a, "diffusion_steps", 6),
                      guidance_scale=0.0, guidance_fn=None).to(device)
    policy = DiffusionOPT(
        state_dim=sd, actor=actor, actor_optim=torch.optim.Adam(fno.parameters(), lr=1e-4),
        action_dim=ad, critic=critic, critic_optim=torch.optim.Adam(critic.parameters(), lr=2e-5),
        device=device, tau=0.005, gamma=geta(a, "gamma", 0.95),
        exploration_noise=geta(a, "exploration_noise", 0.1),
        bc_coef=geta(a, "bc_coef", False), action_space=env.action_space,
    )
    return policy, actor, critic, test_envs


def eval_at_eta(policy, actor, critic, test_envs, eta, episodes, device, seed=42):
    """复现训练 test 协议: eval() + test_collector.collect + consume_metrics。"""
    if eta > 0:
        actor.set_guidance(M.build_guidance_fn(critic, device), eta)
    else:
        actor.set_guidance(None, 0.0)
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    policy.eval()
    coll = Collector(policy, test_envs)
    coll.reset()
    coll.collect(n_episode=episodes)
    envs = getattr(test_envs, "_env_list", [])
    vals = [e.consume_metrics() for e in envs]
    vals = [v for v in vals if v]
    if not vals:
        return None
    agg = {k: float(np.mean([v[k] for v in vals if v.get(k) is not None]))
           for k in ("avg_energy", "avg_violations", "avg_comfort_mean")}
    return agg


def run_config(bld, variant, eta, episodes, device):
    """对一个 variant 的所有 seed 求 (energy, viol) 3-seed 均值。"""
    dirs = RUNS[bld][variant]
    es, vs = [], []
    for d in dirs:
        ck = os.path.join(ROOT, d, CKPT)
        if not os.path.exists(ck):
            print(f"    [skip] no ckpt: {d}/{CKPT}")
            continue
        a = load_args(d)
        policy, actor, critic, test_envs = build_policy(a, device)
        try:
            policy.load_state_dict(torch.load(ck, map_location=device))
        except Exception as ex:
            print(f"    [WARN] load_state_dict partial/failed for {d}: {ex}")
        r = eval_at_eta(policy, actor, critic, test_envs, eta, episodes, device)
        if r:
            es.append(r["avg_energy"]); vs.append(r["avg_violations"])
            print(f"    {d[:40]:42} E={r['avg_energy']:.1f} V={r['avg_violations']:.3f}")
    if not es:
        return None
    z = ZONES[bld]
    return (float(np.mean(es)), float(np.std(es, ddof=1) if len(es) > 1 else 0.0),
            float(np.mean(vs)), float(np.mean(vs)) / z * 100, len(es))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("building", choices=["small", "medium", "school"])
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    bld = args.building; dev = torch.device(args.device)
    print(f"=== guidance 三点分解: {bld} (episodes={args.episodes}, {dev}, std=ddof1) ===\n")

    print("[1/3] NoGuide (训练关/推理关):")
    ng = run_config(bld, "NoGuide", 0.0, args.episodes, dev)
    print("\n[2/3] Full-inferη0 (训练0.5/推理关):")
    f0 = run_config(bld, "Full", 0.0, args.episodes, dev)
    print("\n[3/3] Full-inferη0.5 (训练0.5/推理0.5):")
    f5 = run_config(bld, "Full", 0.5, args.episodes, dev)

    print("\n" + "=" * 62)
    print(f"{'配置':<20}{'能耗':<16}{'每区违规率':<12}{'n_seed'}")
    for name, r in [("NoGuide", ng), ("Full-inferη0", f0), ("Full-inferη0.5", f5)]:
        if r:
            print(f"{name:<20}{r[0]:.1f}±{r[1]:.1f}     {r[3]:<12.1f}{r[4]}")
        else:
            print(f"{name:<20}(缺 checkpoint/数据)")
    print("=" * 62)
    if ng and f0:
        print(f"训练期贡献 (NoGuide→Full-inferη0):   能耗 {ng[0]-f0[0]:+.1f} / 每区率 {ng[3]-f0[3]:+.1f}pp")
    if f0 and f5:
        print(f"推理期贡献 (Full-inferη0→Full-inferη0.5): 能耗 {f0[0]-f5[0]:+.1f} / 每区率 {f0[3]-f5[3]:+.1f}pp")
    print("\n⚠️ 首跑验证闸门: Full-inferη0.5 应≈CSV Full、NoGuide 应≈CSV NoGuide。对不上先排查 reload 协议再解读分解。")


if __name__ == "__main__":
    main()

