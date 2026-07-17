# 已弃用脚本归档

这些是被 `*_bcfix_clean` / `*_bcfixclean` 版本取代的旧脚本。**保留仅为血缘追溯，勿用于新实验。**

当前在用的对应版本（在仓库根目录）：

| 已弃用（本目录） | 现用替代（根目录） | 说明 |
|---|---|---|
| `main_building_fno.py` | `main_building_fno_guided_bcfix_clean.py` | FNO 血缘基准，早期无 guided/bcfix |
| `main_building_fno_guided.py` | `main_building_fno_guided_bcfix_clean.py` | pre-bcfix 版 guided FNO |
| `main_building_fno_guided_nores.py` | `main_building_fno_guided_bcfix_clean_ablation.py` | 残差消融现由 ablation 脚本覆盖 |
| `main_building_icl.py` | （无）| in-context 变体，已无引用 |
| `rl_baseline.py` | `rl_baseline_bcfixclean.py` | 纯 SAC 旧版 |
| `rl_baseline_mpc.py` | `rl_baseline_mpc_bcfixclean.py` | SAC+MPC 旧版 |

已知问题：旧版 SAC 基线用 `critic_lr=2e-5`（actor_lr 的 1/15），SAC 无法学习、能耗发散——详见 `docs/HANDOFF_sac_baseline.md`。公平重跑一律用 `scripts/_sac_launch.sh` + `*_bcfixclean.py`。
