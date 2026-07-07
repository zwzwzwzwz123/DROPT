# HANDOFF —— Guided-DiffFNO 期刊方向 / Sinergym 转向

> 日期：2026-07-07　　接续自 `HANDOFF_journal_direction.md`
> 目标（用户原话）：不追顶刊，只要内容自洽、故事完整、能发表即可（目标档位 IEEE IoT-J 类）。

---

## 0. 一句话现状

BEAR 上的"双路(局部支路)降越界"方向**已被本轮实验数据否决**；唯一被证实的资产是"**FNO 是跨规模稳健骨干**"。决定**转向 Sinergym(EnergyPlus)** 找新的完整增量，Sinergym 已验证在本机 Windows 可跑。方向故事(同环境验证 vs 跨建筑泛化)**尚未拍板**，代码尚未动。

---

## 1. 本轮否决的方向：双路 / 门控（有数据，别再走回头路）

### 1.1 背景
- 用户直觉：FNO 在 School(大楼、多 zone)越界指标不好，想加一条**局部卷积支路**降越界。
- 6 月做过 `mixture`(逐区自适应门控双路)，印象"好用"(越界曾到 4.10)。
- 本轮把双路拉到与纯 FNO **同预算**(160k 步 / seed0 / 无引导 / w64)做公平对比。

### 1.2 关键结论（末段8点窗口均值 ± std，越界 avg_violations，越低越好）

| 模型 | 越界窗均值 | std | 参数量 |
|---|---|---|---|
| 纯FNO-b | **8.60** | 0.67 | 34k 或 211k* |
| 纯FNO-a | 10.31 | 1.28 | (同上，撞目录分不清) |
| 双路-adaptive(门控) | 13.25 | 1.56 | 229k |
| MLP | 14.98 | 2.25 | 235k |
| 双路-fixed0.5 | 21.69(终盘发散) | 1.14 | 195k |

（*两条纯FNO(w48/w128)因同秒启动撞进同一目录、checkpoint 互相覆盖，标量数据未混但分不清谁是谁。）

### 1.3 结论
- **双路(加局部支路)不降越界，反而更差**：adaptive 13.25 > 纯FNO 8.6–10.3，差距 2–3 倍 std，非噪声，且参数量已对齐(排除容量解释)。
- **固定 0.5 更差且不稳**：终盘发散到 21+。→ "门控 vs 固定"里门控胜，但两者都输给纯 FNO。
- **6 月"4.10"是口径错觉**：那是 w96 + 308k步 + 单episode 走运最小值；窗均值重读当时也在 ~10。
- **"FNO 在 School 打不过 MLP"这个原始动机是错的**：清洗后 FNO 各项都赢 MLP(能耗 6000 vs 12800、越界 9 vs 15、reward 更稳)。

### 1.4 血泪教训（务必延续）
- **越界对比一律用"末段窗口均值 + std"，禁用单点最小值**（最小值陷阱坑了两次：MLP 的 7.45、双路的 4.10）。
- **单 seed(std 0.67~2.25)只能筛大信号，筛不出 ~1-2 的小效果**；任何"更好"结论最终要 3 seed(0/1/42)坐实。
- **并行跑同一脚本时，日志目录名精确到秒会撞车**→ 必须用不同 `--log-prefix`。

---

## 2. 唯一被证实的资产：FNO 跨规模稳健

同款扩散策略只换去噪器，OfficeSmall 1M 步：

| 指标 | Diff-MLP | Guided-DiffFNO |
|---|---|---|
| 能耗 | 1106 | **877**(↓21%) |
| 越界率 | 0.266 | **0.088**(3×更好) |
| Action Diff MSE | 0.0178 | **0.0011**(16×更平滑) |

- Office：FNO(34k参数)碾压 MLP(235k)——**优势与容量无关，是结构**。
- School：FNO 仍赢 MLP，但优势收窄(越界 ~9 vs ~15，不再是 3 倍)。
- **交叉规律**：FNO 跨 Office→School 都稳健；这是可发表故事的核心素材。

---

## 3. Sinergym 转向 —— 可行性已验证

### 3.1 为什么换环境
- BEAR 真动力学是**线性**(`X_new=A_d·X+B_d·Y`)→ 梯度引导平凡、创新平凡，是"故事做不大"的病根。
- 换非线性环境后，引导重新非平凡；FNO 的"离散无关/跨规模"卖点也能升级成"跨建筑/跨气候"。
- 用户先排除了两个：**SustainDC**(动作粒度粗，只能升降温，不能精细 setpoint) + **自造 DataCenterEnv**(太简陋)。二者代码用户已删。

### 3.2 本机可行性（已实测，非推测）
- **能跑**：sinergym 3.12.0 + Python 3.12.7 + EnergyPlus **25.2.0** + 隔离 venv `.venv_sinergym/`。
- **两个 Windows 坑及解法**（关键，重装必踩）：
  1. `sinergym/config/modeling.py` `import fcntl` → Unix 专有，Windows 崩。已 patch：改成 try/except shim(fcntl 只在建目录时做文件锁，单进程不需要)。**注意：pip 重装会覆盖这个 patch，需重打。**
  2. `pyenergyplus` 找不到 → 需 `PYTHONPATH` 指向 `C:\EnergyPlusV25-2-0`。
- **依赖冲突**：sinergym 要 numpy≥2.3.2 / gymnasium≥1.2.0，主环境是 numpy1.26.4/gym1.1.1/tianshou0.5.1/torch2.7.1。**必须隔离 venv，绝不能装进主环境**（会崩 BEAR）。
- **控制粒度**：`Eplus-5zone-hot-continuous-v1` 动作 `Box(2)`=[制热setpoint 12–23.25, 制冷setpoint 23.25–30]，obs 17 维 → **真连续 setpoint**，满足精细控制需求（解决了 SustainDC 的痛点）。
- **速度**：纯 env ~2000 steps/s(500步测得，仅覆盖 episode 1.4%，稳态可能慢几倍但仍非瓶颈)。episode 默认=整年 35040 步@15min，可用 `runperiod`/`timesteps-per-hour` 缩短。真正瓶颈是扩散策略 per-step 采样(6步反向扩散)，与环境无关，换环境躲不掉。
- **素材**：3 种数据中心(1Zone/2Zone×2) + 5Zone + OfficeMedium + Warehouse + 大办公储能 + Shop+PV电池 + 辐射住宅，各含 hot/mixed/cool 三气候。

### 3.3 用户 3 月的半成品（可当起点）
- 写过 `main_sinergym_fno_guided.py` + `env/sinergym_env_wrapper.py`，**源码已删、从未进 git，只剩 `.pyc`**(cpython-312/310)在 `__pycache__/`。
- 从 `.pyc` 抽出的字符串还原出当时设计：
  - 支持 env_id：`5zone`/`datacenter`/`smalldatacenter`，RBC 控制器 `RBC5Zone`/`RBCIncrementalDatacenter`
  - 默认 `Eplus-5zone-hot-continuous-v1`，用 `NormalizeAction`/`NormalizeObservation`
  - **BC 专家来源**：`midpoint` 或 `rbc` 控制器（不需要 BEAR 那种 MPC oracle）
  - 骨干直接接 DiffFNO + critic guidance
  - CLI：`--sinergym-env-id/--sinergym-expert-policy/--sinergym-runperiod/--sinergym-timesteps-per-hour` 等
- **可反编译 `.pyc` 还原**（uncompyle 不支持 3.12，但可用 3.10 的 .pyc 或 marshal 抽结构；或直接照字符串重写）。

---

## 4. 待拍板的岔路（下一步的前置，未决）

Sinergym 上讲哪个故事——**这个定了才动代码**：

- **A. 同环境验证**：5zone 上证明"扩散+FNO+引导" > SAC/MLP。**易实现，但弱**（审稿人："BEAR 不是做过了，so what"）。
- **B. 跨建筑/跨气候泛化**：一个 FNO 骨干在多建筑/多气候稳健，甚至零样本迁移。**难但强**，踩中 FNO 离散无关理论卖点，是完整增量。
  - **B 的拦路石**：BEAR 里 FNO 的门控/输出是维度写死的(如 25 维)；跨建筑 zone 数不同，需先做成"维度无关"(如通道全局池化出标量/低维)才能迁移。

倾向：B 更有"完整故事"分量且发挥 FNO 独特性；A 太薄。但换环境是"换地基"(env wrapper/专家/日志全重写，骨干可搬)，成本几天。

---

## 5. 建议的下一步（供接续者选）

1. 反编译/还原 3 月的 `main_sinergym_fno_guided.py` + `sinergym_env_wrapper.py`，看当时到底搭到哪一步。
2. 定 A/B 方向。
3. 若 B：先解决 FNO"维度无关"改造，再设计跨建筑最小实验(先 2 建筑单 seed 筛信号，再放大)。
4. 用 `runperiod` 缩短 episode 控训练成本；先单 seed 筛信号，信号成立再 3 seed(0/1/42)。

---

## 6. 仓库卫生备注（GitHub 推送相关）—— 已于 2026-07-07 推送

- 提交 `2e086b8 "新增sinergym环境，删除不必要文件"` 已 push 到 `origin/dcopt`（66 files，+1819/-30905）。已核对：**未混入 venv/exe/大文件**。
- `.venv_sinergym/`(152文件含 .exe，数百MB)一度被 staged；已加进 `.gitignore` 并 `git rm --cached` 撤出。**确认 HEAD 里跟踪数 0**。pip 重装 sinergym 会覆盖 `modeling.py` 的 fcntl patch，需重打。
- 修复已推送：`diffusion/__init__.py` 现为干净两行（`Diffusion` + `DiffFNO`），无崩溃 import。
- 已从 repo 移除：`rectified_flow*.py`(含 `diffusion/rectified_flow*.py`)、`datacenter*`、`sustaindc*`、`main_datacenter/sustaindc.py`、根 `requirements.txt`、`environment.yml` 等。
- **依赖清单缺失问题已修复**：重新生成了根目录 `requirements.txt`（BEAR 主实验用，钉了真实版本：torch2.7.1+cu118/tianshou0.5.1/gymnasium1.1.1/numpy1.26.4/scipy/pandas/sklearn/cvxpy/tensorboard/matplotlib/tqdm/pyyaml/sb3）。sinergym 依赖**故意不列**（属隔离 venv，且 numpy/gym 版本冲突）。**此文件尚未 commit。**
- BEAR 主链路 import 冒烟：**11 模块全 OK**，删文件不影响主实验。
- **待办**：`requirements.txt`(新) 和 `docs/HANDOFF_sinergym_direction.md`(本文件) 目前是未跟踪状态，尚未 commit/push。

