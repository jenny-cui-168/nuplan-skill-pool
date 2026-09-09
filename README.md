# nuPlan 8-D VAE Skill Pool Pipeline

本项目使用 `real-Skillformer-master` 的冻结 8 维轨迹 VAE，从 nuPlan mini 的 ego future
轨迹中提取 32/64 个候选技能基元。每个基元是一个 8 维 latent，解码后为 3 秒、10 Hz、
形状为 `[30, 2]` 的局部参考轨迹。

## 方法

本实现不使用 K-means 或其他聚类算法：

1. 用 VAE encoder 的均值 `mu` 编码每条轨迹；
2. 从典型直行轨迹中选择中性基元 `z0`；
3. 在 `z0` 计算 decoder Jacobian，并构造 pullback metric
   `G = J_D(z0)^T J_D(z0) / 60 + lambda I`；
4. 在该度量下归一化 `z-z0`，用 spherical farthest-point sampling 选技能方向；
5. 每个方向按强度分位数选择真实数据中的 latent。默认 4 档强度，因此：
   - K=32：8 个方向 × 4 档强度；
   - K=64：16 个方向 × 4 档强度。

这是一版可运行的局部 Riemannian 近似。它使用 decoder 几何，但不声称已经求解精确的全局
geodesic Log map；后者可以作为后续论文消融实验加入。

## 坐标约定

与当前 real-Skillformer 实际代码一致：局部 `+Y` 为车辆前方，`X` 为横向，原点为初始
ego rear axle。原代码头部将 `+X` 写成前方是文档错误。

## 安装

建议使用 Python 3.10 或 3.11：

```bash
cd /Users/cuijianing/Documents/ChatGPT/研究生课题/skill-pool-pipeline
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

如需直接从 nuPlan mini 提取，再在兼容环境中安装 nuPlan devkit。nuPlan 的依赖版本可能
比本项目严格，实际使用时可把本项目以 editable 模式安装到已有 nuPlan 环境。

## 从 nuPlan mini 提取轨迹

```bash
skill-pool extract-nuplan \
  --data-root /path/to/nuplan-v1.1/splits/mini \
  --map-root /path/to/nuplan/maps \
  --output data/ego_trajs.npy
```

先用少量场景验证路径和依赖：

```bash
skill-pool extract-nuplan \
  --data-root /path/to/nuplan-v1.1/splits/mini \
  --map-root /path/to/nuplan/maps \
  --limit-total-scenarios 100 \
  --output data/ego_trajs_smoke.npy
```

## 一键构建 32/64 Skill Pool

```bash
skill-pool build \
  --trajectories data/ego_trajs.npy \
  --checkpoint weights/trajectory_vae_8d_best.pth \
  --pool-sizes 32 64 \
  --output-dir outputs
```

## 输出

```text
outputs/
├── latents_mu.npy
├── latents_logvar.npy
├── neutral_latent.npy
├── decoder_metric_z0.npy
├── skill_pool_32.npz
├── skill_pool_32.pt
├── skill_pool_32.png
├── skill_pool_64.npz
├── skill_pool_64.pt
├── skill_pool_64.png
├── report.json
└── run_config.json
```

`report.json` 包含 VAE 重建误差、真实轨迹到最近技能的 minADE/minFDE、技能间多样性、实际被
分配到样本的技能数量等指标。先检查 VAE reconstruction ADE；如果它异常大，应先排查数据
坐标系和 checkpoint，而不是解释 Skill Pool 结果。

## 与 PufferDrive 的边界

这个仓库完成“nuPlan 轨迹 → 32/64 技能库”。PufferDrive 闭环属于下一阶段，还需要把
`[30,2]` 参考轨迹转换成 acceleration/steering 的轨迹跟踪器。

## 独立中性技能验收

```bash
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/neutral-mpl python -m skill_pool.cli neutral-check \
  --trajectories data/ego_trajs.npy \
  --checkpoint weights/trajectory_vae_8d_best.pth --output-dir outputs/neutral
```

该命令只生成中性诊断，不构建 Pool。默认读取同目录 `ego_trajs_tokens.npy`。
future 点的时间为 0.1–3.0 秒，速度包含原点到首点的区间；航向从 +Y 计算，
航向变化采用整个窗口的最大值减最小值。正常运动（每步前进、平均速度 2–25 m/s）
的轨迹平均速度中位数为参考速度。原始及解码轨迹均须通过固定门限：最大绝对横移
≤0.5 m，最大绝对航向及航向范围 ≤5°，速度标准差 ≤0.5 m/s，速度范围 ≤2 m/s。
通过后按解码到标准轨迹 ADE 升序选择真实样本，平分时按原始行号选择，无合格项则报错停止。
门限是本次工程验收定义，不是通用驾驶标准。参数、前十名和全部检查记录在 JSON。
输出包含标准轨迹、原始来源行号/token/轨迹、z0、解码轨迹、G0 和两张 PNG。
旧 `build` 路径未接入此验收，后续建库须显式复用验收结果。

## 按 nuPlan log 划分 construction / held-out

现有数据优先通过只读数据库 token 关联补录来源，不重提轨迹：

```bash
python -m skill_pool.cli recover-provenance --trajectories data/ego_trajs.npy \
  --database-root ../nuplan/dataset/nuplan-v1.1_mini/data/cache/mini
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/neutral-mpl python -m skill_pool.cli split-logs \
  --trajectories data/ego_trajs.npy --provenance data/ego_trajs_provenance.json --seed 7
```

`ego_trajs_provenance.json` 的每行保存原数组索引、scenario_token、log_name、
scenario_type 和 database_source，并绑定轨迹/token 文件 SHA256。多标签采用与当前
提取 devkit 一致的 `MAX(type)`（先应用原提取类型过滤），缺失或多数据库匹配时停止。
今后的提取会先完成轨迹及全部来源字段，再同步追加成功行，避免失败造成错位。

先排序唯一 log，再用 seed=7 排列，按四舍五入取 80% log。随后确定性地交换整 log，
平衡场景类型占比及轨迹数；目标函数、初始分布和最终分布均记录在 split_report.json。
不按单条轨迹拆分，不使用模型输出，不强制中性来源进入指定集合。数组索引始终指向
原始 ego_trajs.npy；非有限轨迹排除且计数。固定中性四个文件由
`outputs/neutral/baseline_config.json` 的哈希校验，划分不修改它们。

主要类型定义为全体有效轨迹占比 ≥1%。分布检查同时要求两组占比绝对差 ≤10 个百分点，
held-out/construction 占比比值在 [0.5,2]；单 log 占某主要类型 ≥80% 时明确记录集中限制，
不把该类型从失败列表移除。本次 `behind_long_vehicle` 的 653/663 条集中在一个 log，
因此其分布检查失败；其他主要类型通过。测试验证此限制被准确报告，测试通过不代表
所有类型均衡。分组文件是固定可复现的数据准备产物，不表示已经完成模型评估。

结果位于 `splits/`，图位于 `outputs/split/`。当前 log_name 是 devkit 定义的数据库分段名；
这是 log 级隔离，不额外声称同一天同一车辆的不同日志分段也互相隔离。
