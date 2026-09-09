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

## Construction-only V1 baseline（基于 63ccd91）

最新产物在 [`outputs/v1/`](outputs/v1/)，中性检查在
[`outputs/neutral_construction/`](outputs/neutral_construction/)，完整运行记录见
[RUN_REPORT.md](RUN_REPORT.md)。原全数据中性结果逐文件保留在
`outputs/neutral_full_baseline/`，旧 `outputs/neutral/` 也保持不变。

```bash
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/neutral-mpl python -m skill_pool.cli neutral-check \
  --trajectories data/ego_trajs.npy --checkpoint weights/trajectory_vae_8d_best.pth \
  --construction-indices splits/construction_indices.npy --output-dir outputs/neutral_construction
# 打开中性检查两张图，确认通过；visual_review.json 记录目视结论及被检查产物 SHA256。
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/neutral-mpl python -m skill_pool.cli build-v1 \
  --trajectories data/ego_trajs.npy --checkpoint weights/trajectory_vae_8d_best.pth \
  --provenance data/ego_trajs_provenance.json --seed 7
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/neutral-mpl python -m pytest -q
```

`build-v1` 强制检查 construction-only 中性来源、精确 selection_indices、自动检查、
目视记录及其哈希；检查失败则在编码和选择前停止。首次或产物改变后须重新检查图像并
记录 `visual_review.json`，不会由建库命令自动把视觉验收标记为通过。已提交的目视记录
可用于复现完全相同的产物。比较函数为 `skill_pool.v1.compare_neutrals`。

冻结 VAE 的 encoder 均值可对全部有效行批量计算，但 `construction_latents.npy` 与
`heldout_latents.npy` 分别绑定同目录对应 `*_indices.npy`。`all_latents.npy` 的第 i 行
对应 `all_latent_source_indices.npy[i]`。Pool 选择器只接收 construction latent 和原始
行号；方向、0.995 强度裁剪和四档强度分位数全在 construction 计算。输出 .npz/.pt
保存 `selection_split=construction`、原始 source_indices、token、来源轨迹及解码轨迹。

K=32/64 均保留未去重 baseline。`duplicate_diagnostics.json` 保存所有无序技能对 ADE、
分位数、最接近的 10 对及多个参考阈值的敏感性计数，不指定最终去重阈值、不删除技能。
近静止敏感性同时使用平均逐帧速度 ≤t m/s 和最大距原点半径 ≤3t m，t 为
0.05/0.1/0.25/0.5；速度包含原点到首个 future 点，解码抖动会影响此值。

`report.json` 分别保存 construction/heldout 的 minADE、独立最小 minFDE、
FDE-at-minADE、使用技能数、逐技能分配次数及逐类型指标。pairwise ADE 是同一个 Pool 的
属性，不随评估集合变化。`coverage_<split>_<K>.npz` 保存逐样本距离、最近技能及原始行号。
空类别指标为 null；样本 <30 仅作为描述性不足标记，≥30 也不自动构成统计泛化证明。
图中 best/median/worst 是对全部 held-out minADE 稳定排序后取首项、中间项和末项。

此处 held-out 未参与中性估计和 Pool 拟合；冻结 checkpoint 原训练数据与这些 log 是否
独立尚未核实，因此这是 Pool 构建阶段的 held-out 评估。未修改或重训 VAE。

## 小规模路径能量 / geodesic Log 可行性实验

独立实验入口 `python -m skill_pool.log_experiment`，基于 V1 提交 75e6e5e。
`outputs/local_baseline/` 完整保留 V1 文件，`outputs/v1/` 本身也不改动。
仅以 construction 的描述量分层抽取 1,000 条，`log_experiment/candidate_indices.npy`
始终指向原始行；候选前缀通过行为/尾部层轮流取样，适合依次进行 1/10/100 条冒烟。
行为描述包括位移、速度及变化、横向位移、航向范围和曲率；近静止段速度≤0.2 m/s
不用于航向/曲率统计。负 X / 正 X 由当前提取旋转分别对应左 / 右。

```bash
OMP_NUM_THREADS=1 python -m skill_pool.log_experiment prepare
OMP_NUM_THREADS=1 python -m skill_pool.log_experiment stage --count 1
OMP_NUM_THREADS=1 python -m skill_pool.log_experiment stage --count 10
OMP_NUM_THREADS=1 python -m skill_pool.log_experiment stage --count 100
# 只有 100 条收敛率≥95%、数值检查通过、预计 1000 条≤3600 秒才能扩大。
# python -m skill_pool.log_experiment stage --count 1000
OMP_NUM_THREADS=1 python -m skill_pool.log_experiment sensitivity
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/neutral-mpl python -m skill_pool.log_diagnostics
```

采用固定首尾的分段线性 latent 路径，在每段 2 个 Gauss 点计算可微完整 decoder Jacobian，
按 `G(z)=J(z)^T J(z)/60 + lambda I` 积分 `E=1/2 ∫ v^T G v dt`。仅内部 latent 为优化参数，
冻结 VAE 以 float64 运算，不改变 checkpoint。LBFGS strong-Wolfe 线搜索，默认 L=8、
lambda=1e-4、最多100步。返回 `L*(path[1]-z0)`，即离散路径初速度估计。
收敛要求归一化能量梯度无穷范数≤1e-4；能量下降或停滞本身不代表收敛。

这实现了沿路径可变度量的数值 geodesic 优化，不能把它当作精确连续 Log 的证明。
原 decoder 的 ReLU 使度量仅分片光滑，激活边界可能令优化和积分不稳定；额外以4点积分
审计末态能量，并在固定100条上比较 L、damping 和迭代预算。
理论背景：[Rumpf 与 Wirth 的变分离散 geodesic calculus](https://arxiv.org/abs/1210.2097)。
该论文的离散收敛理论不能直接证明本项目非光滑网络的结果。

每次尝试均保留原 source index、能量曲线、梯度残差、路径、耗时、收敛状态；失败包含
完整 traceback。根目录 log_vectors.npy 对齐 evaluated_indices.npy，未尝试的候选不填成
有效零向量。统计分别报告全部暂定结果和已收敛子集；零范数角度/比例不定义，JSON为null。
实验不会调用 Pool 重建或读取 held-out 内容决定候选与 Log。

实验需要 PyTorch ≥2.0 的 `torch.func`，本次实际为 2.11.0、CPU、每进程1线程。
敏感性实际命令分别使用 `sensitivity --variant L16`、`--variant damping`、
`--variant iterations`，三个配置独立并发执行；因此变体的耗时可能含竞争，正式规模估算
以串行100条 baseline 实测为依据，不把并发变体时间当作纯串行性能对比。
