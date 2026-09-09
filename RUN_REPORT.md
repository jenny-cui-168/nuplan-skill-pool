# nuPlan mini Skill Pool 实际运行报告

已完整执行交接流程，新项目位于 skill-pool-pipeline，原 transfer 保留，未修改 real-Skillformer-master。仅 editable 安装新项目，使用 --no-deps --no-build-isolation，没有升级依赖。计算使用 CPU。

## 环境与路径

```json
{
  "python": "3.10.20 (main, Jun 11 2026, 15:17:37) [GCC 14.3.0]",
  "python_executable": "/home/jennycui/anaconda3/envs/skillformer/bin/python",
  "versions": {
    "torch": "2.11.0+cu128",
    "numpy": "2.2.6",
    "matplotlib": "3.10.9",
    "nuplan-devkit": "2.0.0",
    "nuplan-skill-pool": "0.1.0"
  },
  "mini_path": "/home/jennycui/下载/yanjiusheng/nuplan/dataset/nuplan-v1.1_mini/data/cache/mini",
  "map_container_root": "/home/jennycui/下载/yanjiusheng/nuplan/dataset/maps1",
  "map_root_used_by_devkit": "/home/jennycui/下载/yanjiusheng/nuplan/dataset/maps1/nuplan-maps-v1.0/maps",
  "db_count": 64,
  "checkpoint_sha256": "9918138beece445613918bc2c2e9528df3ef8837844347a47e8f15dff96300a9",
  "warnings": [
    "Initial matplotlib import used temporary cache because ~/.config/matplotlib is not writable; subsequent runs set MPLCONFIGDIR."
  ],
  "changes": [
    "Added --scenarios-per-type 0 to disable per-type cap",
    "Retain all extraction failures with full tracebacks and reject nonfinite trajectories",
    "Correct minFDE to independently minimize final displacement; preserve FDE at minimum ADE in separate metric"
  ],
  "evaluation_scope": "Real mini data; 14 configured scenario types, no per-type cap; coverage evaluated on pool source data, not held-out generalization."
}
```

## 数据与范围

mini 有 64 个 .db，目录名并非 nuplan-v1.1/splits/mini。maps1 是包含 nuplan-maps-v1.0 的容器目录，实际 devkit --map-root 使用其下的 nuplan-maps-v1.0/maps，内含版本 JSON 和四个城市地图。

冒烟成功 100，失败 0；shape (100,30,2)，float32，finite=True，重建 ADE 0.077256806 米。已检查 outputs/smoke 的两张技能图和 reconstruction.png。

正式提取使用项目默认 14 类场景，解除每类 100 条上限；不是所有类型、所有帧的穷举。数据库中对应标签行 135741，devkit 筛选后返回 57992 个场景，全部提取成功，失败 0。标签行数与筛选后场景数不是同一计数单位。

数据文件 data/ego_trajs.npy，shape (57992,30,2)，float32，finite=True；13918208 字节（约 13.27 MiB）。每条 3 秒、30 个 future 点、10 Hz，后轴原点，局部 +Y forward、X lateral。tokens 和 metadata 位于 data 同目录。

## 最终指标（米）

| K | minADE | minFDE | FDE at minADE | pairwise ADE diversity | used skills |
|---|---:|---:|---:|---:|---:|
| 32 | 0.657350 | 1.435750 | 1.497778 | 5.547190 | 16 |
| 64 | 0.607371 | 1.195012 | 1.297097 | 4.132502 | 26 |

重建指标：
```json
{
  "vae_reconstruction_ADE_mean_m": 0.0896391049027443,
  "vae_reconstruction_ADE_p90_m": 0.1977764517068863,
  "vae_reconstruction_FDE_mean_m": 0.17682932317256927
}
```

minFDE 已修正为对终点误差独立取最小值；旧算法的 ADE 最佳候选对应 FDE 保留在 coverage_FDE_at_minADE_mean_m。指标来自真实数据，不含合成数据；coverage 在建库数据上计算，不能当作独立测试集泛化能力。

## 验证与局限

6 项测试通过。checkpoint strict 加载，eval 模式且所有参数 requires_grad=False；latent 为 (32,8)/(64,8)，解码为 (32,30,2)/(64,30,2)，全部有限值，各组 source_indices 无重复。

已人工查看最终两张图，行驶轨迹朝局部 +Y 展开；静止点有厘米级解码抖动，最小 Y 分别约 -0.031/-0.046 米，没有终点 Y<-0.1 米的技能。

虽然源索引互异，存在几乎相同的静止技能：最小 pairwise ADE 约 2.63e-6/1.41e-6 米；最近邻实际使用仅 16/32、26/64。当前池存在冗余，候选总数不能当作有效不同行为数。保留原选取方法，没有为美化指标替换算法。

## 错误与警告

提取、构建和测试没有异常 traceback，场景失败明细为空。初次环境验证及一次 devkit 源码检查出现 Matplotlib 默认缓存目录不可写警告，自动使用 /tmp；正式命令已显式设置 MPLCONFIGDIR。路径搜索 /data、/mnt 等不存在或不可访问时 find 返回 1，错误输出按交接搜索命令重定向；工作区数据与地图实际读取成功。没有安装或升级 torch/nuPlan。

logs/extract_smoke.log、extract_final.log、build_smoke.log、build_final.log 保留运行输出；logs/tests.log 为测试结果，logs/environment.json 为版本和校验值。

## 产物

outputs/final/skill_pool_32.pt、skill_pool_64.pt 以及对应 .npz、.png。
outputs/final/report.json、run_config.json、validation.json 为指标、参数和形状验证。
scripts/reproduce_final.sh 为正式运行复现命令（会覆盖同名输出）。


## 2026-09-09 中性技能独立验收

未重新提取 nuPlan，未修改或重训 real-Skillformer-master 的 8 维 VAE，未实现严格 geodesic Log，未构建新 Pool。使用冻结 checkpoint、CPU、encoder 均值选择真实样本。

正式命令（调整绘图横轴范围后重复执行一次）：
```bash
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/neutral-mpl python -m skill_pool.cli neutral-check --trajectories data/ego_trajs.npy --checkpoint weights/trajectory_vae_8d_best.pth --output-dir outputs/neutral
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/neutral-mpl python -m pytest -q
```

全部测试结果：
```text
...............                                                          [100%]
15 passed in 5.81s
```
新增测试覆盖方向、速度、转弯/横移/加减速拒绝、合格直线、非有限值、源索引不因无效行错位、G0 对称正定及 PNG/JSON 生成。

诊断图：[neutral_check.png](outputs/neutral/neutral_check.png)、[neutral_candidates.png](outputs/neutral/neutral_candidates.png)。JSON：[neutral_check.json](outputs/neutral/neutral_check.json)、[metric_check.json](outputs/neutral/metric_check.json)。同目录保存 neutral_target.npy、neutral_source_index.npy、neutral_source_token.npy、neutral_source_trajectory.npy、z0.npy、neutral_decoded.npy、G0.npy。

人工/目视检查记录：执行代理已通过图像查看工具打开并逐图检查最终两张 PNG（并非声称用户本人已审阅）。等比例 +Y 前向轨迹基本重合，前十名没有明显转弯或横移。D(z0) 有厘米级横向抖动及小幅逐帧速度抖动，但没有明显持续加减速；符合固定验收门限，无需替换中性点。相同比例会令厘米级差别不明显，细节由横移、航向与速度曲线及标注补充。

参考速度为正常前进轨迹平均速度的中位数；完整固定门限与排序方法见 README 和 JSON。结果：
```json
{
  "reference_speed_mps": 9.76328751379451,
  "reference_speed_method": "Median of per-trajectory mean speeds, including origin-to-first-future-point; finite, every Y step > 0, mean speed in [2,25] m/s",
  "normal_motion_count": 29680,
  "source_candidate_count": 17054,
  "decoded_eligible_count": 15113,
  "neutral_source_index": 8052,
  "neutral_source_token": "23383639f1415edc",
  "original": {
    "forward_displacement_m": 29.373266220092773,
    "final_lateral_displacement_m": 0.0030084410682320595,
    "max_lateral_displacement_m": 0.05271231010556221,
    "mean_speed_mps": 9.791173930810487,
    "speed_std_mps": 0.09311410573356824,
    "speed_range_mps": 0.3613596150259326,
    "heading_change_deg": 0.8316551848229123,
    "max_heading_deg": 0.529194360807527,
    "min_forward_step_m": 0.9626736640930176,
    "target_ade_m": 0.05224359871441398
  },
  "decoded": {
    "forward_displacement_m": 29.286865234375,
    "final_lateral_displacement_m": 0.05885349214076996,
    "max_lateral_displacement_m": 0.05885349214076996,
    "mean_speed_mps": 9.762910499228814,
    "speed_std_mps": 0.18713751921628471,
    "speed_range_mps": 0.8564436355794616,
    "heading_change_deg": 2.989624559002452,
    "max_heading_deg": 1.9083106397534237,
    "min_forward_step_m": 0.9346485137939453,
    "target_ade_m": 0.02888067501354051
  },
  "original_decoded_ade_m": 0.04507109150290489,
  "automatic_checks": {
    "original": {
      "finite": true,
      "normal_forward": true,
      "small_lateral": true,
      "stable_heading": true,
      "stable_speed": true,
      "passed": true
    },
    "decoded": {
      "finite": true,
      "normal_forward": true,
      "small_lateral": true,
      "stable_heading": true,
      "stable_speed": true,
      "passed": true
    },
    "metric": true,
    "passed": true
  }
}
```

度量检查（正定但条件数约 2.46 万，记录原值，不据此声称已验证全局几何）：
```json
{
  "eigenvalues": [
    0.00012335518532693613,
    0.00020934695199641674,
    0.00028588447154895735,
    0.0005801378204319753,
    0.0007521748690595711,
    0.031897495247437326,
    0.14460105520185307,
    3.029106917172818
  ],
  "min_eigenvalue": 0.00012335518532693613,
  "max_eigenvalue": 3.029106917172818,
  "condition_number": 24555.97556880173,
  "symmetric": true,
  "positive_definite": true,
  "contains_nan": false,
  "contains_inf": false,
  "passed": true,
  "lambda": 0.0001,
  "formula": "J_D(z0)^T J_D(z0)/60 + lambda*I"
}
```

输入 SHA256：
```json
{
  "data/ego_trajs.npy": "183ea120cb1dc53367ed19f55d6cd80269b08d849010ab27b639677b28c9427f",
  "weights/trajectory_vae_8d_best.pth": "9918138beece445613918bc2c2e9528df3ef8837844347a47e8f15dff96300a9"
}
```


## 2026-09-09 按 log 划分 construction / held-out

本阶段只补齐逐条来源、划分数据、生成分布图和运行测试。无需重新提取：原文件缺少 log/type/database 字段，但 57,992 个唯一 token 全部可可靠回查，未缺失、未发生跨库歧义。扫描 64 个 DB，其中 52 个贡献现有轨迹。只读关联 lidar_pc → lidar → log，scenario_type 先过滤原 14 类型再取 MAX，与本机 devkit 生成 scenario 的 SQL 一致。额外调用本机官方 get_scenarios_from_db（原类型过滤、remove_invalid_goals 对应参数）独立核验了 57,992 行，52 个来源数据库，类型不一致数为 0。没有重写 trajectory 或 token 数组。

补录保存在 data/ego_trajs_provenance.json，原 metadata 新增其路径、SHA256 和提取类型列表。每行同时包含 source_index、scenario_token、log_name、scenario_type、database_source；长度均为 57,992，行号/token 和输入文件哈希全部通过。提取模块已修复为先取得完整成功记录再追加，失败不造成索引偏移。

正式命令：
```bash
python -m skill_pool.cli recover-provenance --trajectories data/ego_trajs.npy --database-root ../nuplan/dataset/nuplan-v1.1_mini/data/cache/mini
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/neutral-mpl python -m skill_pool.cli split-logs --trajectories data/ego_trajs.npy --provenance data/ego_trajs_provenance.json --seed 7
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/neutral-mpl python -m pytest -q
```

划分先排序唯一 log，用 numpy default_rng(7) 排列，42/10 个 log；随后进行确定性的整 log 最优改进交换（最多 200 次，无改善即停）。目标为各类两组占比差平方和 + construction 轨迹占比偏离 0.8 的平方 + 缺失主要类型数。不改变 log 数、不拆轨迹，不使用模型指标，不以中性样本位置约束划分。初始纯随机划分总变差距离为 0.22948；最终为 0.02134。优化分布所用的是标签元信息，不是 held-out 模型效果。

| 集合 | 轨迹数 | log 数 | 轨迹比例 | log 比例 |
|---|---:|---:|---:|---:|
| construction | 46514 | 42 | 80.2076% | 80.7692% |
| heldout | 11478 | 10 | 19.7924% | 19.2308% |

总计 57,992 条有效轨迹、52 个 log；无效轨迹 0。log 交集、索引交集、跨组 token 交集、重复 token 均为 0；索引范围合法、并集完整覆盖全部有效行。

| 场景类型 | construction | held-out | construction 占比 | held-out 占比 |
|---|---:|---:|---:|---:|
| behind_long_vehicle | 662 | 1 | 1.4232% | 0.0087% |
| changing_lane | 3 | 2 | 0.0064% | 0.0174% |
| following_lane_with_lead | 26 | 0 | 0.0559% | 0.0000% |
| high_lateral_acceleration | 117 | 43 | 0.2515% | 0.3746% |
| high_magnitude_speed | 16071 | 3953 | 34.5509% | 34.4398% |
| low_magnitude_speed | 1668 | 364 | 3.5860% | 3.1713% |
| near_multiple_vehicles | 658 | 177 | 1.4146% | 1.5421% |
| starting_left_turn | 211 | 57 | 0.4536% | 0.4966% |
| starting_right_turn | 100 | 30 | 0.2150% | 0.2614% |
| starting_straight_traffic_light_intersection_traversal | 136 | 37 | 0.2924% | 0.3224% |
| stationary_in_traffic | 16114 | 4076 | 34.6433% | 35.5114% |
| stopping_with_lead | 64 | 0 | 0.1376% | 0.0000% |
| traversing_pickup_dropoff | 10620 | 2692 | 22.8318% | 23.4536% |
| waiting_for_pedestrian_to_cross | 64 | 46 | 0.1376% | 0.4008% |

分布结论：主要场景（全体占比 ≥1%）中，除 behind_long_vehicle 外均通过绝对占比差 ≤10 个百分点且 held-out/construction 占比比值在 [0.5,2] 的检查。其他主要类型的最大占比差为 0.8681 个百分点。

**未通过的分布限制：behind_long_vehicle 为 662/1 条。其 663 条中 653 条（98.49%）集中于同一个 log，另两个 log 仅有 9 和 1 条。任何整 log 的约 80/20 划分，都只能把至多 10 条或至少 653 条分给 held-out，不可能接近理想约 133 条。** 不拆 log、不掩盖失败、不降低阈值；JSON 的 scenario_distribution_check.passed=false，nonconcentrated_major_types_passed=true。新增测试断言该失衡及其单 log 原因正确报告，不能把测试通过解读成所有主要类型分布都通过。少量稀有类型也可能在一组缺失，完整数量见上表。

最大 log 为 2021.06.14.19.22.11_veh-38_01480_01860，3,457 条，占总量 5.96%，占 construction 7.43%；held-out 最大 log 为 2,855 条，约占 held-out 24.87%。无单 log 主导总体，但 held-out 内各 log 权重并不相等。代理已打开并目视检查两张图，图表与计数一致。

产物：
- splits/construction_indices.npy、heldout_indices.npy（原始数组行号）
- splits/construction_logs.txt、heldout_logs.txt
- [split_report.json](splits/split_report.json)
- [scenario_distribution.png](outputs/split/scenario_distribution.png)
- [log_trajectory_counts.png](outputs/split/log_trajectory_counts.png)

固定中性：8052 所在 log 进入 construction；neutral_source_index.npy、z0.npy、neutral_target.npy、G0.npy 未修改，SHA256 记录于 outputs/neutral/baseline_config.json 并在运行及测试中校验。baseline 选点和参考速度此前使用了完整数据，所以本次 held-out 不应被描述为从未参与中性 baseline 配置。未进行新的 Pool 构建（全部测试中原有合成 Pool 测试只写 pytest 临时目录）、未实现严格 geodesic Log、未接入 PufferDrive。

全部测试：
```text
......................                                                   [100%]
22 passed in 6.87s
```
测试覆盖整 log 隔离、索引隔离/合法/全覆盖、相同 seed 可复现、输入行重排下 log 分组稳定、长度及行 token 对齐、多标签恢复、歧义和缺失拒绝、提取失败对齐保护、分布异常检测及上述真实集中限制、真实划分和 PNG 文件、固定 baseline 哈希。


## 2026-09-09 Construction-only V1 baseline 与 held-out 评估

基于提交 63ccd91。只使用 construction 的 46,514 条（42 logs）拟合中性配置、方向、强度和技能选择；held-out 的 11,478 条（10 logs）只用于独立 coverage 评估和展示样例。没有改变 split，没有调整或重新训练 VAE，没有实现严格 geodesic Log，没有接入 PufferDrive。

### 中性验收先行

原 outputs/neutral 完整复制为 outputs/neutral_full_baseline，逐文件哈希一致且旧目录不变。新的 outputs/neutral_construction 从 construction_indices 开始重新统计、筛选及解码排序，并保存 selection_indices.npy。原全数据结果只用于建库无关的中性对比，不参与新的估计或选择。

- 新旧来源索引均为 8052；这是重新筛选的结果，不是强制沿用。
- 全数据参考速度 9.7632875138 m/s；construction-only 为 9.7558236226 m/s，变化 -0.0074638912 m/s。
- 新旧 z0 欧氏距离为 0，D(z0) 间 ADE 为 0；G0 全部特征值相同。
- 新解码对新标准轨迹 ADE 为 0.0277804 m；最大/最终横移 0.0588535 m，速度标准差 0.1871375 m/s，航向范围 2.98962°。
- 自动检查通过。代理在建库之前通过图像工具打开并目视检查 neutral_check.png 和 neutral_candidates.png：仍为直线近似匀速，有小幅逐帧抖动，无明显转弯、横移或持续加减速。结果和文件哈希保存在 visual_review.json，不声称用户本人已审阅。
- build-v1 在编码或选择前验证自动/目视检查、construction 精确索引、来源和产物哈希，失败则停止。

完整新旧对比：[neutral_comparison.json](outputs/neutral_construction/neutral_comparison.json)。新旧 G0 特征值：
```json
[
  0.00012335518532693613,
  0.00020934695199641674,
  0.00028588447154895735,
  0.0005801378204319753,
  0.0007521748690595711,
  0.031897495247437326,
  0.14460105520185307,
  3.029106917172818
]
```

### 执行命令与数据流

```bash
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/neutral-mpl python -m skill_pool.cli neutral-check --trajectories data/ego_trajs.npy --checkpoint weights/trajectory_vae_8d_best.pth --construction-indices splits/construction_indices.npy --output-dir outputs/neutral_construction
# 检查图像并写入 visual_review.json 后才执行：
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/neutral-mpl python -m skill_pool.cli build-v1 --trajectories data/ego_trajs.npy --checkpoint weights/trajectory_vae_8d_best.pth --provenance data/ego_trajs_provenance.json --seed 7
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/neutral-mpl python -m pytest -q
```

中性对比调用 skill_pool.v1.compare_neutrals('outputs/neutral_full_baseline', 'outputs/neutral_construction')。图像补全后仅用已保存 coverage 和 pool 结果重绘，未根据 held-out 指标重新选择技能。

encoder 以冻结、eval 的均值编码全部有效行。all_latents.npy 为 (57992,8)，其行号映射保存于 all_latent_source_indices.npy；construction_latents.npy 为 (46514,8)，heldout_latents.npy 为 (11478,8)，分别配套 construction_indices.npy 与 heldout_indices.npy。

Pool 选择器只接收 construction_latents 和 construction_indices。保留 v_i≈z_i−z0 和 G0=JᵀJ/60+1e-4 I，0.995 强度裁剪、spherical FPS 和四档 [0.25,0.4666667,0.6833333,0.90] 强度分位数都只在 construction 计算。两组 Pool 选择完成后才开始 held-out 评估。source_indices 始终是原始 57,992 行的索引，.npz/.pt 同时保存 source_tokens、source_trajectories 和 selection_split=construction；来源全部通过 construction membership 且与 held-out 无交集。

### 总体结果（米）

| K | 评估集合 | n | minADE | minFDE | FDE-at-minADE | 使用技能数 |
|---|---|---:|---:|---:|---:|---:|
| 32 | construction | 46514 | 0.578212 | 1.168736 | 1.319720 | 32/32 |
| 32 | heldout | 11478 | 0.636949 | 1.266348 | 1.401052 | 29/32 |
| 64 | construction | 46514 | 0.437892 | 0.910570 | 1.043679 | 64/64 |
| 64 | heldout | 11478 | 0.468211 | 0.965268 | 1.081185 | 58/64 |

minFDE 独立按终点最小距离计算，FDE-at-minADE 为 ADE 最近技能的终点距离。近邻按完整 30 点轨迹 ADE，平分时选择较小 skill ID。Pairwise ADE 是同一 Pool 属性，construction 和 held-out 共享相同数值：

| K | 最小 pairwise ADE | 平均 pairwise ADE | p01 | p05 | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|
| 32 | 1.3595215e-05 | 5.916738 | 0.086973 | 0.205404 | 5.534096 | 15.211849 |
| 64 | 2.2281691e-05 | 4.484090 | 0.109250 | 0.206041 | 2.074825 | 15.049673 |

逐技能分配次数（按 skill ID 0…K−1）：
```json
{
  "32": {
    "construction": [
      2357,
      1877,
      2977,
      2425,
      262,
      56,
      20627,
      3546,
      178,
      162,
      115,
      11,
      385,
      122,
      106,
      890,
      204,
      41,
      231,
      2209,
      287,
      81,
      99,
      3026,
      147,
      54,
      659,
      1914,
      71,
      893,
      482,
      20
    ],
    "heldout": [
      841,
      329,
      284,
      1082,
      106,
      18,
      5096,
      1035,
      37,
      25,
      12,
      4,
      103,
      25,
      13,
      212,
      34,
      0,
      34,
      520,
      111,
      0,
      0,
      691,
      36,
      9,
      145,
      423,
      10,
      164,
      75,
      4
    ]
  },
  "64": {
    "construction": [
      2254,
      2485,
      2351,
      2201,
      147,
      18334,
      2549,
      2683,
      123,
      123,
      18,
      51,
      66,
      29,
      22,
      138,
      72,
      20,
      27,
      13,
      119,
      65,
      31,
      52,
      75,
      91,
      232,
      623,
      58,
      26,
      46,
      91,
      43,
      27,
      370,
      22,
      88,
      271,
      110,
      34,
      18,
      104,
      15,
      1362,
      129,
      1203,
      958,
      1361,
      78,
      38,
      141,
      161,
      110,
      85,
      436,
      766,
      1375,
      91,
      516,
      530,
      45,
      788,
      11,
      13
    ],
    "heldout": [
      759,
      346,
      247,
      1072,
      71,
      4243,
      937,
      686,
      27,
      11,
      4,
      5,
      24,
      19,
      2,
      29,
      0,
      8,
      0,
      0,
      14,
      7,
      15,
      2,
      16,
      16,
      35,
      148,
      7,
      3,
      12,
      18,
      21,
      20,
      63,
      3,
      16,
      39,
      25,
      6,
      2,
      5,
      0,
      230,
      0,
      89,
      372,
      430,
      12,
      7,
      34,
      90,
      22,
      43,
      183,
      205,
      231,
      0,
      122,
      289,
      4,
      116,
      6,
      10
    ]
  }
}
```

### 场景覆盖

下表列出各类数量及 minADE；完整逐类型 minFDE、FDE-at-minADE、使用数、分配次数与解释见 [report.json](outputs/v1/report.json)。0 样本指标为 null，不伪造为 0。

| 类型 | construction n | held-out n | K32 C ADE | K32 H ADE | K64 C ADE | K64 H ADE |
|---|---:|---:|---:|---:|---:|---:|
| behind_long_vehicle | 662 | 1 | 0.162509 | 0.253575 | 0.170108 | 0.192673 |
| changing_lane | 3 | 2 | 1.630468 | 2.671831 | 1.216135 | 1.587619 |
| following_lane_with_lead | 26 | 0 | 1.025782 | null | 0.898666 | null |
| high_lateral_acceleration | 117 | 43 | 1.355134 | 1.185401 | 0.936691 | 0.894794 |
| high_magnitude_speed | 16071 | 3953 | 0.543601 | 0.615310 | 0.421492 | 0.523229 |
| low_magnitude_speed | 1668 | 364 | 0.630949 | 0.509923 | 0.572743 | 0.468334 |
| near_multiple_vehicles | 658 | 177 | 1.249784 | 2.365741 | 0.983195 | 1.936035 |
| starting_left_turn | 211 | 57 | 1.175866 | 1.314800 | 1.064146 | 1.114644 |
| starting_right_turn | 100 | 30 | 1.273883 | 1.423025 | 1.049622 | 1.373210 |
| starting_straight_traffic_light_intersection_traversal | 136 | 37 | 1.328067 | 1.481346 | 1.051975 | 1.127983 |
| stationary_in_traffic | 16114 | 4076 | 0.073364 | 0.088140 | 0.069212 | 0.076026 |
| stopping_with_lead | 64 | 0 | 0.943072 | null | 0.845951 | null |
| traversing_pickup_dropoff | 10620 | 2692 | 1.335259 | 1.354938 | 0.950441 | 0.840109 |
| waiting_for_pedestrian_to_cross | 64 | 46 | 0.111850 | 0.836127 | 0.111590 | 0.714914 |

behind_long_vehicle 的 held-out 只有 1 条，following_lane_with_lead 和 stopping_with_lead 为 0 条，不作泛化结论。所有类型指标只作描述性 coverage；<30 条额外标记样本不足，≥30 条也不自动构成统计泛化证明。本阶段 held-out 不参与新的中性和 Pool 拟合；冻结 VAE 的原训练日志是否与 held-out 独立尚未证实，因此这里明确评估的是 Pool 构建的 held-out coverage。

### 近重复、近静止与图像检查

未施加任何最终去重阈值或删除技能。完整无序技能对（K32:496 对，K64:2016 对）按 ADE 升序保存在 duplicate_diagnostics.json；列出最接近的 10 对并全部绘制于 duplicate_skill_pairs.png。阈值扫描只是诊断，不参与选择。

| ADE 阈值（m） | K32 对数 | K64 对数 |
|---:|---:|---:|
| 1e-06 | 0 | 0 |
| 0.001 | 3 | 1 |
| 0.01 | 3 | 3 |
| 0.05 | 3 | 4 |
| 0.1 | 6 | 13 |
| 0.25 | 33 | 150 |
| 0.5 | 102 | 556 |

解码严格静止计数（最大距原点半径≤1e-6 m）均为 0。近静止诊断同时要求平均逐帧速度≤t m/s、半径≤3t m：t=0.05/0.1 时均为 0，t=0.25/0.5 时均为 3 个（skill 4、5、6）。这是候选阈值敏感性，不是最终停车定义。静止附近的 VAE 厘米级抖动会使逐帧速度高于实际净位移速度。

代理已打开并目视检查全部 8 张 V1 图：使用次数明显集中于少数近静止及常见行驶技能，技能来源不同也仍有几乎重合的解码轨迹。nearest_skill_examples 来自 held-out ADE 全排序的最好/中位/最差。K32 最差 row 35772（changing_lane）ADE 3.650 m，明显右向曲线未被其最近技能充分覆盖；K64 最差 row 43856（high_magnitude_speed）ADE 3.010 m，横向运动覆盖仍有限。最好样例 row 40016 接近静止，ADE 约 0.012 m，放大图中的锯齿是厘米级解码抖动。未根据这些 held-out 结果调整 Pool。

### 产物与验证

- outputs/neutral_full_baseline/：完整原全数据中性归档。
- outputs/neutral_construction/：neutral_check.png、neutral_candidates.png、neutral_check.json、metric_check.json、z0.npy、G0.npy、neutral_target.npy、来源及选取索引、neutral_comparison.json、visual_review.json。
- outputs/v1/：all/construction/heldout latents 及索引映射，skill_pool_32/64.npz 与 .pt，coverage_construction/heldout_32/64.npz，report.json、duplicate_diagnostics.json。
- 图：[skill_pool_32.png](outputs/v1/skill_pool_32.png)、[skill_pool_64.png](outputs/v1/skill_pool_64.png)、[skill_usage_32.png](outputs/v1/skill_usage_32.png)、[skill_usage_64.png](outputs/v1/skill_usage_64.png)、[nearest_skill_examples_32.png](outputs/v1/nearest_skill_examples_32.png)、[nearest_skill_examples_64.png](outputs/v1/nearest_skill_examples_64.png)、[duplicate_skill_pairs.png](outputs/v1/duplicate_skill_pairs.png)、[coverage_comparison.png](outputs/v1/coverage_comparison.png)。

全部测试：
```text
................................                                         [100%]
32 passed in 12.28s
```
新增验收覆盖：中性统计忽略 held-out（改变 held-out 速度后结果不变）；自动/目视失败阻止建库；非法 split 拒绝；选择器只收到 construction；同 seed 改变 held-out 轨迹 1000 倍后方向、强度、来源及 latent 完全不变；K32/64 形状、.pt/.npz 一致性；原始 source 回指、完整/分组编码映射；分组 coverage 独立计算和 minFDE/FDE-at-minADE 区分；空场景保留 null；合成重复轨迹识别；全部正式产物与原中性归档哈希。

输入 checkpoint SHA256：9918138beece445613918bc2c2e9528df3ef8837844347a47e8f15dff96300a9。数据 SHA256：183ea120cb1dc53367ed19f55d6cd80269b08d849010ab27b639677b28c9427f。与此前 baseline 输入一致。
