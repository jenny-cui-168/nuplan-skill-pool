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
