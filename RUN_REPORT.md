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
