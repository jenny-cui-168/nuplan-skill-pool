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
