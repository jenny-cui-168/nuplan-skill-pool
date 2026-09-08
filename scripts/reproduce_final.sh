#!/usr/bin/env bash
set -euo pipefail
cd '/home/jennycui/下载/yanjiusheng/skill-pool-pipeline'
export MPLCONFIGDIR=/tmp/skill-pool-mpl
export OMP_NUM_THREADS=4
mkdir -p logs
skill-pool extract-nuplan --data-root '/home/jennycui/下载/yanjiusheng/nuplan/dataset/nuplan-v1.1_mini/data/cache/mini' --map-root '/home/jennycui/下载/yanjiusheng/nuplan/dataset/maps1/nuplan-maps-v1.0/maps' --scenarios-per-type 0 --output data/ego_trajs.npy > logs/extract_final.log 2>&1
skill-pool build --trajectories data/ego_trajs.npy --checkpoint weights/trajectory_vae_8d_best.pth --pool-sizes 32 64 --output-dir outputs/final --device cpu > logs/build_final.log 2>&1
