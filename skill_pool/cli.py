from __future__ import annotations

import argparse
import json
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skill-pool",
        description="Build a 32/64-entry non-clustered skill pool from nuPlan trajectories.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract-nuplan", help="Extract [N,30,2] ego trajectories")
    extract.add_argument("--data-root", required=True)
    extract.add_argument("--map-root", required=True)
    extract.add_argument("--output", default="data/ego_trajs.npy")
    extract.add_argument("--map-version", default="nuplan-maps-v1.0")
    extract.add_argument("--scenarios-per-type", type=int, default=100, help="Maximum per type; 0 disables the limit")
    extract.add_argument("--limit-total-scenarios", type=int)
    extract.add_argument("--all-scenario-types", action="store_true")
    extract.add_argument("--shuffle", action="store_true")

    build = subparsers.add_parser("build", help="Encode trajectories and build skill pools")
    build.add_argument("--trajectories", required=True)
    build.add_argument("--checkpoint", required=True)
    build.add_argument("--output-dir", default="outputs")
    build.add_argument("--pool-sizes", type=int, nargs="+", default=[32, 64])
    build.add_argument("--strength-levels", type=int, default=4)
    build.add_argument("--batch-size", type=int, default=1024)
    build.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    build.add_argument("--seed", type=int, default=7)
    neutral = subparsers.add_parser("neutral-check", help="Validate neutral skill without building pools")
    neutral.add_argument("--trajectories", required=True)
    neutral.add_argument("--checkpoint", required=True)
    neutral.add_argument("--tokens")
    neutral.add_argument("--construction-indices")
    neutral.add_argument("--output-dir", default="outputs/neutral")
    neutral.add_argument("--dt", type=float, default=0.1)
    neutral.add_argument("--damping", type=float, default=1e-4)
    neutral.add_argument("--batch-size", type=int, default=1024)
    recover = subparsers.add_parser("recover-provenance", help="Recover row metadata from existing tokens and databases")
    recover.add_argument("--trajectories", required=True)
    recover.add_argument("--database-root", required=True)
    split = subparsers.add_parser("split-logs", help="Split whole logs without building pools")
    split.add_argument("--trajectories", required=True)
    split.add_argument("--provenance", required=True)
    split.add_argument("--output-dir", default="splits")
    split.add_argument("--plot-dir", default="outputs/split")
    split.add_argument("--seed", type=int, default=7)
    split.add_argument("--baseline-dir", default="outputs/neutral")
    v1 = subparsers.add_parser("build-v1", help="Construction-only pools with held-out evaluation after reviewed neutral gate")
    v1.add_argument("--trajectories", required=True)
    v1.add_argument("--checkpoint", required=True)
    v1.add_argument("--provenance", required=True)
    v1.add_argument("--split-dir", default="splits")
    v1.add_argument("--neutral-dir", default="outputs/neutral_construction")
    v1.add_argument("--output-dir", default="outputs/v1")
    v1.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "extract-nuplan":
        from .nuplan_extract import extract_nuplan_trajectories

        result = extract_nuplan_trajectories(
            data_root=args.data_root,
            map_root=args.map_root,
            output_path=args.output,
            map_version=args.map_version,
            scenarios_per_type=args.scenarios_per_type or None,
            limit_total_scenarios=args.limit_total_scenarios,
            all_scenario_types=args.all_scenario_types,
            shuffle=args.shuffle,
        )
    elif args.command == "build-v1":
        from .v1 import run_v1
        result = run_v1(args.trajectories, args.checkpoint, args.provenance, args.split_dir, args.neutral_dir, args.output_dir, args.seed)
    elif args.command == "recover-provenance":
        from .splits import recover_provenance
        provenance = recover_provenance(args.trajectories, args.database_root)
        result = {"recovered_rows": len(provenance["rows"]), "method": provenance["recovery_method"]}
    elif args.command == "split-logs":
        from .splits import run_split
        result = run_split(args.trajectories, args.provenance, args.output_dir, args.plot_dir, args.seed, args.baseline_dir)
    elif args.command == "neutral-check":
        from .neutral import run_neutral_check
        import numpy as np
        indices = np.load(args.construction_indices, allow_pickle=False) if args.construction_indices else None
        result = run_neutral_check(args.trajectories, args.checkpoint, args.output_dir, args.tokens, args.dt, args.damping, args.batch_size, indices)
    else:
        from .pipeline import run_pipeline

        result = run_pipeline(
            trajectories_path=args.trajectories,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            pool_sizes=tuple(args.pool_sizes),
            strength_levels=args.strength_levels,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
