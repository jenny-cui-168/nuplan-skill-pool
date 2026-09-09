from __future__ import annotations

from pathlib import Path
from typing import Iterable
import inspect
import traceback

import numpy as np


DEFAULT_SCENARIO_TYPES = (
    "starting_left_turn",
    "starting_right_turn",
    "starting_straight_traffic_light_intersection_traversal",
    "stopping_with_lead",
    "high_lateral_acceleration",
    "high_magnitude_speed",
    "low_magnitude_speed",
    "traversing_pickup_dropoff",
    "waiting_for_pedestrian_to_cross",
    "behind_long_vehicle",
    "stationary_in_traffic",
    "near_multiple_vehicles",
    "changing_lane",
    "following_lane_with_lead",
)


def _scenario_filter(
    scenario_types: list[str] | None,
    scenarios_per_type: int | None,
    limit_total_scenarios: int | None,
    shuffle: bool,
):
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter

    values = {
        "scenario_types": scenario_types,
        "scenario_tokens": None,
        "log_names": None,
        "map_names": None,
        "num_scenarios_per_type": scenarios_per_type,
        "limit_total_scenarios": limit_total_scenarios,
        "timestamp_threshold_s": None,
        "ego_displacement_minimum_m": None,
        "expand_scenarios": False,
        "remove_invalid_goals": True,
        "shuffle": shuffle,
        "ego_start_speed_threshold": None,
        "ego_stop_speed_threshold": None,
        "speed_noise_tolerance": None,
        "token_set_path": None,
        "fraction_in_token_set_threshold": None,
        "ego_route_radius": None,
    }
    # nuPlan 1.1/1.2 added filter fields. Pass exactly the fields supported by
    # the installed devkit so the extractor works with either version.
    supported = inspect.signature(ScenarioFilter).parameters
    return ScenarioFilter(**{key: value for key, value in values.items() if key in supported})


def _local_future(scenario, samples: int, horizon_seconds: float) -> np.ndarray:
    initial = scenario.initial_ego_state.rear_axle
    rotation = -initial.heading + np.pi / 2.0  # real-Skillformer: +Y forward
    cosine, sine = np.cos(rotation), np.sin(rotation)
    rotation_matrix = np.asarray([[cosine, -sine], [sine, cosine]])
    states: Iterable = scenario.get_ego_future_trajectory(
        iteration=0, num_samples=samples, time_horizon=horizon_seconds
    )
    points = []
    for state in states:
        offset = np.asarray(
            [state.rear_axle.x - initial.x, state.rear_axle.y - initial.y]
        )
        points.append(rotation_matrix @ offset)
    trajectory = np.asarray(points, dtype=np.float32)
    if trajectory.shape != (samples, 2):
        raise ValueError(f"Expected {(samples, 2)}, got {trajectory.shape}")
    if not np.isfinite(trajectory).all():
        raise ValueError("Non-finite future trajectory")
    return trajectory


def _successful_row(scenario, samples, horizon_seconds, source_index):
    """Resolve every field before appending anything, so failed rows cannot shift arrays."""
    trajectory = _local_future(scenario, samples, horizon_seconds)
    row = dict(source_index=source_index, scenario_token=str(scenario.token),
               log_name=scenario.log_name, scenario_type=scenario.scenario_type,
               database_source=str(scenario._log_file_load_path))
    if any(not isinstance(row[key], str) or not row[key].strip() for key in
           ['scenario_token', 'log_name', 'scenario_type', 'database_source']):
        raise ValueError('Missing scenario provenance')
    return trajectory, row


def extract_nuplan_trajectories(
    data_root: str | Path,
    map_root: str | Path,
    output_path: str | Path,
    map_version: str = "nuplan-maps-v1.0",
    samples: int = 30,
    horizon_seconds: float = 3.0,
    scenarios_per_type: int | None = 100,
    limit_total_scenarios: int | None = None,
    all_scenario_types: bool = False,
    shuffle: bool = False,
) -> dict:
    try:
        from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
            NuPlanScenarioBuilder,
        )
        from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
            ScenarioMapping,
        )
        from nuplan.planning.utils.multithreading.worker_parallel import (
            SingleMachineParallelExecutor,
        )
    except ImportError as error:
        raise RuntimeError(
            "nuPlan devkit is not installed. Install the optional nuplan dependencies first."
        ) from error

    data_root = Path(data_root).expanduser().resolve()
    map_root = Path(map_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"nuPlan data root not found: {data_root}")
    if not map_root.exists():
        raise FileNotFoundError(f"nuPlan map root not found: {map_root}")

    builder = NuPlanScenarioBuilder(
        data_root=str(data_root),
        map_root=str(map_root),
        sensor_root=None,
        db_files=None,
        map_version=map_version,
        scenario_mapping=ScenarioMapping(
            scenario_map={
                name: (15.0, -3.0)
                for name in ([] if all_scenario_types else DEFAULT_SCENARIO_TYPES)
            },
            subsample_ratio_override=0.5,
        ),
    )
    scenario_filter = _scenario_filter(
        None if all_scenario_types else list(DEFAULT_SCENARIO_TYPES),
        scenarios_per_type,
        limit_total_scenarios,
        shuffle,
    )
    worker = SingleMachineParallelExecutor(use_process_pool=False)
    scenarios = builder.get_scenarios(scenario_filter, worker)

    trajectories: list[np.ndarray] = []
    tokens: list[str] = []
    provenance_rows: list[dict] = []
    failures: list[dict[str, str]] = []
    for scenario in scenarios:
        try:
            trajectory, row = _successful_row(scenario, samples, horizon_seconds, len(trajectories))
            trajectories.append(trajectory)
            tokens.append(row["scenario_token"])
            provenance_rows.append(row)
        except Exception as error:  # retain an audit trail instead of aborting a long extraction
            failures.append({"token": str(scenario.token), "error": repr(error), "traceback": traceback.format_exc()})
    if not trajectories:
        raise RuntimeError("No valid future trajectories were extracted")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, np.stack(trajectories).astype(np.float32))
    np.save(output.with_name(f"{output.stem}_tokens.npy"), np.asarray(tokens))
    from .splits import sha256, write_json
    provenance_path = output.with_name(f"{output.stem}_provenance.json")
    write_json(provenance_path, {
        "schema_version": 1, "trajectory_sha256": sha256(output),
        "tokens_sha256": sha256(output.with_name(f"{output.stem}_tokens.npy")),
        "recovery_method": "Captured synchronously from successfully extracted scenario",
        "rows": provenance_rows,
    })
    metadata = {
        "row_provenance_file": provenance_path.name,
        "row_provenance_sha256": sha256(provenance_path),
        "scenario_types": None if all_scenario_types else list(DEFAULT_SCENARIO_TYPES),
        "data_root": str(data_root),
        "map_root": str(map_root),
        "map_version": map_version,
        "trajectory_count": len(trajectories),
        "trajectory_shape": [samples, 2],
        "horizon_seconds": horizon_seconds,
        "coordinate_convention": "+Y forward, X lateral, ego rear-axle origin",
        "failure_count": len(failures),
        "scenario_count": len(scenarios),
        "scenarios_per_type": scenarios_per_type,
        "all_scenario_types": all_scenario_types,
        "failures": failures,
    }
    output.with_name(f"{output.stem}_metadata.json").write_text(
        __import__("json").dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata
