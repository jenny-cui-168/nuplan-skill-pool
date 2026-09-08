import numpy as np

from skill_pool.geometry import (
    choose_neutral_index,
    metric_norm,
    normalize_directions,
    spherical_fps,
)
from skill_pool.nuplan_extract import _local_future


def test_metric_normalization_produces_unit_directions():
    metric = np.diag([1.0, 4.0])
    vectors = np.asarray([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    directions, strengths, valid = normalize_directions(vectors, metric)
    assert valid.all()
    assert np.all(strengths > 0)
    np.testing.assert_allclose(metric_norm(directions, metric), 1.0)


def test_spherical_fps_spreads_cardinal_directions():
    directions = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    selected = spherical_fps(directions, np.eye(2), 4)
    assert len(set(selected.tolist())) == 4


def test_neutral_prefers_typical_straight_motion():
    time = np.linspace(0, 3, 30)
    trajectories = []
    for lateral, speed in [(0.0, 4.0), (0.1, 4.1), (5.0, 4.0), (0.0, 12.0)]:
        trajectories.append(np.stack([lateral * time / 3, speed * time], axis=-1))
    index = choose_neutral_index(np.asarray(trajectories, dtype=np.float32))
    assert index in (0, 1)


def test_nuplan_transform_places_heading_direction_on_positive_y():
    class RearAxle:
        def __init__(self, x, y, heading=0.0):
            self.x, self.y, self.heading = x, y, heading

    class State:
        def __init__(self, x, y, heading=0.0):
            self.rear_axle = RearAxle(x, y, heading)

    class Scenario:
        initial_ego_state = State(10.0, 20.0, 0.0)

        @staticmethod
        def get_ego_future_trajectory(iteration, num_samples, time_horizon):
            assert iteration == 0 and num_samples == 3 and time_horizon == 0.3
            return [State(11.0, 20.0), State(12.0, 20.0), State(13.0, 20.0)]

    trajectory = _local_future(Scenario(), samples=3, horizon_seconds=0.3)
    np.testing.assert_allclose(trajectory[:, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(trajectory[:, 1], [1.0, 2.0, 3.0], atol=1e-6)
