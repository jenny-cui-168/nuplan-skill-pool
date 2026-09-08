import numpy as np

from skill_pool.evaluation import coverage_metrics


def test_minfde_can_choose_a_different_skill_than_minade():
    target = np.zeros((1, 30, 2), dtype=np.float32)
    pool = np.zeros((2, 30, 2), dtype=np.float32)
    pool[0, -1, 0] = 3
    pool[1, :-1, 0] = 10
    metrics = coverage_metrics(target, pool)
    assert metrics["coverage_minFDE_mean_m"] == 0
    assert metrics["coverage_FDE_at_minADE_mean_m"] == 3
    assert np.isclose(metrics["coverage_minADE_mean_m"], 0.1)
