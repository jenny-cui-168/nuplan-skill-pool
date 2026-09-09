import json
import numpy as np
import pytest
import torch
from skill_pool.neutral import neutral_target, curves, neutral_checks, metric_diagnostics, run_neutral_check
from skill_pool.geometry import decoder_metric
from skill_pool.model import TrajectoryVAE


def test_target_and_speed():
    t = neutral_target(7)
    np.testing.assert_array_equal(t[:, 0], 0)
    np.testing.assert_allclose(t[:, 1], 7 * np.arange(1, 31) / 10)
    np.testing.assert_allclose(curves(t)[0], 7)
    assert neutral_checks(t)['passed']


@pytest.mark.parametrize('kind', ['turn', 'lateral', 'accelerate', 'decelerate', 'reverse', 'nan'])
def test_reject_non_neutral(kind):
    time = np.arange(1, 31)/10
    traj = neutral_target(7)
    if kind == 'turn':
        traj[:, 0] = -10*(1-np.cos(time/3))
        traj[:, 1] = 10*np.sin(time/3)
    elif kind == 'lateral':
        traj[:, 0] = time*2
    elif kind == 'accelerate':
        traj[:, 1] = 2*time + time**2
    elif kind == 'decelerate':
        traj[:, 1] = 10*time-time**2
    elif kind == 'reverse':
        traj[:, 1] *= -1
    else:
        traj[3, 0] = np.nan
    assert not neutral_checks(traj)['passed']


def test_metric():
    torch.manual_seed(3)
    model = TrajectoryVAE().eval()
    metric = decoder_metric(model, np.zeros(8), 'cpu')
    assert metric_diagnostics(metric)['passed']
    np.testing.assert_allclose(metric, metric.T, atol=1e-8)
    assert np.linalg.eigvalsh(metric).min() > 0
    assert not metric_diagnostics(np.full((8, 8), np.nan))['passed']


def test_artifacts_and_source_mapping(tmp_path):
    model = TrajectoryVAE()
    # Deterministic decoder fixture isolates file generation and source indexing.
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.decoder[-1].bias.copy_(torch.tensor(neutral_target(7).flatten(), dtype=torch.float32))
    torch.save(model.state_dict(), tmp_path/'vae.pth')
    trajs = np.stack([np.full((30, 2), np.nan)] + [neutral_target(7)]*10).astype(np.float32)
    np.save(tmp_path/'trajs.npy', trajs)
    np.save(tmp_path/'trajs_tokens.npy', np.array([f'token-{i}' for i in range(11)]))
    result = run_neutral_check(tmp_path/'trajs.npy', tmp_path/'vae.pth', tmp_path/'out')
    assert result['neutral_source_index'] == 1
    assert result['neutral_source_token'] == 'token-1'
    assert result['automatic_checks']['passed']
    assert len(result['top_candidates']) == 10
    for name in ['neutral_check.png', 'neutral_candidates.png']:
        assert (tmp_path/'out'/name).read_bytes().startswith(b'\x89PNG')
    for name in ['neutral_check.json', 'metric_check.json']:
        assert json.loads((tmp_path/'out'/name).read_text())
    np.testing.assert_array_equal(np.load(tmp_path/'out/neutral_source_trajectory.npy'), trajs[1])
    assert not list((tmp_path/'out').glob('skill_pool*'))
