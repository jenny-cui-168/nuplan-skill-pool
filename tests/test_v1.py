import json
from pathlib import Path
import numpy as np
import pytest
import torch
from skill_pool.model import TrajectoryVAE, load_vae
from skill_pool.data import encode_trajectories
from skill_pool.geometry import decoder_metric
from skill_pool.neutral import neutral_target, run_neutral_check
from skill_pool.splits import sha256, write_json
from skill_pool.v1 import (pairwise_diagnostics, evaluate_split, validate_partition,
                           select_construction_pool, verify_neutral, run_v1)

ROOT = Path(__file__).resolve().parents[1]


def test_duplicate_diagnostic_identifies_exact_and_near_duplicates():
    straight = neutral_target(5)
    near = straight.copy(); near[:, 0] = .001
    report = pairwise_diagnostics(np.array([straight, straight, near, neutral_target(10)]))
    assert report['pair_count'] == 6
    assert report['near_duplicate_pairs'][0] == dict(skill_i=0, skill_j=1, ade_m=0.0)
    assert report['threshold_sensitivity'][0]['count'] == 1
    assert report['min_ade_m'] == 0
    assert len(report['all_pairs_sorted']) == 6


def test_separate_metrics_and_zero_sample_types():
    pool = np.zeros((2, 30, 2))
    pool[1, :, 0] = 10; pool[1, -1, 0] = 1
    construction = np.zeros((2, 30, 2))
    heldout = np.zeros((1, 30, 2)); heldout[0, -1, 0] = 1
    labels = ['a', 'absent']
    c, _ = evaluate_split(construction, pool, np.array(['a', 'a']), labels)
    h, samples = evaluate_split(heldout, pool, np.array(['a']), labels)
    assert c['minADE_m'] == 0 and h['minADE_m'] > 0
    assert h['minFDE_m'] == 0 and h['FDE_at_minADE_m'] == 1
    assert h['assignment_counts'] == [1, 0]
    assert h['by_scenario_type']['absent']['trajectory_count'] == 0
    assert h['by_scenario_type']['absent']['minADE_m'] is None
    assert samples['nearest_skill'].tolist() == [0]


@pytest.mark.parametrize('c,h', [(np.array([0, 1]), np.array([1, 2])), (np.array([0]), np.array([2])), (np.array([-1, 0]), np.array([1, 2]))])
def test_invalid_partition_rejected(c, h):
    with pytest.raises(ValueError):
        validate_partition(c, h, np.arange(3), 3, np.array(['a', 'b', 'c']))


def test_neutral_ignores_heldout_statistics(tmp_path):
    model = TrajectoryVAE()
    with torch.no_grad():
        for p in model.parameters(): p.zero_()
        model.decoder[-1].bias.copy_(torch.tensor(neutral_target(7).flatten(), dtype=torch.float32))
    checkpoint = tmp_path/'vae.pth'; torch.save(model.state_dict(), checkpoint)
    path = tmp_path/'trajs.npy'
    raw = np.stack([neutral_target(7)]*12 + [neutral_target(20)]*30).astype(np.float32)
    np.save(path, raw); np.save(tmp_path/'trajs_tokens.npy', np.array([str(i) for i in range(len(raw))]))
    c = np.arange(12)
    first = run_neutral_check(path, checkpoint, tmp_path/'first', selection_indices=c)
    raw[12:] = neutral_target(2); np.save(path, raw)
    second = run_neutral_check(path, checkpoint, tmp_path/'second', selection_indices=c)
    assert first['reference_speed_mps'] == second['reference_speed_mps'] == 7
    assert first['neutral_source_index'] == second['neutral_source_index'] == 0
    assert first['normal_motion_count'] == first['selection_count'] == 12
    assert first['top_candidates'] == second['top_candidates']
    for name in ['z0.npy', 'G0.npy', 'neutral_target.npy']:
        np.testing.assert_array_equal(np.load(tmp_path/'first'/name), np.load(tmp_path/'second'/name))


@pytest.mark.parametrize("auto,visual", [(False, True), (True, False)])
def test_failed_neutral_gate_stops(tmp_path, auto, visual):
    write_json(tmp_path/'neutral_check.json', {'automatic_checks': {'passed': auto}})
    write_json(tmp_path/'visual_review.json', {'passed': visual})
    with pytest.raises(ValueError, match='stop before'):
        verify_neutral(tmp_path, np.arange(10), np.zeros((10, 30, 2)), 'unused')


def test_pipeline_passes_only_construction_to_selector_and_reproduces(tmp_path, monkeypatch):
    import skill_pool.v1 as v1
    torch.manual_seed(7)
    checkpoint = tmp_path/'vae.pth'; torch.save(TrajectoryVAE().state_dict(), checkpoint)
    model = load_vae(checkpoint)
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(240, 30, 2)).astype(np.float32)
    h = np.arange(5, 240, 6)
    c = np.array([i for i in range(240) if i % 6 != 5])
    split = tmp_path/'splits'; split.mkdir()
    np.save(split/'construction_indices.npy', c); np.save(split/'heldout_indices.npy', h)
    neutral = tmp_path/'neutral'; neutral.mkdir()
    path = tmp_path/'raw.npy'; np.save(path, raw)
    z0 = encode_trajectories(model, raw[c[:1]], 1024, 'cpu')[0][0]
    metric = decoder_metric(model, z0, 'cpu')
    monkeypatch.setattr(v1, 'verify_neutral', lambda *args: (z0, metric, int(c[0])))
    tokens = np.array([str(i) for i in range(240)])
    logs = np.where(np.isin(np.arange(240), c), 'construction-log', 'heldout-log')
    monkeypatch.setattr(v1, 'aligned_arrays', lambda *args: (raw, tokens, logs, np.array(['type']*240)))
    monkeypatch.setattr(v1, 'plot_pool', lambda *args: None)
    monkeypatch.setattr(v1, 'plot_v1_evaluation', lambda *args: None)
    calls = []
    actual_selector = v1.select_direction_strength_pool
    def spy(latents, *args, **kwargs):
        expected = encode_trajectories(model, raw[c], 1024, 'cpu')[0]
        np.testing.assert_array_equal(latents, expected)
        calls.append(len(latents))
        return actual_selector(latents, *args, **kwargs)
    monkeypatch.setattr(v1, 'select_direction_strength_pool', spy)
    for number in [1, 2]:
        if number == 2:
            raw[h] *= 1000  # Held-out counterfactual cannot affect directions or strengths.
            np.save(path, raw)
        write_json(neutral/'neutral_check.json', {'trajectory_sha256': sha256(path)})
        run_v1(path, checkpoint, path, split, neutral, tmp_path/f'out{number}', seed=7)
    assert calls == [200, 200, 200, 200]
    for k in [32, 64]:
        one = np.load(tmp_path/f'out1/skill_pool_{k}.npz')
        two = np.load(tmp_path/f'out2/skill_pool_{k}.npz')
        assert one['latents'].shape == (k, 8) and one['trajectories'].shape == (k, 30, 2)
        assert np.isin(one['source_indices'], c).all()
        assert not np.isin(one['source_indices'], h).any()
        for name in ['source_indices', 'latents', 'direction_ids', 'strengths']:
            np.testing.assert_array_equal(one[name], two[name])
        np.testing.assert_array_equal(one['source_trajectories'], raw[one['source_indices']])


def test_formal_v1_artifacts_and_mapping():
    out = ROOT/'outputs/v1'; neutral = ROOT/'outputs/neutral_construction'
    construction = np.load(ROOT/'splits/construction_indices.npy'); heldout = np.load(ROOT/'splits/heldout_indices.npy')
    check = json.loads((neutral/'neutral_check.json').read_text())
    assert check['selection_count'] == 46514 and check['neutral_source_index'] in construction
    np.testing.assert_array_equal(np.load(neutral/'selection_indices.npy'), construction)
    report = json.loads((out/'report.json').read_text())
    all_latents = np.load(out/'all_latents.npy')
    all_sources = np.load(out/'all_latent_source_indices.npy')
    for split, indices in [('construction', construction), ('heldout', heldout)]:
        np.testing.assert_array_equal(np.load(out/f'{split}_indices.npy'), indices)
        np.testing.assert_array_equal(np.load(out/f'{split}_latents.npy'), all_latents[np.searchsorted(all_sources, indices)])
    c_latents = np.load(out/'construction_latents.npy')
    z0 = np.load(neutral/'z0.npy'); metric = np.load(neutral/'G0.npy')
    for k in [32, 64]:
        pool = np.load(out/f'skill_pool_{k}.npz')
        assert pool['selection_split'].item() == 'construction'
        pt = torch.load(out/f'skill_pool_{k}.pt', weights_only=True)
        assert pt['selection_split'] == 'construction'
        assert tuple(pt['latents'].shape) == (k, 8)
        np.testing.assert_array_equal(pt['source_indices'].numpy(), pool['source_indices'])
        assert pool['latents'].shape == (k, 8) and pool['trajectories'].shape == (k, 30, 2)
        assert np.isin(pool['source_indices'], construction).all()
        assert not np.isin(pool['source_indices'], heldout).any()
        selected, indices, _, _ = select_construction_pool(c_latents, construction, z0, metric, k)
        np.testing.assert_array_equal(indices, pool['source_indices'])
        np.testing.assert_array_equal(selected, pool['latents'])
        for split, expected in [('construction', construction), ('heldout', heldout)]:
            values = np.load(out/f'coverage_{split}_{k}.npz')
            np.testing.assert_array_equal(values['source_indices'], expected)
            metrics = report['pools'][str(k)][split]
            assert metrics['trajectory_count'] == len(expected)
            assert sum(metrics['assignment_counts']) == len(expected)
            np.testing.assert_allclose(metrics['minADE_m'], values['minADE_m'].mean())
        by_type = report['pools'][str(k)]['heldout']['by_scenario_type']
        assert by_type['behind_long_vehicle']['trajectory_count'] == 1
        assert by_type['following_lane_with_lead']['trajectory_count'] == 0
        assert by_type['stopping_with_lead']['trajectory_count'] == 0
        for name in ['skill_pool', 'skill_usage', 'nearest_skill_examples']:
            assert (out/f'{name}_{k}.png').read_bytes().startswith(b'\x89PNG')
        if (ROOT/'data/ego_trajs.npy').exists():
            raw = np.load(ROOT/'data/ego_trajs.npy', mmap_mode='r')
            np.testing.assert_array_equal(pool['source_trajectories'], raw[pool['source_indices']])
    for name in ['duplicate_skill_pairs', 'coverage_comparison']:
        assert (out/f'{name}.png').read_bytes().startswith(b'\x89PNG')
    for path in (ROOT/'outputs/neutral').iterdir():
        if path.is_file(): assert sha256(path) == sha256(ROOT/'outputs/neutral_full_baseline'/path.name)
