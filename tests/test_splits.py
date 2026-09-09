import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from skill_pool.splits import (split_by_log, distribution, recover_provenance,
                               aligned_arrays, sha256, write_json)
from skill_pool.nuplan_extract import _successful_row


def assert_partition(logs, valid, construction, heldout):
    assert not set(logs[construction]) & set(logs[heldout])
    assert not set(construction) & set(heldout)
    np.testing.assert_array_equal(np.union1d(construction, heldout), np.flatnonzero(valid))
    for indices in [construction, heldout]:
        assert np.all((indices >= 0) & (indices < len(logs)))
    for log in set(logs[valid]):
        assert (log in logs[construction]) != (log in logs[heldout])


def test_log_partition_reproducible_and_order_independent():
    logs = np.repeat([f'log-{i:02}' for i in range(20)], 20)
    types = np.tile(['straight', 'turn'], 200)
    valid = np.ones(len(logs), dtype=bool); valid[0] = False
    c, h = split_by_log(logs, valid, scenario_types=types)
    assert_partition(logs, valid, c, h)
    c2, h2 = split_by_log(logs, valid, scenario_types=types)
    np.testing.assert_array_equal(c, c2); np.testing.assert_array_equal(h, h2)
    reverse = np.arange(len(logs))[::-1]
    cr, hr = split_by_log(logs[reverse], valid[reverse], scenario_types=types[reverse])
    assert set(logs[c]) == set(logs[reverse][cr])
    assert distribution(types, c, h)[1]['passed']


def test_distribution_detects_missing_major_class():
    _, checks = distribution(np.array(['a']*80+['b']*20), np.arange(80), np.arange(80, 100))
    assert not checks['passed']
    assert checks['missing_major_types'] == ['a', 'b']


def make_recovery_fixture(tmp_path):
    path = tmp_path/'trajs.npy'
    np.save(path, np.zeros((2, 30, 2), dtype=np.float32))
    np.save(tmp_path/'trajs_tokens.npy', np.array(['02', '01']))
    write_json(tmp_path/'trajs_metadata.json', dict(trajectory_count=2, all_scenario_types=False,
               scenario_types=['low_magnitude_speed', 'stationary_in_traffic']))
    db = tmp_path/'log-a.db'
    with sqlite3.connect(db) as conn:
        conn.executescript('''CREATE TABLE log (token BLOB, logfile TEXT);
        CREATE TABLE lidar (token BLOB, log_token BLOB);
        CREATE TABLE lidar_pc (token BLOB, lidar_token BLOB);
        CREATE TABLE scenario_tag (lidar_pc_token BLOB, type TEXT);
        INSERT INTO log VALUES (x'aa', 'log-a');
        INSERT INTO lidar VALUES (x'bb', x'aa');
        INSERT INTO lidar_pc VALUES (x'01', x'bb'), (x'02', x'bb');
        INSERT INTO scenario_tag VALUES (x'01', 'stationary_in_traffic'),
        (x'01', 'low_magnitude_speed'), (x'01', 'zzz_not_selected'), (x'02', 'low_magnitude_speed');''')
    return path, db


def test_recovery_alignment_and_filtered_multilabel(tmp_path):
    path, _ = make_recovery_fixture(tmp_path)
    before = sha256(path)
    p = recover_provenance(path, tmp_path)
    assert sha256(path) == before
    assert [r['scenario_token'] for r in p['rows']] == ['02', '01']
    assert [r['scenario_type'] for r in p['rows']] == ['low_magnitude_speed', 'stationary_in_traffic']
    arrays = aligned_arrays(path, tmp_path/'trajs_provenance.json')
    assert [len(a) for a in arrays] == [2]*4
    p['rows'].reverse(); write_json(tmp_path/'bad.json', p)
    with pytest.raises(ValueError, match='alignment'):
        aligned_arrays(path, tmp_path/'bad.json')
    p['rows'].pop(); write_json(tmp_path/'bad.json', p)
    with pytest.raises(ValueError, match='length'):
        aligned_arrays(path, tmp_path/'bad.json')


@pytest.mark.parametrize('failure', ['missing', 'ambiguous'])
def test_recovery_refuses_unreliable_join(tmp_path, failure):
    path, db = make_recovery_fixture(tmp_path)
    if failure == 'missing':
        np.save(tmp_path/'trajs_tokens.npy', np.array(['03', '01']))
    else:
        (tmp_path/'duplicate.db').write_bytes(db.read_bytes())
    with pytest.raises(ValueError):
        recover_provenance(path, tmp_path)
    assert not (tmp_path/'trajs_provenance.json').exists()


def test_extraction_prepares_complete_row_before_append(monkeypatch):
    import skill_pool.nuplan_extract as extract
    monkeypatch.setattr(extract, '_local_future', lambda *args: np.zeros((30, 2)))
    bad = SimpleNamespace(token='bad', log_name='', scenario_type='turn', _log_file_load_path='a.db')
    good = SimpleNamespace(token='ok', log_name='a', scenario_type='turn', _log_file_load_path='a.db')
    trajectories, rows = [], []
    for scenario in [bad, good]:
        try:
            trajectory, row = _successful_row(scenario, 30, 3, len(trajectories))
        except ValueError:
            continue
        trajectories.append(trajectory); rows.append(row)
    assert len(trajectories) == len(rows) == 1
    assert rows[0]['source_index'] == 0 and rows[0]['scenario_token'] == 'ok'


def test_formal_split_and_baseline_artifacts():
    root = Path(__file__).resolve().parents[1]
    p = json.loads((root/'data/ego_trajs_provenance.json').read_text())
    rows = p['rows']; logs = np.array([r['log_name'] for r in rows]); types = np.array([r['scenario_type'] for r in rows])
    c = np.load(root/'splits/construction_indices.npy'); h = np.load(root/'splits/heldout_indices.npy')
    report = json.loads((root/'splits/split_report.json').read_text())
    valid = np.ones(len(rows), dtype=bool)  # Formal run records zero invalid rows.
    assert report['invalid_trajectory_count'] == 0
    assert_partition(logs, valid, c, h)
    assert report['alignment_checks']['passed']
    assert report['token_duplicate_count'] == report['cross_split_token_overlap_count'] == 0
    quality = distribution(types, c, h, logs)[1]
    assert quality['nonconcentrated_major_types_passed']
    assert quality['imbalanced_major_types'] == ['behind_long_vehicle']
    assert not quality['passed']  # Real single-log concentration must remain visible.
    assert quality['major_types_concentrated_in_one_log']['behind_long_vehicle']['largest_log_count'] == 653
    c2, h2 = split_by_log(logs, valid, seed=7, scenario_types=types)
    np.testing.assert_array_equal(c, c2); np.testing.assert_array_equal(h, h2)
    for name, indices in [('construction', c), ('heldout', h)]:
        assert set((root/f'splits/{name}_logs.txt').read_text().splitlines()) == set(logs[indices])
    for name in ['scenario_distribution', 'log_trajectory_counts']:
        assert (root/f'outputs/split/{name}.png').read_bytes().startswith(b'\x89PNG')
    baseline = root/'outputs/neutral'
    config = json.loads((baseline/'baseline_config.json').read_text())
    assert config['neutral_source_index'] == 8052
    for name, expected in config['files'].items():
        assert sha256(baseline/name) == expected
    if (root/'data/ego_trajs.npy').exists():
        arrays = aligned_arrays(root/'data/ego_trajs.npy', root/'data/ego_trajs_provenance.json')
        assert [len(a) for a in arrays] == [57992]*4
