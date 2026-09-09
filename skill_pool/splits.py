"""Recover row provenance and split whole nuPlan logs, without touching trajectories."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import sqlite3
import numpy as np
from .nuplan_extract import DEFAULT_SCENARIO_TYPES


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def recover_provenance(trajectory_path, database_root):
    """Exact token join, retaining input row order and devkit MAX(filtered type)."""
    path = Path(trajectory_path)
    tokens_path = path.with_name(path.stem+'_tokens.npy')
    metadata_path = path.with_name(path.stem+'_metadata.json')
    raw = np.load(path, mmap_mode='r', allow_pickle=False)
    tokens = np.load(tokens_path, allow_pickle=False)
    metadata = json.loads(metadata_path.read_text())
    if raw.ndim != 3 or raw.shape[1:] != (30, 2) or tokens.shape != (len(raw),):
        raise ValueError('Trajectory/token shape mismatch')
    if metadata.get('trajectory_count') != len(raw):
        raise ValueError('Metadata trajectory count mismatch')
    wanted = set(tokens.tolist())
    found = {}
    databases = sorted(Path(database_root).rglob('*.db'))
    if not databases:
        raise ValueError('No databases found')
    types = None if metadata['all_scenario_types'] else metadata.get('scenario_types', list(DEFAULT_SCENARIO_TYPES))
    where = '' if types is None else 'WHERE st.type IN ('+','.join('?' for _ in types)+')'
    for db in databases:
        with sqlite3.connect(db.resolve().as_uri()+'?mode=ro', uri=True) as conn:
            rows = conn.execute(f'''SELECT lower(hex(lp.token)), l.logfile, MAX(st.type)
                FROM lidar_pc lp JOIN lidar ld ON lp.lidar_token=ld.token
                JOIN log l ON ld.log_token=l.token
                LEFT JOIN scenario_tag st ON st.lidar_pc_token=lp.token
                {where} GROUP BY lp.token, l.logfile''', types or [])
            for token, log_name, scenario_type in rows:
                if token not in wanted:
                    continue
                if token in found:
                    raise ValueError(f'Ambiguous token in multiple database/log rows: {token}')
                if not log_name or log_name != db.stem:
                    raise ValueError(f'Database log name disagrees with devkit filename convention: {db}')
                found[token] = (log_name, scenario_type or 'unknown', str(db.resolve()))
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f'Cannot reliably recover {len(missing)} tokens; metadata not written')
    rows = [dict(source_index=i, scenario_token=str(t), log_name=found[t][0],
                 scenario_type=found[t][1], database_source=found[t][2]) for i, t in enumerate(tokens)]
    provenance = dict(schema_version=1, trajectory_sha256=sha256(path), tokens_sha256=sha256(tokens_path),
        recovery_method='Read-only token join via lidar_pc -> lidar -> log; MAX(scenario_tag.type) after original type filter, matching installed nuPlan devkit',
        database_count=len(databases), scenario_type_filter=types, rows=rows)
    destination = path.with_name(path.stem+'_provenance.json')
    write_json(destination, provenance)
    metadata['row_provenance_file'] = destination.name
    metadata['row_provenance_sha256'] = sha256(destination)
    metadata['scenario_types'] = types
    write_json(metadata_path, metadata)
    return provenance


def aligned_arrays(trajectory_path, provenance_path):
    path = Path(trajectory_path)
    tokens_path = path.with_name(path.stem+'_tokens.npy')
    raw = np.load(path, mmap_mode='r', allow_pickle=False)
    tokens = np.load(tokens_path, allow_pickle=False)
    p = json.loads(Path(provenance_path).read_text())
    rows = p['rows']
    if raw.ndim != 3 or raw.shape[1:] != (30, 2) or tokens.shape != (len(raw),) or len(rows) != len(raw):
        raise ValueError('Trajectory/token/provenance length or shape mismatch')
    if p['trajectory_sha256'] != sha256(path) or p['tokens_sha256'] != sha256(tokens_path):
        raise ValueError('Input hash mismatch: provenance belongs to different arrays')
    for i, row in enumerate(rows):
        if row['source_index'] != i or row['scenario_token'] != tokens[i]:
            raise ValueError(f'Row/token alignment mismatch at {i}')
        if any(not isinstance(row[k], str) or not row[k].strip() for k in ['log_name', 'scenario_type', 'database_source']):
            raise ValueError(f'Missing provenance at {i}')
    return raw, tokens, np.array([r['log_name'] for r in rows]), np.array([r['scenario_type'] for r in rows])


def split_by_log(logs, valid, seed=7, construction_fraction=0.8, scenario_types=None):
    logs, valid = np.asarray(logs), np.asarray(valid, dtype=bool)
    if logs.ndim != 1 or valid.shape != logs.shape or not 0 < construction_fraction < 1:
        raise ValueError('Invalid logs, mask or fraction')
    unique = np.unique(logs[valid])  # Sort before permutation: independent of input row order.
    if len(unique) < 2:
        raise ValueError('At least two valid logs required')
    shuffled = np.random.default_rng(seed).permutation(unique)
    count = min(len(unique)-1, max(1, int(np.floor(len(unique)*construction_fraction+0.5))))
    chosen = shuffled[:count]
    if scenario_types is not None:
        types = np.asarray(scenario_types)
        if types.shape != logs.shape:
            raise ValueError('Scenario type length mismatch')
        labels = np.unique(types[valid])
        matrix = np.array([[np.sum(valid & (logs == log) & (types == label))
                            for label in labels] for log in shuffled], dtype=float)
        total = matrix.sum(axis=0)
        major = total / total.sum() >= 0.01
        selected = np.arange(count)

        def objective(c):
            h = total-c
            gap = c/c.sum()-h/h.sum()
            missing = np.count_nonzero((c[major] == 0) | (h[major] == 0))
            # Equal class weighting, plus trajectory-size balance. No model outputs.
            return float(np.sum(gap**2) + (c.sum()/total.sum()-construction_fraction)**2 + missing)

        current = matrix[selected].sum(axis=0)
        score = objective(current)
        for _ in range(200):
            best = None
            for left in selected:
                for right in np.setdiff1d(np.arange(len(shuffled)), selected):
                    candidate = current-matrix[left]+matrix[right]
                    value = objective(candidate)
                    if value < score-1e-12:
                        score = value
                        best = left, right, candidate
            if best is None:
                break
            left, right, current = best
            selected[selected == left] = right
            selected.sort()
        chosen = shuffled[selected]
    construction = np.flatnonzero(valid & np.isin(logs, chosen))
    heldout = np.flatnonzero(valid & ~np.isin(logs, chosen))
    return construction, heldout


def distribution(types, construction, heldout, logs=None):
    labels = sorted(set(types[construction]) | set(types[heldout]))
    counts = [Counter(types[indices]) for indices in [construction, heldout]]
    result = {label: dict(construction_count=counts[0][label], heldout_count=counts[1][label],
        construction_fraction=counts[0][label]/len(construction), heldout_fraction=counts[1][label]/len(heldout)) for label in labels}
    # Declared diagnostic, never retry seed or move logs to improve held-out composition.
    major = [label for label in labels if sum(c[label] for c in counts)/(len(construction)+len(heldout)) >= 0.01]
    gaps = {label: abs(result[label]['construction_fraction']-result[label]['heldout_fraction']) for label in labels}
    checks = dict(major_type_min_total_fraction=0.01, max_allowed_absolute_fraction_gap=0.10,
        major_types=major, missing_major_types=[t for t in major if not counts[0][t] or not counts[1][t]],
        absolute_fraction_gaps=gaps,
        total_variation_distance=0.5*sum(gaps.values()))
    relative = {t: (result[t]['heldout_fraction']/result[t]['construction_fraction']
                    if result[t]['construction_fraction'] else None) for t in major}
    checks['major_type_heldout_to_construction_ratio'] = relative
    checks['allowed_major_type_ratio'] = [0.5, 2.0]
    checks['imbalanced_major_types'] = [t for t in major if relative[t] is None or not 0.5 <= relative[t] <= 2.0 or gaps[t] > 0.10]
    concentrated = {}
    if logs is not None:
        indices = np.union1d(construction, heldout)
        for t in major:
            per_log = Counter(logs[indices][types[indices] == t])
            largest = max(per_log, key=per_log.get)
            fraction = per_log[largest]/sum(per_log.values())
            if fraction >= 0.8:
                concentrated[t] = dict(largest_log=largest, largest_log_count=per_log[largest],
                                      total_count=sum(per_log.values()), largest_log_fraction=fraction)
    checks['major_types_concentrated_in_one_log'] = concentrated
    checks['nonconcentrated_major_types_passed'] = not any(t not in concentrated for t in checks['imbalanced_major_types'])
    checks['passed'] = not checks['imbalanced_major_types']
    return result, checks


def plot_split(output, logs, valid, construction, heldout, scenario_counts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    labels = list(scenario_counts); y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), sharey=True)
    for ax, suffix, xlabel in zip(axes, ['count', 'fraction'], ['Trajectory count', 'Within-split proportion']):
        for offset, group in [(-0.2, 'construction'), (0.2, 'heldout')]:
            ax.barh(y+offset, [scenario_counts[t][group+'_'+suffix] for t in labels], height=0.4, label=group)
        ax.set(yticks=y, yticklabels=labels, xlabel=xlabel); ax.legend(); ax.grid(axis='x', alpha=0.3)
    fig.tight_layout(); fig.savefig(output/'scenario_distribution.png', dpi=160); plt.close(fig)
    counts = Counter(logs[valid]); names = sorted(counts, key=lambda x: (-counts[x], x))
    construction_logs = set(logs[construction])
    fig, ax = plt.subplots(figsize=(16, max(8, len(names)*0.24)))
    ax.barh(np.arange(len(names)), [counts[n] for n in names], color=['tab:blue' if n in construction_logs else 'tab:orange' for n in names])
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='tab:blue', label='construction'), Patch(color='tab:orange', label='heldout')])
    ax.set(yticks=np.arange(len(names)), yticklabels=names, xlabel='Valid trajectories per log'); ax.invert_yaxis()
    ax.tick_params(axis='y', labelsize=7); ax.grid(axis='x', alpha=0.3)
    fig.tight_layout(); fig.savefig(output/'log_trajectory_counts.png', dpi=160); plt.close(fig)


def run_split(trajectory_path, provenance_path, output_dir='splits', plot_dir='outputs/split', seed=7, baseline_dir='outputs/neutral'):
    raw, tokens, logs, types = aligned_arrays(trajectory_path, provenance_path)
    valid = np.isfinite(raw).all(axis=(1, 2))
    initial_c, initial_h = split_by_log(logs, valid, seed)
    _, initial_distribution_check = distribution(types, initial_c, initial_h, logs)
    construction, heldout = split_by_log(logs, valid, seed, scenario_types=types)
    c_logs, h_logs = np.unique(logs[construction]), np.unique(logs[heldout])
    counts, distribution_check = distribution(types, construction, heldout, logs)
    checks = dict(lengths_equal=len(raw)==len(tokens)==len(logs)==len(types),
        hashes_and_row_tokens_aligned=True,
        log_intersection_count=len(np.intersect1d(c_logs, h_logs)),
        index_intersection_count=len(np.intersect1d(construction, heldout)),
        covers_all_valid=bool(np.array_equal(np.union1d(construction, heldout), np.flatnonzero(valid))),
        indices_in_range=bool(all(np.all((idx>=0)&(idx<len(raw))) for idx in [construction, heldout])),
        each_log_in_one_split=not bool(set(c_logs)&set(h_logs)))
    checks['passed'] = checks['lengths_equal'] and checks['covers_all_valid'] and checks['indices_in_range'] and checks['each_log_in_one_split'] and checks['index_intersection_count']==0
    baseline = Path(baseline_dir)
    baseline_names = ['neutral_source_index.npy', 'z0.npy', 'neutral_target.npy', 'G0.npy']
    baseline_hashes = {n: sha256(baseline/n) for n in baseline_names}
    fixed = json.loads((baseline/'baseline_config.json').read_text())
    if baseline_hashes != fixed['files']:
        raise ValueError('Fixed baseline hash mismatch')
    source = int(np.load(baseline/'neutral_source_index.npy', allow_pickle=False))
    if source != 8052 or source >= len(tokens) or tokens[source] != fixed['neutral_source_token']:
        raise ValueError('Expected fixed baseline source index 8052')
    log_counts = Counter(logs[valid]); largest = max(log_counts, key=log_counts.get)
    report = dict(seed=seed, method='Sorted unique valid logs; default_rng(seed) permutation; round-half-up 80% log count; deterministic best-improving whole-log swaps (up to 200), minimizing sum of squared class proportion gaps + squared construction trajectory fraction deviation + missing major-type count; ties follow seeded log order' ,
        requested_construction_fraction=0.8, total_trajectory_count=len(raw), valid_trajectory_count=int(valid.sum()),
        invalid_trajectory_count=int((~valid).sum()), total_log_count=len(np.unique(logs)), valid_log_count=len(log_counts),
        construction=dict(trajectory_count=len(construction), log_count=len(c_logs), trajectory_fraction=len(construction)/int(valid.sum()), log_fraction=len(c_logs)/len(log_counts)),
        heldout=dict(trajectory_count=len(heldout), log_count=len(h_logs), trajectory_fraction=len(heldout)/int(valid.sum()), log_fraction=len(h_logs)/len(log_counts)),
        scenario_counts=counts, scenario_distribution_check=distribution_check,
        initial_random_distribution_check=initial_distribution_check,
        log_trajectory_counts=dict(sorted(log_counts.items())), largest_log=dict(log_name=largest, trajectory_count=log_counts[largest], fraction=log_counts[largest]/int(valid.sum())),
        token_duplicate_count=len(tokens)-len(np.unique(tokens)), valid_token_duplicate_count=int(valid.sum())-len(np.unique(tokens[valid])),
        cross_split_token_overlap_count=len(np.intersect1d(tokens[construction], tokens[heldout])),
        alignment_checks=checks, trajectory_sha256=sha256(trajectory_path),
        tokens_sha256=sha256(Path(trajectory_path).with_name(Path(trajectory_path).stem+'_tokens.npy')), provenance_sha256=sha256(provenance_path),
        fixed_neutral_baseline=dict(source_index=source, hashes=baseline_hashes,
            source_split='construction' if source in construction else 'heldout' if source in heldout else 'excluded',
            note='Baseline selected before split using full data; held-out is not untouched with respect to this fixed baseline. No baseline recomputation; split membership is not constrained by baseline.'))
    if report['cross_split_token_overlap_count']:
        raise ValueError('Duplicate token spans splits; investigate provenance')
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    for name, indices, names in [('construction', construction, c_logs), ('heldout', heldout, h_logs)]:
        np.save(out/(name+'_indices.npy'), indices)
        (out/(name+'_logs.txt')).write_text('\n'.join(names)+'\n')
    write_json(out/'split_report.json', report)
    plot_split(plot_dir, logs, valid, construction, heldout, counts)
    if baseline_hashes != {n: sha256(baseline/n) for n in baseline_names}:
        raise RuntimeError('Baseline changed during split')
    return report
