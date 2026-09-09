"""Construction-only local pullback baseline and separate held-out evaluation."""
from pathlib import Path
import json
import numpy as np
import torch
from .data import encode_trajectories, decode_latents
from .geometry import select_direction_strength_pool, decoder_metric
from .model import load_vae
from .neutral import ade, describe, neutral_checks, metric_diagnostics
from .splits import aligned_arrays, sha256, write_json
from .evaluation import plot_pool, reconstruction_metrics


def compare_neutrals(full_dir, construction_dir):
    full, construction = Path(full_dir), Path(construction_dir)
    report = {}
    for label, directory in [('full', full), ('construction', construction)]:
        check = json.loads((directory/'neutral_check.json').read_text())
        report[label] = {k: check[k] for k in ['neutral_source_index', 'reference_speed_mps', 'decoded', 'original']}
        report[label]['G0_eigenvalues'] = np.linalg.eigvalsh(np.load(directory/'G0.npy')).tolist()
    report['latent_euclidean_distance'] = float(np.linalg.norm(np.load(full/'z0.npy')-np.load(construction/'z0.npy')))
    report['decoded_trajectory_ADE_m'] = ade(np.load(full/'neutral_decoded.npy'), np.load(construction/'neutral_decoded.npy'))
    report['reference_speed_change_mps'] = report['construction']['reference_speed_mps']-report['full']['reference_speed_mps']
    write_json(construction/'neutral_comparison.json', report)
    return report


def validate_partition(construction, heldout, valid_indices, count, logs):
    for indices in [construction, heldout]:
        if indices.ndim != 1 or indices.dtype.kind not in 'iu' or not len(indices) or len(np.unique(indices)) != len(indices) or np.any((indices < 0) | (indices >= count)):
            raise ValueError('Invalid split indices')
    if np.intersect1d(construction, heldout).size or np.intersect1d(logs[construction], logs[heldout]).size:
        raise ValueError('Split leakage')
    if not np.array_equal(np.union1d(construction, heldout), valid_indices):
        raise ValueError('Splits must cover all valid original rows')


def select_construction_pool(construction_latents, construction_indices, z0, metric, pool_size):
    """The selector receives only construction rows; returned indices are global."""
    if len(construction_latents) != len(construction_indices):
        raise ValueError('Latent/index mapping mismatch')
    pool, local, directions, strengths = select_direction_strength_pool(
        construction_latents, z0, metric, pool_size, strength_levels=4)
    return pool, construction_indices[local], directions, strengths


def pairwise_diagnostics(trajectories):
    matrix = np.linalg.norm(trajectories[:, None]-trajectories[None, :], axis=-1).mean(axis=-1)
    i, j = np.triu_indices(len(trajectories), 1)
    values = matrix[i, j]
    order = np.argsort(values, kind='stable')
    pairs = [dict(skill_i=int(i[n]), skill_j=int(j[n]), ade_m=float(values[n])) for n in order]
    quantiles = {str(q): float(np.quantile(values, q)) for q in [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1]}
    speed = np.array([describe(t)['mean_speed_mps'] for t in trajectories])
    radius = np.linalg.norm(trajectories, axis=-1).max(axis=1)
    return dict(pair_count=len(values), min_ade_m=float(values.min()), mean_ade_m=float(values.mean()),
        quantiles_m=quantiles, all_pairs_sorted=pairs, near_duplicate_pairs=pairs[:10],
        near_duplicate_policy='Ten closest pairs for inspection; no final threshold, no removal',
        threshold_sensitivity=[dict(ade_threshold_m=t, count=int(np.sum(values <= t))) for t in [1e-6, .001, .01, .05, .1, .25, .5]],
        exact_stationary_count=int(np.sum(radius <= 1e-6)),
        near_stationary_sensitivity=[dict(max_mean_speed_mps=t, max_radius_m=3*t,
            skill_ids=np.flatnonzero((speed <= t)&(radius <= 3*t)).tolist(),
            count=int(np.sum((speed <= t)&(radius <= 3*t)))) for t in [.05, .1, .25, .5]],
        skill_mean_speed_mps=speed.tolist(), skill_max_radius_m=radius.tolist())


def coverage_arrays(trajectories, pool, chunk_size=1024):
    result = {k: [] for k in ['minADE_m', 'minFDE_m', 'FDE_at_minADE_m', 'nearest_skill']}
    for start in range(0, len(trajectories), chunk_size):
        batch = trajectories[start:start+chunk_size]
        distances = np.linalg.norm(batch[:, None]-pool[None], axis=-1)
        average = distances.mean(axis=-1)
        nearest = np.argmin(average, axis=1)
        result['minADE_m'].append(average[np.arange(len(batch)), nearest])
        result['minFDE_m'].append(distances[:, :, -1].min(axis=1))
        result['FDE_at_minADE_m'].append(distances[np.arange(len(batch)), nearest, -1])
        result['nearest_skill'].append(nearest)
    return {k: np.concatenate(v) if v else np.array([], dtype=np.int64 if k=='nearest_skill' else float) for k, v in result.items()}


def summarize_coverage(samples, pool_size):
    count = len(samples['nearest_skill'])
    result = dict(trajectory_count=count, assignment_counts=np.bincount(samples['nearest_skill'], minlength=pool_size).tolist())
    result['used_skill_count'] = int(np.count_nonzero(result['assignment_counts']))
    for key in ['minADE_m', 'minFDE_m', 'FDE_at_minADE_m']:
        result[key] = float(samples[key].mean()) if count else None
    result['minADE_p90_m'] = float(np.quantile(samples['minADE_m'], .9)) if count else None
    return result


def evaluate_split(trajectories, pool, scenario_types, labels):
    samples = coverage_arrays(trajectories, pool)
    result = summarize_coverage(samples, len(pool))
    result['by_scenario_type'] = {}
    for label in labels:
        mask = scenario_types == label
        metrics = summarize_coverage({k: v[mask] for k, v in samples.items()}, len(pool))
        metrics['interpretation'] = 'no samples' if not mask.sum() else 'insufficient samples; descriptive only, no generalization conclusion' if mask.sum() < 30 else 'descriptive coverage; no statistical generalization claim'
        result['by_scenario_type'][label] = metrics
    return result, samples


def verify_neutral(neutral_dir, construction, raw, checkpoint):
    directory = Path(neutral_dir)
    report = json.loads((directory/'neutral_check.json').read_text())
    review = json.loads((directory/'visual_review.json').read_text())
    if not report['automatic_checks']['passed'] or not review['passed']:
        raise ValueError('Neutral check/review failed; stop before encoding or pool selection')
    if report['selection_split'] != 'construction' or not np.array_equal(np.sort(construction), np.load(directory/'selection_indices.npy')):
        raise ValueError('Neutral statistics were not computed on exactly construction')
    if report['checkpoint_sha256'] != sha256(checkpoint):
        raise ValueError('Neutral checkpoint mismatch')
    for name in ['z0.npy', 'G0.npy', 'neutral_decoded.npy', 'neutral_check.png', 'neutral_candidates.png', 'neutral_check.json', 'selection_indices.npy']:
        if review['artifact_sha256'].get(name) != sha256(directory/name):
            raise ValueError('Neutral artifacts changed after visual review')
    source = report['neutral_source_index']
    if source not in construction or not np.array_equal(raw[source], np.load(directory/'neutral_source_trajectory.npy')):
        raise ValueError('Neutral source mapping mismatch')
    decoded = np.load(directory/'neutral_decoded.npy')
    if not neutral_checks(decoded)['passed'] or not neutral_checks(raw[source])['passed']:
        raise ValueError('Neutral no longer passes checks')
    metric = np.load(directory/'G0.npy')
    if not metric_diagnostics(metric)['passed']:
        raise ValueError('Neutral metric invalid')
    return np.load(directory/'z0.npy'), metric, source


def plot_v1_evaluation(out, raw, heldout, types, pools, reports, samples):
    import matplotlib.pyplot as plt
    for k, pool in pools.items():
        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        for offset, split in [(-.2, 'construction'), (.2, 'heldout')]:
            counts = np.array(reports[str(k)][split]['assignment_counts'])
            axes[0].bar(np.arange(k)+offset, counts, width=.4, label=split)
            axes[1].bar(np.arange(k)+offset, counts/counts.sum(), width=.4, label=split)
        axes[0].set(ylabel='Assigned trajectories', title=f'K={k}: nearest by trajectory ADE')
        axes[1].set(xlabel='Skill ID', ylabel='Within-split assignment fraction')
        for ax in axes: ax.legend(); ax.grid(axis='y', alpha=.25)
        fig.tight_layout(); fig.savefig(out/f'skill_usage_{k}.png', dpi=160); plt.close(fig)
        values = samples[k]['heldout']; order = np.argsort(values['minADE_m'], kind='stable')
        fig, axes = plt.subplots(1, 3, figsize=(15, 7))
        for ax, rank, label in zip(axes, [0, len(order)//2, len(order)-1], ['Best', 'Median', 'Worst']):
            local = int(order[rank]); source = int(heldout[local]); skill = int(values['nearest_skill'][local])
            for traj, name in [(raw[source], 'Held-out source'), (pool[skill], f'Skill {skill}')]:
                full = np.vstack((np.zeros(2), traj)); ax.plot(full[:, 0], full[:, 1], label=name)
            ax.set_title(f"{label}: row {source}, ADE={values['minADE_m'][local]:.3f} m\n{types[source]}", fontsize=9)
            ax.set(xlabel='X lateral (m)', ylabel='Y forward (m)'); ax.set_aspect('equal', adjustable='datalim'); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out/f'nearest_skill_examples_{k}.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2*len(pools), 5, figsize=(20, 9*len(pools)), squeeze=False)
    for row, (k, pool) in enumerate(pools.items()):
        for position, pair in enumerate(reports[str(k)]['pairwise']['near_duplicate_pairs']):
            ax = axes[2*row+position//5, position%5]
            for index in [pair['skill_i'], pair['skill_j']]:
                ax.plot(pool[index, :, 0], pool[index, :, 1], label=f'Skill {index}')
            ax.set_title(f"K={k}, ADE={pair['ade_m']:.3g} m")
            ax.set(xlabel='X lateral (m)', ylabel='Y forward (m)'); ax.set_aspect('equal', adjustable='datalim'); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out/'duplicate_skill_pairs.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric in zip(axes, ['minADE_m', 'minFDE_m', 'FDE_at_minADE_m']):
        for offset, split in [(-.18, 'construction'), (.18, 'heldout')]:
            ax.bar(np.arange(len(pools))+offset, [reports[str(k)][split][metric] for k in pools], width=.36, label=split)
        ax.set(xticks=np.arange(len(pools)), xticklabels=[f'K={k}' for k in pools], ylabel='Distance (m)', title=metric); ax.legend(); ax.grid(axis='y', alpha=.25)
    fig.tight_layout(); fig.savefig(out/'coverage_comparison.png', dpi=160); plt.close(fig)


def run_v1(trajectory_path, checkpoint, provenance_path, split_dir='splits', neutral_dir='outputs/neutral_construction', output_dir='outputs/v1', seed=7):
    np.random.seed(seed); torch.manual_seed(seed)
    raw, tokens, logs, types = aligned_arrays(trajectory_path, provenance_path)
    split_dir, out = Path(split_dir), Path(output_dir)
    construction = np.load(split_dir/'construction_indices.npy', allow_pickle=False)
    heldout = np.load(split_dir/'heldout_indices.npy', allow_pickle=False)
    valid = np.flatnonzero(np.isfinite(raw).all(axis=(1, 2)))
    validate_partition(construction, heldout, valid, len(raw), logs)
    z0, metric, neutral_source = verify_neutral(neutral_dir, construction, raw, checkpoint)
    neutral_report = json.loads((Path(neutral_dir)/'neutral_check.json').read_text())
    if neutral_report['trajectory_sha256'] != sha256(trajectory_path):
        raise ValueError('Neutral input data mismatch')
    model = load_vae(checkpoint, 'cpu')
    means, _ = encode_trajectories(model, np.asarray(raw[valid], dtype=np.float32), 1024, 'cpu')
    reverse = np.full(len(raw), -1, dtype=np.int64); reverse[valid] = np.arange(len(valid))
    construction_latents, heldout_latents = means[reverse[construction]], means[reverse[heldout]]
    np.testing.assert_allclose(z0, means[reverse[neutral_source]], atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(metric, decoder_metric(model, z0, 'cpu'), atol=1e-6, rtol=1e-5)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out/'all_latents.npy', means); np.save(out/'all_latent_source_indices.npy', valid)
    for name, indices, latents in [('construction', construction, construction_latents), ('heldout', heldout, heldout_latents)]:
        np.save(out/f'{name}_indices.npy', indices); np.save(out/f'{name}_latents.npy', latents)
    report = dict(base_commit='63ccd91', seed=seed, selection_split='construction', evaluation_splits=['construction', 'heldout'],
        method='v_i = z_i-z0; G0=J_D(z0)^T J_D(z0)/60 + lambda I; spherical FPS and four strength quantiles',
        strength_quantiles=np.linspace(.25,.90,4).tolist(), strength_trim_quantile=.995, lambda_value=1e-4,
        deduplication_applied=False, neutral_source_index=neutral_source, neutral_selection_count=len(construction),
        trajectory_sha256=sha256(trajectory_path), checkpoint_sha256=sha256(checkpoint), provenance_sha256=sha256(provenance_path),
        construction_indices_sha256=sha256(split_dir/'construction_indices.npy'), heldout_indices_sha256=sha256(split_dir/'heldout_indices.npy'),
        interpretation='Held-out not used to fit reference speed, neutral, metric, directions or strengths. Frozen VAE training provenance is not established here; this evaluates pool construction generalization, not proof of VAE train/test independence.', pools={})
    pools, sample_records = {}, {}
    # All selection calls occur before held-out evaluation; selector has no held-out argument.
    for k in [32, 64]:
        latent, indices, directions, strengths = select_construction_pool(construction_latents, construction, z0, metric, k)
        if not np.isin(indices, construction).all() or np.isin(indices, heldout).any():
            raise ValueError('Pool source leakage')
        decoded = decode_latents(model, latent, 1024, 'cpu'); pools[k] = decoded
        payload = dict(latents=latent, trajectories=decoded, source_indices=indices, source_tokens=tokens[indices],
            source_trajectories=np.asarray(raw[indices]), direction_ids=directions, strengths=strengths,
            neutral_latent=z0, decoder_metric=metric, selection_split=np.array('construction'), seed=np.array(seed))
        np.savez_compressed(out/f'skill_pool_{k}.npz', **payload)
        torch.save({key: torch.from_numpy(value) if isinstance(value, np.ndarray) and value.dtype.kind not in 'US' else value.tolist() for key, value in payload.items()}, out/f'skill_pool_{k}.pt')
        plot_pool(decoded, directions, out/f'skill_pool_{k}.png', f'Construction-only V1, K={k} (no deduplication)')
        report['pools'][str(k)] = dict(selection_split='construction', source_indices=indices.tolist(), pairwise=pairwise_diagnostics(decoded))
    for k, decoded in pools.items():
        sample_records[k] = {}
        for split, indices, latents in [('construction', construction, construction_latents), ('heldout', heldout, heldout_latents)]:
            metrics, samples = evaluate_split(raw[indices], decoded, types[indices], sorted(set(types)))
            metrics['evaluation_split'] = split
            metrics['pairwise_trajectory_ADE_mean_m'] = report['pools'][str(k)]['pairwise']['mean_ade_m']
            metrics['pairwise_trajectory_ADE_min_m'] = report['pools'][str(k)]['pairwise']['min_ade_m']
            report['pools'][str(k)][split] = metrics
            sample_records[k][split] = samples
            np.savez_compressed(out/f'coverage_{split}_{k}.npz', source_indices=indices, **samples)
    report['reconstruction'] = {split: reconstruction_metrics(raw[indices], decode_latents(model, latents, 1024, 'cpu')) for split, indices, latents in [('construction', construction, construction_latents), ('heldout', heldout, heldout_latents)]}
    write_json(out/'report.json', report)
    write_json(out/'duplicate_diagnostics.json', {k: v['pairwise'] for k, v in report['pools'].items()})
    plot_v1_evaluation(out, raw, heldout, types, pools, report['pools'], sample_records)
    return report
