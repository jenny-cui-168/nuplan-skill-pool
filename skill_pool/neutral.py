"""Standalone neutral validation; never builds a pool or updates VAE weights."""
from pathlib import Path
import json
import numpy as np
from .data import encode_trajectories, decode_latents
from .model import load_vae
from .geometry import decoder_metric

LIMITS = dict(min_speed=2.0, max_speed=25.0, max_lateral=0.5,
              max_heading_deg=5.0, max_heading_change_deg=5.0,
              max_speed_std=0.5, max_speed_range=2.0)


def neutral_target(speed, dt=0.1):
    if not np.isfinite(speed) or speed <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError('speed and dt must be finite and positive')
    return np.column_stack((np.zeros(30), speed * dt * np.arange(1, 31)))


def curves(traj, dt=0.1):
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    delta = np.diff(np.concatenate((np.zeros((1, 2)), traj)), axis=0)
    return np.linalg.norm(delta, axis=1) / dt, np.degrees(np.unwrap(np.arctan2(delta[:, 0], delta[:, 1]))), delta


def describe(traj, dt=0.1):
    speed, heading, delta = curves(traj, dt)
    return dict(forward_displacement_m=float(traj[-1, 1]),
                final_lateral_displacement_m=float(traj[-1, 0]),
                max_lateral_displacement_m=float(np.max(np.abs(traj[:, 0]))),
                mean_speed_mps=float(speed.mean()), speed_std_mps=float(speed.std()),
                speed_range_mps=float(np.ptp(speed)),
                heading_change_deg=float(np.ptp(heading)),
                max_heading_deg=float(np.max(np.abs(heading))),
                min_forward_step_m=float(delta[:, 1].min()))


def neutral_checks(traj, dt=0.1):
    if not np.isfinite(traj).all():
        return {'finite': False, 'passed': False}
    m = describe(traj, dt)
    checks = dict(finite=True,
        normal_forward=LIMITS['min_speed'] <= m['mean_speed_mps'] <= LIMITS['max_speed'] and m['min_forward_step_m'] > 0,
        small_lateral=m['max_lateral_displacement_m'] <= LIMITS['max_lateral'],
        stable_heading=m['heading_change_deg'] <= LIMITS['max_heading_change_deg'] and m['max_heading_deg'] <= LIMITS['max_heading_deg'],
        stable_speed=m['speed_std_mps'] <= LIMITS['max_speed_std'] and m['speed_range_mps'] <= LIMITS['max_speed_range'])
    checks['passed'] = all(checks.values())
    return checks


def metric_diagnostics(metric):
    finite = bool(np.isfinite(metric).all())
    symmetric = finite and bool(np.allclose(metric, metric.T, atol=1e-8, rtol=1e-6))
    eig = np.linalg.eigvalsh(metric) if finite else None
    positive = symmetric and bool(eig.min() > 0)
    return dict(eigenvalues=eig.tolist() if finite else None,
                min_eigenvalue=float(eig.min()) if finite else None,
                max_eigenvalue=float(eig.max()) if finite else None,
                condition_number=float(np.linalg.cond(metric)) if positive else None,
                symmetric=symmetric, positive_definite=positive,
                contains_nan=bool(np.isnan(metric).any()), contains_inf=bool(np.isinf(metric).any()),
                passed=positive)


def ade(a, b):
    return float(np.linalg.norm(a-b, axis=-1).mean())


def plot_diagnostics(out, target, raw, decoded, records, dt):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    time = dt * np.arange(1, 31)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for traj, label in [(target, 'Neutral target'), (raw[0], 'Source'), (decoded[0], 'D(z0)')]:
        speed, heading, _ = curves(traj, dt)
        full = np.vstack((np.zeros(2), traj))
        axes[0, 0].plot(full[:, 0], full[:, 1], label=label)
        axes[0, 1].plot(time, speed, label=label)
        axes[1, 0].plot(time, traj[:, 0], label=label)
        axes[1, 1].plot(time, heading, label=label)
    axes[0, 0].set(xlabel='X lateral (m)', ylabel='Y forward (m)', xlim=(-5, 5))
    axes[0, 0].set_aspect('equal', adjustable='box')
    for ax, ylabel in [(axes[0, 1], 'Speed (m/s)'), (axes[1, 0], 'X lateral (m)'), (axes[1, 1], 'Heading from +Y (deg)')]:
        ax.set(xlabel='Time (s)', ylabel=ylabel)
    for ax in axes.flat:
        ax.legend(); ax.grid(True)
    fig.tight_layout(); fig.savefig(out / 'neutral_check.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 5, figsize=(22, 12))
    for ax, src, dec, rec in zip(axes.flat, raw, decoded, records):
        for traj, label in [(target, 'Target'), (src, 'Source'), (dec, 'Decoded')]:
            full = np.vstack((np.zeros(2), traj))
            ax.plot(full[:, 0], full[:, 1], label=label)
        lines = [f"#{rec['rank']} source {rec['source_index']}"]
        for key, label in [('original', 'S'), ('decoded', 'D')]:
            m = rec[key]
            lines.append(f"{label}: v={m['mean_speed_mps']:.2f}, std={m['speed_std_mps']:.2f} m/s")
            lines.append(f"x_end={m['final_lateral_displacement_m']:.2f}, |x|max={m['max_lateral_displacement_m']:.2f} m")
            lines.append(f"heading range={m['heading_change_deg']:.2f} deg; ADE={m['target_ade_m']:.3f} m")
        ax.set_title('\n'.join(lines), fontsize=8)
        ax.set(xlabel='X lateral (m)', ylabel='Y forward (m)', xlim=(-5, 5))
        ax.set_aspect('equal', adjustable='box'); ax.grid(True); ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(records):]:
        ax.set_visible(False)
    fig.tight_layout(); fig.savefig(out / 'neutral_candidates.png', dpi=180); plt.close(fig)


def run_neutral_check(trajectories_path, checkpoint_path, output_dir, tokens_path=None, dt=0.1, damping=1e-4, batch_size=1024):
    if not np.isfinite(damping) or damping <= 0 or batch_size < 1:
        raise ValueError('positive damping and batch_size required')
    trajectories_path = Path(trajectories_path)
    raw = np.load(trajectories_path, allow_pickle=False)
    if raw.ndim != 3 or raw.shape[1:] != (30, 2):
        raise ValueError('Expected [N,30,2] future points')
    tokens_path = Path(tokens_path) if tokens_path else trajectories_path.with_name(trajectories_path.stem + '_tokens.npy')
    tokens = np.load(tokens_path, allow_pickle=False)
    if tokens.shape != (len(raw),):
        raise ValueError('Tokens must align with original input rows')
    # Never compact source rows: saved indices refer directly to the input file.
    normal_speeds = []
    indices = []
    for i, traj in enumerate(raw):
        if not np.isfinite(traj).all():
            continue
        m = describe(traj, dt)
        if LIMITS['min_speed'] <= m['mean_speed_mps'] <= LIMITS['max_speed'] and m['min_forward_step_m'] > 0:
            normal_speeds.append(m['mean_speed_mps'])
        if neutral_checks(traj, dt)['passed']:
            indices.append(i)
    if not normal_speeds or not indices:
        raise ValueError('No normal moving trajectories or neutral source candidates; no fallback')
    v_ref = float(np.median(normal_speeds))
    target = neutral_target(v_ref, dt)
    model = load_vae(checkpoint_path, 'cpu')
    indices = np.asarray(indices)
    latents, _ = encode_trajectories(model, raw[indices].astype(np.float32), batch_size, 'cpu')
    decoded = decode_latents(model, latents, batch_size, 'cpu')
    scores = np.array([ade(t, target) if np.isfinite(t).all() else np.inf for t in decoded])
    eligible = np.array([neutral_checks(t, dt)['passed'] for t in decoded])
    order = np.flatnonzero(eligible)
    order = order[np.argsort(scores[order], kind='stable')]
    if not len(order):
        raise ValueError('No decoded candidate passes neutral checks; stop, do not build pool')
    records = []
    for rank, j in enumerate(order[:10], 1):
        i = int(indices[j])
        records.append(dict(rank=rank, source_index=i, token=str(tokens[i]),
            original={**describe(raw[i], dt), 'target_ade_m': ade(raw[i], target)},
            decoded={**describe(decoded[j], dt), 'target_ade_m': float(scores[j])},
            original_decoded_ade_m=ade(raw[i], decoded[j])))
    best = order[0]; source = int(indices[best]); z0 = latents[best]
    metric = decoder_metric(model, z0, 'cpu', damping)
    metric_report = {**metric_diagnostics(metric), 'lambda': damping, 'formula': 'J_D(z0)^T J_D(z0)/60 + lambda*I'}
    checks = dict(original=neutral_checks(raw[source], dt), decoded=neutral_checks(decoded[best], dt), metric=metric_report['passed'])
    checks['passed'] = checks['original']['passed'] and checks['decoded']['passed'] and checks['metric']
    report = dict(reference_speed_mps=v_ref,
        reference_speed_method='Median of per-trajectory mean speeds, including origin-to-first-future-point; finite, every Y step > 0, mean speed in [2,25] m/s',
        normal_motion_count=len(normal_speeds), normal_speed_quantiles_mps=np.quantile(normal_speeds, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
        source_candidate_count=len(indices), decoded_eligible_count=len(order),
        selection_method='Hard neutrality gates on source and decoded; ascending decoded-to-target ADE; source index breaks ties',
        neutral_source_index=source, neutral_source_token=str(tokens[source]),
        dt_seconds=dt, future_times_seconds=(dt*np.arange(1,31)).tolist(), coordinate_convention='+Y forward, X lateral; origin at t=0',
        thresholds=LIMITS, original=records[0]['original'], decoded=records[0]['decoded'],
        original_decoded_ade_m=records[0]['original_decoded_ade_m'], automatic_checks=checks, top_candidates=records)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    for name, value in dict(neutral_target=target, neutral_source_index=np.array(source), neutral_source_token=np.array(str(tokens[source])), neutral_source_trajectory=raw[source], z0=z0, neutral_decoded=decoded[best], G0=metric).items():
        np.save(out / (name+'.npy'), value)
    for name, value in [('neutral_check', report), ('metric_check', metric_report)]:
        (out / (name+'.json')).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    plot_diagnostics(out, target, raw[indices[order[:10]]], decoded[order[:10]], records, dt)
    return report
