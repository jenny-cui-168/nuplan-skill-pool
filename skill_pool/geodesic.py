"""Variational discrete geodesics for the *unchanged* frozen decoder.

A numerical initial-velocity estimate, not a claim of exact continuous Log or
unique/global minimal geodesics. ReLU makes the metric only piecewise smooth.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
import traceback
import numpy as np
import torch
from .splits import sha256, write_json


@dataclass(frozen=True)
class SolverConfig:
    segments: int = 8
    damping: float = 1e-4
    max_iterations: int = 100
    gradient_tolerance: float = 1e-4
    quadrature_points: int = 2
    seed: int = 7

    def validate(self):
        if any(not isinstance(v,int) for v in [self.segments,self.max_iterations,self.quadrature_points]):
            raise ValueError('Segment, iteration and quadrature counts must be integers')
        if self.segments < 2 or self.max_iterations < 1 or self.quadrature_points < 1:
            raise ValueError('segments >=2, positive iterations and quadrature required')
        if not np.isfinite(self.damping) or self.damping <= 0 or not np.isfinite(self.gradient_tolerance) or self.gradient_tolerance <= 0:
            raise ValueError('Positive finite damping and gradient tolerance required')


def decoder_jacobians(model, points):
    """[P,60,8]; retain dependence of the Jacobian on path points."""
    if not hasattr(torch, 'func'):
        raise RuntimeError('Geodesic experiment requires PyTorch >=2.0 (torch.func)')
    def decode_one(z):
        return model.decode(z[None]).reshape(-1)
    return torch.vmap(torch.func.jacfwd(decode_one))(points)


def pullback_metrics(model, points, damping):
    j = decoder_jacobians(model, points)
    return j.transpose(-1, -2) @ j / j.shape[-2] + damping*torch.eye(points.shape[-1], dtype=points.dtype, device=points.device)


def path_energy(model, path, damping=1e-4, quadrature_points=2):
    """E_L= L/2 sum_j sum_q w_q delta_j^T G(p_j+u_q delta_j) delta_j.

    Gauss-Legendre integration on each straight latent segment, time in [0,1].
    For constant G the energy is exactly 1/2 (z1-z0)^T G (z1-z0).
    """
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
    nodes = torch.as_tensor((nodes+1)/2, dtype=path.dtype, device=path.device)
    weights = torch.as_tensor(weights/2, dtype=path.dtype, device=path.device)
    delta = path[1:]-path[:-1]
    points = path[:-1, None]+nodes[None, :, None]*delta[:, None]
    metric = pullback_metrics(model, points.reshape(-1, path.shape[-1]), damping)
    metric = metric.reshape(len(delta), quadrature_points, path.shape[-1], path.shape[-1])
    return .5*len(delta)*torch.einsum('li,lqij,lj,q->', delta, metric, delta, weights)


def resident_memory_mb():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return float(line.split()[1])/1024
    return None


def solve_log(model, z0, target, source_index, config=SolverConfig()):
    started = time.perf_counter()
    record = dict(source_index=int(source_index), config=asdict(config), converged=False,
                  failure_reason=None, traceback=None, initial_energy=None, final_energy=None,
                  iterations=0, closure_evaluations=0, energy_curve=[], gradient_residual=None,
                  finite=False, endpoints_fixed=False, energy_nonincreasing=False,
                  rss_before_mb=resident_memory_mb(), device=str(next(model.parameters()).device))
    best_path = None
    try:
        config.validate()
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('Decoder/model parameters must be frozen; latent gradients remain enabled')
        torch.manual_seed(config.seed)
        parameter = next(model.parameters())
        start = torch.as_tensor(z0, dtype=parameter.dtype, device=parameter.device).detach()
        end = torch.as_tensor(target, dtype=parameter.dtype, device=parameter.device).detach()
        if start.ndim != 1 or start.shape != end.shape or not torch.isfinite(start).all() or not torch.isfinite(end).all():
            raise ValueError('Nonfinite or incompatible endpoints')
        times = torch.linspace(0, 1, config.segments+1, dtype=start.dtype, device=start.device)
        initial = start[None]+times[:, None]*(end-start)[None]
        initial[0] = start; initial[-1] = end
        best_path = initial.detach().clone()
        internal = torch.nn.Parameter(initial[1:-1].clone())
        def path(): return torch.cat((start[None], internal, end[None]))
        initial_energy = float(path_energy(model, initial, config.damping, config.quadrature_points).detach())
        if not np.isfinite(initial_energy): raise FloatingPointError('Nonfinite initial energy')
        record['initial_energy'] = initial_energy
        record['energy_curve'] = [initial_energy]
        scale = max(initial_energy, 1e-12)
        optimizer = torch.optim.LBFGS([internal], lr=1, max_iter=1, history_size=20,
            tolerance_grad=0, tolerance_change=0, line_search_fn='strong_wolfe')
        best_energy = initial_energy
        def closure():
            optimizer.zero_grad()
            energy = path_energy(model, path(), config.damping, config.quadrature_points)/scale
            if not torch.isfinite(energy): raise FloatingPointError('Nonfinite energy')
            energy.backward()
            if not torch.isfinite(internal.grad).all(): raise FloatingPointError('Nonfinite latent gradient')
            record['closure_evaluations'] += 1
            return energy
        for iteration in range(config.max_iterations+1):
            energy = closure()
            residual = float(internal.grad.abs().max())
            value = float(energy.detach())*scale
            if value <= best_energy:
                best_energy = value; best_path = path().detach().clone()
            record['gradient_residual'] = residual
            if residual <= config.gradient_tolerance:
                record['converged'] = True
                best_path = path().detach().clone()
                break
            if iteration == config.max_iterations: break
            optimizer.step(closure)
            record['iterations'] += 1
            current = float(path_energy(model, path(), config.damping, config.quadrature_points).detach())
            record['energy_curve'].append(current)
        record['final_energy'] = float(path_energy(model, best_path, config.damping, config.quadrature_points).detach())
        # Recompute residual at returned path: a low-energy iterate is not automatically stationary.
        internal.data.copy_(best_path[1:-1])
        closure()
        record['gradient_residual'] = float(internal.grad.abs().max())
        record['converged'] = record['gradient_residual'] <= config.gradient_tolerance
        record['validation_energy_4point'] = float(path_energy(model, best_path, config.damping, 4).detach())
        record['quadrature_relative_gap'] = abs(record['validation_energy_4point']-record['final_energy'])/max(record['final_energy'], 1e-12)
        record['endpoints_fixed'] = bool(torch.equal(best_path[0], start) and torch.equal(best_path[-1], end))
        record['finite'] = bool(torch.isfinite(best_path).all())
        record['energy_nonincreasing'] = record['final_energy'] <= initial_energy + 1e-10*max(1, initial_energy)
        if not (record['finite'] and record['endpoints_fixed'] and record['energy_nonincreasing']):
            record['converged'] = False
            raise RuntimeError('Path numerical acceptance failed')
        if not record['converged']:
            raise RuntimeError(f"Nonconvergence: normalized energy gradient infinity norm {record['gradient_residual']:.6g} exceeds {config.gradient_tolerance}; budget {config.max_iterations}")
    except Exception as error:
        record['converged'] = False
        record['failure_reason'] = str(error)
        record['traceback'] = traceback.format_exc()
    if best_path is None:
        length = max(2, int(config.segments)+1)
        dimension = len(np.atleast_1d(z0))
        path_array = np.full((length, dimension), np.nan)
        log = np.full(dimension, np.nan)
    else:
        path_array = best_path.cpu().numpy()
        log = config.segments*(path_array[1]-path_array[0])
    record['runtime_seconds'] = time.perf_counter()-started
    record['rss_after_mb'] = resident_memory_mb()
    record['cuda_allocated_bytes'] = torch.cuda.memory_allocated() if torch.cuda.is_available() else None
    record['log_vector'] = log.tolist() if np.isfinite(log).all() else None
    return log, path_array, record


def candidate_descriptors(trajectories, dt=.1):
    delta = np.diff(np.concatenate((np.zeros((len(trajectories), 1, 2)), trajectories), axis=1), axis=1)
    speed = np.linalg.norm(delta, axis=-1)/dt
    heading = np.unwrap(np.arctan2(delta[..., 0], delta[..., 1]), axis=1)
    # Ignore nearly stationary segments in heading/curvature descriptors.
    moving = speed > .2
    dh = np.diff(heading, axis=1)
    distance = (np.linalg.norm(delta[:, 1:], axis=-1)+np.linalg.norm(delta[:, :-1], axis=-1))/2
    reliable = moving[:, 1:] & moving[:, :-1]
    curvature = np.where(reliable, np.abs(dh)/np.maximum(distance, .02), 0)
    headings = np.where(moving, heading, np.nan)
    heading_range = np.array([np.ptp(row[np.isfinite(row)]) if np.isfinite(row).any() else 0 for row in headings])
    return dict(forward_displacement_m=trajectories[:, -1, 1], mean_speed_mps=speed.mean(axis=1),
        speed_std_mps=speed.std(axis=1), speed_change_mps=speed[:, -5:].mean(axis=1)-speed[:, :5].mean(axis=1),
        final_lateral_m=trajectories[:, -1, 0], max_lateral_m=np.abs(trajectories[..., 0]).max(axis=1),
        heading_change_deg=np.degrees(heading_range), max_curvature_inv_m=curvature.max(axis=1))


def stratified_candidates(construction_trajectories, construction_indices, count=1000, seed=7):
    """Only accepts construction data. Round-robin behavior+quantile cells."""
    x = np.asarray(construction_trajectories); indices = np.asarray(construction_indices)
    if len(x) != len(indices) or len(np.unique(indices)) != len(indices) or not np.isfinite(x).all():
        raise ValueError('Invalid construction-only candidate input')
    d = candidate_descriptors(x)
    speed = d['mean_speed_mps']; lateral = d['final_lateral_m']
    behavior = np.full(len(x), 'other', dtype='<U32')
    behavior[speed < .2] = 'stationary'
    moving = speed >= .2
    straight = moving & (d['max_lateral_m'] < .5) & (d['heading_change_deg'] < 5)
    speed_edges = np.quantile(speed[moving], [1/3, 2/3]) if moving.any() else np.array([0, 0])
    for i, label in enumerate(['straight_slow', 'straight_medium', 'straight_fast']):
        behavior[straight & (np.digitize(speed, speed_edges)==i)] = label
    behavior[moving & (d['speed_change_mps'] > 1)] = 'accelerating'
    behavior[moving & (d['speed_change_mps'] < -1)] = 'decelerating'
    # Sign labels use X sign, without assuming undocumented left/right handedness.
    turning = moving & (d['heading_change_deg'] >= 10) & (np.abs(lateral) >= 1)
    behavior[turning & (lateral < 0)] = 'turn_negative_X'
    behavior[turning & (lateral > 0)] = 'turn_positive_X'
    side = moving & (d['max_lateral_m'] >= 1) & (d['heading_change_deg'] < 10)
    behavior[side & (lateral < 0)] = 'lateral_negative_X'
    behavior[side & (lateral >= 0)] = 'lateral_positive_X'
    quantiles = {k: np.quantile(v, [.01,.25,.5,.75,.99]) for k,v in d.items()}
    bins = np.column_stack([np.digitize(d[k], quantiles[k]) for k in sorted(d)])
    tail = np.any((bins == 0) | (bins == 5), axis=1)
    cells = {}
    for i in range(len(x)):
        key = (str(behavior[i]), bool(tail[i]), *bins[i].tolist())
        cells.setdefault(key, []).append(i)
    rng = np.random.default_rng(seed)
    groups = {}
    for key in sorted(cells):
        bucket = (key[0],key[1]); groups.setdefault(bucket, []).append(rng.permutation(cells[key]).tolist())
    for bucket in groups:
        groups[bucket] = [groups[bucket][j] for j in rng.permutation(len(groups[bucket]))]
    chosen = []
    active = sorted(groups)
    # Round-robin by behavior/tail and then descriptor quantile cells; prefixes remain diverse.
    while len(chosen) < min(count, len(x)) and active:
        next_active = []
        for bucket in active:
            cell = groups[bucket].pop(0)
            chosen.append(cell.pop())
            if cell: groups[bucket].append(cell)
            if groups[bucket]: next_active.append(bucket)
            if len(chosen) == min(count, len(x)): break
        active = next_active
    chosen = np.array(chosen, dtype=np.int64)
    descriptions = [dict(source_index=int(indices[i]), behavior=str(behavior[i]), tail=bool(tail[i]),
                         **{k:float(v[i]) for k,v in d.items()}) for i in chosen]
    from collections import Counter
    summary = dict(seed=seed, construction_count=len(x), selected_count=len(chosen),
        descriptor_quantiles={k:v.tolist() for k,v in quantiles.items()}, straight_speed_tertiles_mps=speed_edges.tolist(),
        available_behavior_counts=dict(Counter(behavior.tolist())), selected_behavior_counts=dict(Counter(behavior[chosen].tolist())),
        selected_tail_count=int(tail[chosen].sum()), cell_count=len(cells),
        available_stratum_counts=dict(Counter(f'{b}|tail={bool(t)}' for b,t in zip(behavior,tail))),
        selected_stratum_counts=dict(Counter(f'{behavior[i]}|tail={bool(tail[i])}' for i in chosen)),
        method='Round-robin behavior/tail strata and descriptor quantile cells; all quantiles from construction only; no heldout input',
        sign_convention='+Y forward, X lateral; positive/negative X labels used explicitly')
    return indices[chosen], descriptions, summary
