"""Staged, bounded feasibility experiment. Never constructs or modifies pools."""
from pathlib import Path
from dataclasses import asdict
import argparse
import json
import shutil
import time
import numpy as np
import torch
from .geodesic import SolverConfig, solve_log, stratified_candidates
from .model import load_vae
from .splits import sha256, write_json


def prepare(root='log_experiment'):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    if (root/'candidate_indices.npy').exists():
        raise FileExistsError('Candidate experiment already exists; use a new root to preserve results')
    c = np.load('splits/construction_indices.npy')
    mapping = np.load('outputs/v1/construction_indices.npy')
    if not np.array_equal(c, mapping): raise ValueError('Construction latent index mismatch')
    raw = np.load('data/ego_trajs.npy', mmap_mode='r')
    selected, descriptors, report = stratified_candidates(raw[c], c, count=1000, seed=7)
    np.save(root/'candidate_indices.npy', selected)
    np.save(root/'candidate_latents.npy', np.load('outputs/v1/construction_latents.npy')[np.searchsorted(c,selected)])
    write_json(root/'candidate_descriptors.json', descriptors)
    write_json(root/'candidate_summary.json', report)
    baseline = Path('outputs/local_baseline')
    if not baseline.exists(): shutil.copytree('outputs/v1', baseline)
    hashes = {p.name:sha256(p) for p in Path('outputs/v1').iterdir() if p.is_file()}
    for name, digest in hashes.items():
        if sha256(baseline/name) != digest: raise ValueError('local_baseline differs from V1')
    write_json(root/'experiment_config.json', dict(base_commit='75e6e5e', selection_split='construction',
        baseline_solver=asdict(SolverConfig()), minimum_convergence_rate=.95, max_projected_1000_seconds=3600,
        maximum_stage_rss_range_mb=512, device='cpu', cuda_available=torch.cuda.is_available(),
        local_baseline_sha256=hashes, checkpoint_sha256=sha256('weights/trajectory_vae_8d_best.pth'),
        candidate_indices_sha256=sha256(root/'candidate_indices.npy'),
        interpretation='Numerical variational discrete Log estimate L*(p1-p0); not exact continuous Log. Frozen ReLU decoder induces nonsmooth metric. No heldout information used.',
        convergence='Infinity norm of gradient of E/max(E_initial,1e-12) w.r.t. internal latent nodes <=1e-4. Energy decrease or optimizer stall alone is not convergence.',
        references=['https://arxiv.org/abs/1210.2097']))
    return report


def run_batch(root, name, n, config):
    root = Path(root); out = root/name; out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists(): raise ValueError(f'Existing completed run: {out}; preserve results')
    ids = np.load(root/'candidate_indices.npy')[:n]
    targets = np.load(root/'candidate_latents.npy')[:n]
    z0 = np.load('outputs/neutral_construction/z0.npy')
    model = load_vae('weights/trajectory_vae_8d_best.pth','cpu').double()
    original = {k:v.detach().clone() for k,v in model.state_dict().items()}
    logs, paths, records = [], [], []
    started = time.perf_counter()
    with (out/'solver_records.jsonl').open('w') as stream:
        for i, (source, target) in enumerate(zip(ids,targets)):
            log, path, record = solve_log(model,z0,target,int(source),config)
            records.append(record); logs.append(log); paths.append(path)
            stream.write(json.dumps(record,allow_nan=False)+'\n'); stream.flush()
            if (i+1)%10==0 or n==1:
                print(f'{name}: {i+1}/{n}, converged={sum(r["converged"] for r in records)}, elapsed={time.perf_counter()-started:.1f}s',flush=True)
    elapsed = time.perf_counter()-started
    unchanged = all(torch.equal(v,original[k]) for k,v in model.state_dict().items())
    np.save(out/'log_vectors.npy', np.array(logs)); np.save(out/'paths.npy',np.array(paths)); np.save(out/'evaluated_indices.npy',ids)
    rss = [r['rss_after_mb'] for r in records]
    summary = dict(name=name, count=len(ids), config=asdict(config), convergence_count=sum(r['converged'] for r in records),
        convergence_rate=float(np.mean([r['converged'] for r in records])), all_finite=bool(np.isfinite(logs).all() and np.isfinite(paths).all()),
        all_endpoints_fixed=all(r['endpoints_fixed'] for r in records), all_energy_nonincreasing=all(r['energy_nonincreasing'] for r in records),
        decoder_weights_bitwise_unchanged=unchanged, elapsed_seconds=elapsed,
        runtime_mean_seconds=float(np.mean([r['runtime_seconds'] for r in records])), runtime_p90_seconds=float(np.quantile([r['runtime_seconds'] for r in records],.9)),
        rss_min_mb=min(rss),rss_max_mb=max(rss),rss_range_mb=max(rss)-min(rss),
        rss_first_last_change_mb=rss[-1]-rss[0], cuda_memory_measured=False,
        quadrature_gap_p90=float(np.quantile([r.get('quadrature_relative_gap',0) for r in records],.9)),
        source_indices=ids.tolist(), failure_count=sum(not r['converged'] for r in records))
    summary['numerical_checks_passed'] = summary['all_finite'] and summary['all_endpoints_fixed'] and summary['all_energy_nonincreasing'] and unchanged and summary['rss_range_mb']<=512
    write_json(out/'summary.json',summary)
    return summary


def run_stage(root,n):
    root=Path(root)
    if n not in [1,10,100,1000]: raise ValueError('Only 1,10,100,1000 stages permitted')
    config=json.loads((root/'experiment_config.json').read_text())
    if n>1:
        previous={10:1,100:10,1000:100}[n]
        summary=json.loads((root/f'stage_{previous}/summary.json').read_text())
        if not summary['numerical_checks_passed']: raise RuntimeError('Prior stage numerical checks failed; expansion stopped')
        if n==1000 and (summary['convergence_rate']<config['minimum_convergence_rate'] or 1000*summary['runtime_mean_seconds']>config['max_projected_1000_seconds']):
            raise RuntimeError('100-sample convergence/runtime gate failed; no 1000-sample expansion')
    result=run_batch(root,f'stage_{n}',n,SolverConfig(**config['baseline_solver']))
    for name in ['log_vectors.npy','solver_records.jsonl','evaluated_indices.npy','paths.npy']:
        shutil.copy2(root/f'stage_{n}'/name, root/name)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','stage','sensitivity'])
    parser.add_argument('--root',default='log_experiment')
    parser.add_argument('--count',type=int)
    parser.add_argument('--variant',choices=['L16','damping','iterations'])
    args=parser.parse_args()
    if args.action=='prepare': result=prepare(args.root)
    elif args.action=='stage': result=run_stage(args.root,args.count)
    else:
        root=Path(args.root)
        if not (root/'stage_100/summary.json').exists(): raise RuntimeError('Run fixed 100 stage before sensitivity')
        result={}
        variants={'L16':('sensitivity_L16',SolverConfig(segments=16)),
                  'damping':('sensitivity_damping_1e3',SolverConfig(damping=1e-3)),
                  'iterations':('sensitivity_iterations50',SolverConfig(max_iterations=50))}
        for name, config in ([variants[args.variant]] if args.variant else variants.values()):
            if (root/name/'summary.json').exists():
                result[name]=json.loads((root/name/'summary.json').read_text())
            else: result[name]=run_batch(root,name,100,config)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
