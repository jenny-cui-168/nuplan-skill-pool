"""Diagnostics retain failed iterates and distinguish them from converged Logs."""
from pathlib import Path
import json
import numpy as np
from .splits import write_json, sha256


def vector_comparison(local, geodesic, metric):
    local=np.asarray(local); geodesic=np.asarray(geodesic)
    def norm(x): return np.sqrt(np.maximum(np.einsum('ni,ij,nj->n',x,metric,x),0))
    a,b=norm(local),norm(geodesic)
    finite=np.isfinite(local).all(axis=1)&np.isfinite(geodesic).all(axis=1)
    valid=finite&(a>1e-10)&(b>1e-10)
    angle=np.full(len(local),np.nan); ratio=np.full(len(local),np.nan); error=np.full(len(local),np.nan)
    dot=np.einsum('ni,ij,nj->n',local,metric,geodesic)
    angle[valid]=np.degrees(np.arccos(np.clip(dot[valid]/(a[valid]*b[valid]),-1,1)))
    ratio[finite&(a>1e-10)]=b[finite&(a>1e-10)]/a[finite&(a>1e-10)]
    error[finite&(a>1e-10)]=norm(geodesic-local)[finite&(a>1e-10)]/a[finite&(a>1e-10)]
    return dict(angle_deg=angle,strength_ratio=ratio,relative_error=error)


def statistics(values):
    values=np.asarray(values); values=values[np.isfinite(values)]
    return dict(count=len(values),mean=float(values.mean()) if len(values) else None,
                p50=float(np.quantile(values,.5)) if len(values) else None,
                p90=float(np.quantile(values,.9)) if len(values) else None,
                max=float(values.max()) if len(values) else None)


def read_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def analyze(root='log_experiment'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root)
    for name in ['stage_100','sensitivity_L16','sensitivity_damping_1e3','sensitivity_iterations50']:
        if not (root/name/'summary.json').exists():
            raise RuntimeError(f'Missing completed fixed-100 run: {name}')
    config=json.loads((root/'experiment_config.json').read_text())
    descriptions=json.loads((root/'candidate_descriptors.json').read_text())
    desc={r['source_index']:r for r in descriptions}
    ids=np.load(root/'evaluated_indices.npy')
    all_ids=np.load(root/'candidate_indices.npy')
    latent_map={int(i):z for i,z in zip(all_ids,np.load(root/'candidate_latents.npy'))}
    z0=np.load('outputs/neutral_construction/z0.npy');metric=np.load('outputs/neutral_construction/G0.npy')
    local=np.array([latent_map[int(i)].astype(np.float64)-z0.astype(np.float64) for i in ids])
    np.save(root/'local_vectors.npy', np.load(root/'candidate_latents.npy').astype(np.float64)-z0.astype(np.float64))
    vectors=np.load(root/'log_vectors.npy'); records=read_records(root/'solver_records.jsonl')
    attempted={r['source_index']:r for r in records}
    with (root/'candidate_status.jsonl').open('w') as stream:
        for source in all_ids:
            row=attempted.get(int(source))
            status='not_attempted' if row is None else 'converged_discrete' if row['converged'] else 'failed_provisional'
            stream.write(json.dumps(dict(source_index=int(source),status=status))+'\n')
    converged=np.array([r['converged'] for r in records])
    values=vector_comparison(local,vectors,metric)
    result=dict(metric='fixed construction G0 for angles, strengths and relative error',
        interpretation='Nonconverged iterates are provisional; differences do not establish properties of a strict continuous Log.',
        candidate_count=len(all_ids),evaluated_count=len(ids),not_evaluated_count=len(all_ids)-len(ids),converged_count=int(converged.sum()),
        all_attempted={k:statistics(v) for k,v in values.items()},
        converged_only={k:statistics(v[converged]) for k,v in values.items()},by_behavior={},samples=[])
    behavior=np.array([desc[int(i)]['behavior'] for i in ids])
    for label in sorted(set(behavior)):
        mask=behavior==label
        result['by_behavior'][label]=dict(attempted_count=int(mask.sum()),converged_count=int((mask&converged).sum()),
            all_attempted={k:statistics(v[mask]) for k,v in values.items()},
            converged_only={k:statistics(v[mask&converged]) for k,v in values.items()})
    for row,source in enumerate(ids):
        result['samples'].append(dict(source_index=int(source),behavior=str(behavior[row]),converged=bool(converged[row]),
            **{k:float(v[row]) if np.isfinite(v[row]) else None for k,v in values.items()}))
    finite=np.flatnonzero(np.isfinite(values['relative_error']))
    largest=finite[np.argsort(-values['relative_error'][finite],kind='stable')[:20]]
    result['largest_difference_20']=[result['samples'][i] for i in largest]
    write_json(root/'local_vs_geodesic.json',result)
    stages={};sensitivity={}
    for directory in sorted(root.iterdir()):
        if directory.is_dir() and (directory/'summary.json').exists():
            summary=json.loads((directory/'summary.json').read_text())
            if directory.name.startswith('stage_'): stages[directory.name]=summary
            elif directory.name.startswith('sensitivity_'): sensitivity[directory.name]=summary
    comparison={}
    base=np.load(root/'stage_100/log_vectors.npy')
    base_records=read_records(root/'stage_100/solver_records.jsonl')
    for name,summary in sensitivity.items():
        alternative=np.load(root/name/'log_vectors.npy')
        comparison[name]=dict(config=summary['config'],convergence_rate=summary['convergence_rate'],
            versus_L8_baseline={k:statistics(v) for k,v in vector_comparison(base,alternative,metric).items()},
            joint_converged_count=sum(a['converged'] and b['converged'] for a,b in zip(base_records,read_records(root/name/'solver_records.jsonl'))),
            interpretation='Configuration differences include nonconverged iterates; no discretization-convergence claim')
    write_json(root/'sensitivity_comparison.json',comparison)
    main=stages['stage_100']
    reasons=[]
    if main['convergence_rate']<config['minimum_convergence_rate']: reasons.append('100-sample convergence below 95%')
    if 1000*main['runtime_mean_seconds']>config['max_projected_1000_seconds']: reasons.append('1000-sample projection exceeds predeclared 3600-second budget')
    if not main['numerical_checks_passed']: reasons.append('100-sample numerical checks failed')
    projection={name:{str(n):float(s['runtime_mean_seconds']*n) for n in [1000,10000,20000]} for name,s in {'baseline_L8':main,**sensitivity}.items()}
    preserved=all(sha256(Path('outputs/v1')/name)==digest and sha256(Path('outputs/local_baseline')/name)==digest for name,digest in config['local_baseline_sha256'].items())
    summary=dict(base_commit=config['base_commit'],candidate_count=len(all_ids),evaluated_count=len(ids),not_evaluated_count=len(all_ids)-len(ids),
        stages=stages,sensitivity=sensitivity,expansion_to_1000_allowed=not reasons,expansion_stop_reasons=reasons,
        projection_seconds=projection,projection_method='Mean attempted-source runtime times count; baseline100 measured serially, sensitivity configurations concurrently (contention-dependent); includes failed solves, not actual large runs',
        local_baseline_bitwise_preserved=preserved,checkpoint_unchanged=sha256('weights/trajectory_vae_8d_best.pth')==config['checkpoint_sha256'],
        conclusion='Not validated as stable strict Log; do not replace pools' if reasons else 'Numerical gates passed; resolution and nonsmoothness limitations remain',
        nonsmoothness='Frozen ReLU decoder metric is piecewise smooth, with activation-boundary jumps. LBFGS stationarity and quadrature refinement are not guaranteed.',
        cuda_memory='CUDA unavailable; CPU RSS measured. No GPU stability claim.',
        failed_rows_retained=True,
        baseline_energy_statistics=dict(
            strict_decrease_count=sum(r['final_energy'] < r['initial_energy']*(1-1e-8) for r in base_records),
            relative_reduction=statistics([(r['initial_energy']-r['final_energy'])/max(r['initial_energy'],1e-12) for r in base_records]),
            gradient_residual=statistics([r['gradient_residual'] for r in base_records]),
            quadrature_relative_gap=statistics([r['quadrature_relative_gap'] for r in base_records])))
    write_json(root/'convergence_summary.json',summary)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for record in records:
        curve=np.array(record['energy_curve']); scale=max(record['initial_energy'] or 0,1e-12)
        axes[0].plot(np.arange(len(curve)),curve/scale,alpha=.25)
    axes[0].set(xlabel='Optimizer iterations',ylabel='Energy / initial energy',title='All attempted paths (not convergence certification)')
    groups={'L8':main,**sensitivity}
    axes[1].bar(np.arange(len(groups)),[s['convergence_rate'] for s in groups.values()])
    axes[1].axhline(.95,color='red',linestyle='--',label='95% gate')
    axes[1].set(xticks=np.arange(len(groups)),xticklabels=list(groups),ylabel='Converged fraction',ylim=(0,1)); axes[1].tick_params(axis='x',rotation=25);axes[1].legend()
    fig.tight_layout();fig.savefig(root/'energy_curves.png',dpi=160);plt.close(fig)
    for metric_name,filename in [('angle_deg','angle_error_hist.png'),('strength_ratio','strength_ratio_hist.png')]:
        fig,ax=plt.subplots(figsize=(9,5));v=values[metric_name]
        ax.hist(v[np.isfinite(v)&~converged],bins=25,alpha=.65,label='Nonconverged provisional')
        if converged.any(): ax.hist(v[np.isfinite(v)&converged],bins=25,alpha=.65,label='Converged discrete')
        ax.set(xlabel=metric_name,ylabel='Candidate count',title='Versus local z-z0, fixed G0');ax.legend();fig.tight_layout();fig.savefig(root/filename,dpi=160);plt.close(fig)
    # Decode only the evaluated construction targets for illustration.
    from .model import load_vae
    from .data import decode_latents
    model=load_vae('weights/trajectory_vae_8d_best.pth')
    decoded=decode_latents(model,np.array([latent_map[int(i)] for i in ids]),1024,'cpu')
    raw=np.load('data/ego_trajs.npy',mmap_mode='r')
    fig,axes=plt.subplots(4,5,figsize=(20,18))
    for ax,position in zip(axes.flat,largest):
        source=int(ids[position]);v=result['samples'][position]
        for trajectory,label in [(raw[source],'Source'),(decoded[position],'D(target)')]:
            ax.plot(trajectory[:,0],trajectory[:,1],label=label)
        ax.set_title(f"row {source}: {v['behavior']}\nangle={v['angle_deg']:.1f}, ratio={v['strength_ratio']:.2f}\nrel.err={v['relative_error']:.2f}; converged={v['converged']}",fontsize=8)
        ax.set(xlabel='X lateral (m)',ylabel='Y forward (m)');ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=7);ax.grid(alpha=.25)
    for ax in list(axes.flat)[len(largest):]: ax.set_visible(False)
    fig.suptitle('Largest differences: provisional failed iterates must not be treated as validated Logs')
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(root/'largest_difference_examples.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6))
    for name,p in projection.items(): ax.plot([int(n) for n in p],[v/3600 for v in p.values()],marker='o',label=name)
    ax.set(xlabel='Number of sources (projection only)',ylabel='Hours at observed per-source rate',title='Fixed 100 sources: serial baseline; concurrent sensitivity timings');ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(root/'runtime_scaling.png',dpi=160);plt.close(fig)
    return summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='log_experiment')
    print(json.dumps(analyze(parser.parse_args().root),indent=2))
