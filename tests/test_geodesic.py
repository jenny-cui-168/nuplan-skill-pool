from dataclasses import replace
import numpy as np
import pytest
import torch
from skill_pool.geodesic import (SolverConfig, path_energy, pullback_metrics, solve_log,
                                 stratified_candidates)


class AffineDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(60*8, dtype=torch.float64).reshape(60,8)/1000, requires_grad=False)
    def decode(self, z): return (z @ self.weight.T).reshape(-1,30,2)


class SmoothDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(8, dtype=torch.float64), requires_grad=False)
    def decode(self, z): return torch.cat((torch.sin(z)*self.scale, z**2),dim=-1)


def test_constant_metric_log_energy_and_zero():
    model=AffineDecoder(); z0=np.zeros(8); end=np.arange(8)/10
    config=SolverConfig(segments=4,max_iterations=10)
    log,path,record=solve_log(model,z0,end,123,config)
    assert record['converged'] and record['endpoints_fixed']
    np.testing.assert_allclose(log,end,atol=1e-12)
    np.testing.assert_array_equal(path[0],z0); np.testing.assert_array_equal(path[-1],end)
    metric=model.weight.T@model.weight/60+config.damping*torch.eye(8)
    expected=.5*end@metric.numpy()@end
    assert record['initial_energy']==pytest.approx(expected,rel=1e-10)
    zero,_,r=solve_log(model,z0,z0,9,config)
    assert np.linalg.norm(zero)<1e-12 and r['converged']


def test_energy_gradient_includes_metric_derivative():
    model=SmoothDecoder(); p=torch.tensor([[.1]*8,[.4]*8,[.8]*8],dtype=torch.float64,requires_grad=True)
    energy=path_energy(model,p)
    gradient=torch.autograd.grad(energy,p)[0]
    eps=1e-6
    plus=p.detach().clone();minus=p.detach().clone();plus[1,2]+=eps;minus[1,2]-=eps
    difference=(path_energy(model,plus)-path_energy(model,minus))/(2*eps)
    torch.testing.assert_close(gradient[1,2],difference,rtol=1e-5,atol=1e-7)
    metric=pullback_metrics(model,p.detach(),1e-4)
    assert not torch.allclose(metric[0],metric[-1])


def test_nonlinear_energy_finite_fixed_weights_and_reproducible():
    model=SmoothDecoder(); before={k:v.clone() for k,v in model.state_dict().items()}
    start=np.zeros(8);end=np.ones(8)
    config=SolverConfig(segments=4,max_iterations=40,gradient_tolerance=1e-5)
    a,p,r=solve_log(model,start,end,42,config)
    b,q,s=solve_log(model,start,end,42,config)
    assert r['finite'] and r['endpoints_fixed'] and r['energy_nonincreasing']
    assert r['final_energy']<=r['initial_energy']
    assert np.isfinite(a).all() and np.isfinite(p).all()
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(p,q)
    assert r['converged']==s['converged']
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())


def test_failure_keeps_original_source_and_traceback():
    _,_,r=solve_log(AffineDecoder(),np.zeros(8),np.full(8,np.nan),9876)
    assert not r['converged'] and r['source_index']==9876
    assert 'Traceback' in r['traceback'] and 'Nonfinite' in r['failure_reason']
    _,p,r=solve_log(SmoothDecoder(),np.zeros(8),np.ones(8),222,
                   SolverConfig(max_iterations=1,gradient_tolerance=1e-14))
    assert not r['converged'] and r['source_index']==222 and r['traceback']
    assert np.isfinite(p).all()


def test_construction_only_sampling_and_heldout_independence():
    rng=np.random.default_rng(7)
    all_data=np.cumsum(rng.normal(size=(150,30,2)),axis=1)
    construction=np.arange(0,150,2); heldout=np.arange(1,150,2)
    a,da,_=stratified_candidates(all_data[construction],construction,count=30)
    # Solver only accepts endpoints; heldout cannot enter metric or optimization.
    model=AffineDecoder(); target=all_data[a[0]].reshape(-1)[:8]
    va,pa,ra=solve_log(model,np.zeros(8),target,int(a[0]))
    all_data[heldout]=1e9
    b,db,_=stratified_candidates(all_data[construction],construction,count=30)
    vb,pb,rb=solve_log(model,np.zeros(8),all_data[b[0]].reshape(-1)[:8],int(b[0]))
    np.testing.assert_array_equal(a,b); assert da==db
    assert np.isin(a,construction).all() and not np.isin(a,heldout).any()
    np.testing.assert_array_equal(va,vb);np.testing.assert_array_equal(pa,pb)


def test_g0_angle_ratio_and_zero_denominator():
    from skill_pool.log_diagnostics import vector_comparison
    local=np.array([[1.,0.],[0.,1.],[0.,0.]])
    geo=np.array([[2.,0.],[1.,0.],[0.,0.]])
    result=vector_comparison(local,geo,np.diag([4.,1.]))
    np.testing.assert_allclose(result['angle_deg'][:2],[0,90])
    np.testing.assert_allclose(result['strength_ratio'][:2],[2,2])
    assert np.isnan(result['angle_deg'][2])


def test_formal_candidates_only_construction_and_baseline_preserved():
    from pathlib import Path
    import json
    from skill_pool.splits import sha256
    root=Path(__file__).resolve().parents[1]
    ids=np.load(root/'log_experiment/candidate_indices.npy')
    construction=np.load(root/'splits/construction_indices.npy')
    heldout=np.load(root/'splits/heldout_indices.npy')
    assert len(ids)==len(np.unique(ids))==1000
    assert np.isin(ids,construction).all() and not np.isin(ids,heldout).any()
    descriptors=json.loads((root/'log_experiment/candidate_descriptors.json').read_text())
    assert [r['source_index'] for r in descriptors]==ids.tolist()
    assert {'stationary','straight_slow','straight_medium','straight_fast','accelerating','decelerating','turn_negative_X','turn_positive_X','lateral_negative_X','lateral_positive_X'} <= {r['behavior'] for r in descriptors}
    assert any(r['tail'] for r in descriptors)
    config=json.loads((root/'log_experiment/experiment_config.json').read_text())
    for name,expected in config['local_baseline_sha256'].items():
        assert sha256(root/'outputs/v1'/name)==sha256(root/'outputs/local_baseline'/name)==expected


def test_1000_gate_rejects_low_convergence(tmp_path):
    from skill_pool.log_experiment import run_stage
    from skill_pool.splits import write_json
    write_json(tmp_path/'experiment_config.json',dict(minimum_convergence_rate=.95,max_projected_1000_seconds=3600))
    (tmp_path/'stage_100').mkdir()
    write_json(tmp_path/'stage_100/summary.json',dict(numerical_checks_passed=True,convergence_rate=.94,runtime_mean_seconds=.1))
    with pytest.raises(RuntimeError,match='gate failed'):
        run_stage(tmp_path,1000)
    assert not (tmp_path/'stage_1000').exists()


def test_real_experiment_records_and_artifacts():
    from pathlib import Path
    import json
    root=Path(__file__).resolve().parents[1]/'log_experiment'
    summary=json.loads((root/'convergence_summary.json').read_text())
    ids=np.load(root/'evaluated_indices.npy'); logs=np.load(root/'log_vectors.npy');paths=np.load(root/'paths.npy')
    rows=[json.loads(x) for x in (root/'solver_records.jsonl').read_text().splitlines()]
    assert len(rows)==len(ids)==len(logs)==summary['evaluated_count']
    assert [r['source_index'] for r in rows]==ids.tolist()
    assert np.isfinite(logs).all() and np.isfinite(paths).all()
    for row in rows:
        assert row['endpoints_fixed'] and row['energy_nonincreasing']
        assert row['final_energy']<=row['initial_energy']+1e-9
        if not row['converged']: assert 'Traceback' in row['traceback'] and row['failure_reason']
    for stage in summary['stages'].values(): assert stage['decoder_weights_bitwise_unchanged']
    for variant in summary['sensitivity'].values():
        assert variant['source_indices']==np.load(root/'candidate_indices.npy')[:100].tolist()
        assert variant['count']==100 and variant['decoder_weights_bitwise_unchanged']
    for name in ['energy_curves','angle_error_hist','strength_ratio_hist','largest_difference_examples','runtime_scaling']:
        assert (root/f'{name}.png').read_bytes().startswith(b'\x89PNG')
    if not summary['expansion_to_1000_allowed']:
        assert not (root/'stage_1000').exists()
