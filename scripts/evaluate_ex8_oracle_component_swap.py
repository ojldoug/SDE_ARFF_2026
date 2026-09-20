"""Evaluation-only oracle swaps; no training entry points or GPU access."""
import os
os.environ.update(JAX_PLATFORMS='cpu', JAX_ENABLE_X64='false', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MPLBACKEND='Agg')
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
os.nice(19)
from pathlib import Path
import sys, json, csv, hashlib
import numpy as np
import jax
import jax.numpy as jnp
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments.dataset import load_dataset
from src.experiments.definitions import get_experiment
from src.arff.covariance import project_spd, raw_covariance
from src.arff.regression import predict
from scripts.run_ex8_arff_validation_selected_crossfit import reconstruct_model, SPD_EPSILON

BASE = ROOT / 'results/controlled_study_2026/float64_v2'
HYB = BASE / 'hybrid_jointmlp_arff'
OUT = BASE / 'oracle_component_swap/evaluation_v2'
FLOORS = [1e-8, 1e-6, 1e-4, 1e-3]
CONSTANT = float(np.log(2*np.pi))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def stats(v):
    a = np.asarray(v, dtype=float)
    assert np.isfinite(a).all()
    return dict(n=a.size, mean=float(a.mean()), sd=float(a.std(ddof=1)) if a.size > 1 else None,
                median=float(np.median(a)), min=float(a.min()), max=float(a.max()))

def write_json(p, value):
    with p.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)

def native_parts(f, covariance, r, h):
    """Exact accepted float32 solve/logdet/reduction; covariance already projected."""
    residual = r-h*jnp.asarray(f)
    variance = covariance*h[:, :, None]
    sign, logdet = jnp.linalg.slogdet(variance)
    assert bool(jnp.all(sign > 0))
    sol = jnp.linalg.solve(variance, residual[:, :, None])[:, :, 0]
    quadratic = jnp.sum(residual*sol, axis=1)
    constant = r.shape[1]*jnp.log(2*jnp.pi)/2
    return dict(quadratic=float(jnp.mean(quadratic/2)), logdet=float(jnp.mean(logdet/2)),
                constant=float(constant), nll=float(jnp.mean(.5*(quadratic+logdet+2*constant))))

def spectral_parts(f, eig, q, r, h):
    """Same Gaussian convention in float64 spectral coordinates, stable at 1e-8."""
    residual = np.asarray(r, np.float64)-np.asarray(h, np.float64)*np.asarray(f, np.float64)
    rotated = np.einsum('nji,nj->ni', q, residual)
    quadratic = .5*np.sum(rotated**2/(h*eig), axis=1)
    logdet = .5*np.sum(np.log(h*eig), axis=1)
    return dict(quadratic=float(quadratic.mean()), logdet=float(logdet.mean()),
                constant=CONSTANT, nll=float((quadratic+logdet+CONSTANT).mean()))

def rmse(a, b):
    return float(np.sqrt(np.mean((a-b)**2)))

def main():
    OUT.mkdir(exist_ok=False)
    assert jax.default_backend() == 'cpu' and SPD_EPSILON == .001
    prior = json.loads((HYB/'evaluation_v2/summary.json').read_text())
    protected = dict(prior['protected_sha256'])
    for p in HYB.rglob('*'):
        if p.is_file():
            protected[str(p)] = sha(p)
    for p, digest in protected.items():
        assert sha(p) == digest, p
    write_json(OUT/'protocol.json', dict(seeds=list(range(30)), floors=FLOORS, accepted_floor=.001,
        task='Post-hoc evaluation only; no floor selection', script_sha256=sha(__file__),
        precision='Native float32 accepted-floor controls; float64 oracle and spectral sensitivity. Model predictions unchanged.',
        protected_sha256=protected))
    d = load_dataset(ROOT/'data/ex8_float64_v2.npz')
    dataset_sha = sha(ROOT/'data/ex8_float64_v2.npz')
    assert dataset_sha == '6fb009e3d6fa5f241cb1a15f9a9815cbd16ea19c3a9e4ca546ae99bdae091a72'
    saved = np.load(HYB/'evaluation_v2/test_predictions.npz')
    np.testing.assert_array_equal(saved['seeds'], np.arange(30))
    np.testing.assert_array_equal(saved['test_idx'], d.test_idx)
    assert len(d.test_idx) == 10000
    previous = list(csv.DictReader((HYB/'per_seed.csv').open()))
    comparator = json.loads((HYB/'same_backend_comparisons.json').read_text())['per_seed']
    native_arff = {v['seed']: v['comparator_cpu_nll'] for v in comparator if v['method']=='arff'}
    x, r32, h32 = [jnp.asarray(v[d.test_idx]) for v in [d.x, d.r, d.h]]
    # Use archived float64 increments for stable mathematical oracle/sensitivity;
    # native controls retain the accepted float32 input conversion.
    r64, h64 = [np.asarray(v[d.test_idx], np.float64) for v in [d.r, d.h]]
    definition = get_experiment('ex8')
    with jax.enable_x64():
        x64 = jnp.asarray(d.x[d.test_idx], dtype=jnp.float64)
        ftrue = np.asarray(definition.drift(x64))
        sigma = np.asarray(definition.diffusion_factor(x64))
    truecov = sigma @ sigma.swapaxes(1, 2)
    te, tq = np.linalg.eigh(truecov)
    np.testing.assert_allclose(te[:, 0], 1e-8, rtol=1e-7, atol=2e-16)
    np.testing.assert_allclose(te[:, 1], 1, rtol=1e-14)
    oracle = spectral_parts(ftrue, te, tq, r64, h64)
    controls = {}
    for floor in FLOORS:
        e = np.maximum(te, floor)
        c = (tq*e[:, None, :])@tq.swapaxes(1, 2)
        controls[str(floor)] = dict(**spectral_parts(ftrue, e, tq, r64, h64), projected_covariance_rmse=rmse(c, truecov))
    rows, projection_rows, sensitivity, checks = [], [], [], []
    def add(seed, combination, scope, parts):
        rows.append(dict(seed=seed, combination=combination, scope=scope, **parts))
    for seed in range(30):
        a_path = BASE/f'production/baseline/K_128_N_80000/arff/seed_{seed}/artifact.npz'
        with np.load(a_path, allow_pickle=False) as z:
            a = {k:z[k] for k in z.files}
        assert int(a['seed']) == int(previous[seed]['seed']) == seed
        assert float(a['spd_epsilon']) == SPD_EPSILON
        model = reconstruct_model(a)
        fa = np.asarray(predict(model.drift, x))
        fm, raw = saved['drift'][seed], saved['raw_covariance'][seed]
        np.testing.assert_array_equal(raw, np.asarray(raw_covariance(model.covariance, x, model.diff_type)))
        projected = project_spd(jnp.asarray(raw), epsilon=SPD_EPSILON)
        # Decomposition requires residual evaluation, but the hybrid total itself
        # is reused, with an independent exact-native integrity check.
        hp = native_parts(fm, projected, r32, h32)
        assert hp['nll'] == float(previous[seed]['nll'])
        hp['nll'] = float(previous[seed]['nll'])
        ap = native_parts(fa, projected, r32, h32)
        assert ap['nll'] == native_arff[seed]
        for name, parts in [('hybrid', hp), ('arff_original', ap),
                            ('true_drift_arff_covariance', native_parts(ftrue, projected, r32, h32))]:
            add(seed, name, 'native_float32_floor_1e-3', parts)
        for name, drift in [('oracle', ftrue), ('joint_mlp_drift_true_covariance', fm), ('arff_drift_true_covariance', fa)]:
            add(seed, name, 'float64_unprojected_true_covariance', oracle if name=='oracle' else spectral_parts(drift, te, tq, r64, h64))
        er, qr = np.linalg.eigh(np.asarray(raw, np.float64))
        change = np.maximum(er, .001)-er
        diag = dict(seed=seed, raw_covariance_rmse=rmse(raw, truecov),
                    projected_covariance_rmse=rmse(np.asarray(projected, np.float64), truecov),
                    raw_spd_violation_rate=float(np.mean(er[:, 0]<=0)), min_raw_eigenvalue=float(er.min()),
                    eigenvalue_fraction_changed=float(np.mean(change>0)), eigenvalue_change_mean=float(change.mean()),
                    changed_only_mean=float(change[change>0].mean()))
        for quantile in [0, .25, .5, .75, .9, .95, .99, 1]:
            diag[f'change_q{quantile:g}'] = float(np.quantile(change, quantile))
            diag[f'changed_only_q{quantile:g}'] = float(np.quantile(change[change>0], quantile))
        projection_rows.append(diag)
        for floor in FLOORS:
            e = np.maximum(er, floor)
            cov = (qr*e[:, None, :])@qr.swapaxes(1, 2)
            for name, drift in [('true_drift_arff_covariance', ftrue), ('hybrid', fm), ('arff_original', fa)]:
                parts = spectral_parts(drift, e, qr, r64, h64)
                sensitivity.append(dict(seed=seed, floor=floor, combination=name,
                                        projected_covariance_rmse=rmse(cov, truecov), **parts))
                if floor == .001:
                    add(seed, name, 'float64_floor_1e-3', parts)
        checks.append(dict(seed=seed, reused_hybrid_nll=hp['nll'], decomposition_rounding=hp['nll']-sum(hp[k] for k in ['quadratic','logdet','constant'])))
        print('Validated oracle swaps', seed, flush=True)
    for name, data in [('per_seed.csv', rows), ('projection_per_seed.csv', projection_rows), ('floor_sensitivity_per_seed.csv', sensitivity)]:
        with (OUT/name).open('x', newline='') as f:
            writer=csv.DictWriter(f, fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    grouped = {}
    for row in rows:
        key = row['scope']+'/'+row['combination']
        if key not in grouped:
            subset=[v for v in rows if v['scope']+'/'+v['combination']==key]
            grouped[key] = {k:stats([v[k] for v in subset]) for k in ['quadratic','logdet','constant','nll']}
    sens = {}
    for floor in FLOORS:
        sens[str(floor)] = {}
        for name in ['true_drift_arff_covariance','hybrid','arff_original']:
            subset=[v for v in sensitivity if v['floor']==floor and v['combination']==name]
            sens[str(floor)][name]={k:stats([v[k] for v in subset]) for k in ['quadratic','logdet','nll','projected_covariance_rmse']}
    projection = {k:stats([v[k] for v in projection_rows]) for k in projection_rows[0] if k!='seed'}
    def values(name, scope):
        return np.array([v['nll'] for v in rows if v['combination']==name and v['scope']==scope])
    diff = {
        'arff_minus_joint_mlp_drift_under_true_covariance':stats(values('arff_drift_true_covariance','float64_unprojected_true_covariance')-values('joint_mlp_drift_true_covariance','float64_unprojected_true_covariance')),
        'true_drift_arff_covariance_minus_oracle_float64':stats(values('true_drift_arff_covariance','float64_floor_1e-3')-oracle['nll']),
        'hybrid_minus_arff_native':stats(values('hybrid','native_float32_floor_1e-3')-values('arff_original','native_float32_floor_1e-3')),
        'projection_rmse_change':stats([v['projected_covariance_rmse']-v['raw_covariance_rmse'] for v in projection_rows]),
        'native_minus_float64_hybrid':stats(values('hybrid','native_float32_floor_1e-3')-values('hybrid','float64_floor_1e-3'))}
    for p,digest in protected.items():
        assert sha(p)==digest, p
    summary=dict(oracle=oracle, true_covariance_floor_controls=controls, combinations=grouped, projection=projection,
                 floor_sensitivity=sens, paired_differences=diff, consistency_checks=checks,
                 dataset_sha256=dataset_sha, test_idx_sha256=hashlib.sha256(d.test_idx.tobytes()).hexdigest(),
                 all_protected_files_unchanged=True, oracle_seed_sd_not_applicable=True,
                 evaluation_precision_note='Float64 oracle/sensitivity use archived r and unchanged float32 learned predictions; native controls use accepted float32 inputs. No model conversion.',
                 environment=dict(jax=jax.__version__, numpy=np.__version__, backend=jax.default_backend()))
    write_json(OUT/'summary.json', summary)
    print(json.dumps(dict(oracle=oracle, paired_differences=diff), indent=2), flush=True)

if __name__ == '__main__':
    main()
