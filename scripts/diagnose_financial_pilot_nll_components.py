"""CPU-only decomposition of archived forecasts; no fitting or model inference."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/financial_covariance_pilot_v1'
OUT = ROOT / 'results/financial_covariance_pilot_closure_v1'
FLOOR = 1e-4
RTOL = ATOL = 1e-9  # Unchanged financial common-evaluator tolerances.


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    names = ['selected_validation_predictions.npz', 'selected_test_predictions.npz',
             'validation_score_arrays.npz', 'test_score_arrays.npz', 'summary.json',
             'checkpoint_summary.csv', 'candidate_validation.json', 'ARTIFACT_INDEX.json',
             'PROTOCOL.md', 'SELECTION.json', 'COMPLETE.json', 'OUTPUT_VALIDATION.json']
    paths = [SOURCE / name for name in names]
    paths += [ROOT / 'results/experimental_trajectory_pilot_v1/VALIDATION.json',
              ROOT / 'results/experimental_trajectory_evaluation_float64_v1/summary.json']
    before = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    index = json.loads((SOURCE / 'ARTIFACT_INDEX.json').read_text())
    expected = {r['path']: r['sha256'] for r in index['files']}
    for p in paths[:len(names)]:
        if p.name != 'ARTIFACT_INDEX.json':
            assert sha(p) == expected[str(p.relative_to(ROOT))], p
    summary = json.loads((SOURCE / 'summary.json').read_text())
    candidate = json.loads((SOURCE / 'candidate_validation.json').read_text())
    arff = [r for r in candidate if r['method'] == 'arff']
    assert len(arff) == 9 and all(r['selected'] == 1 for r in arff)
    with (SOURCE / 'checkpoint_summary.csv').open() as f:
        checks = [r for r in csv.DictReader(f) if r['method'] == 'arff']
    assert len(checks) == 9 and all(r['selected_native_index'] == '1' for r in checks)

    rows, validation, quadratic_shares = [], [], []
    for split in ['validation', 'test']:
        with np.load(SOURCE / f'selected_{split}_predictions.npz') as p, np.load(SOURCE / f'{split}_score_arrays.npz') as s:
            np.testing.assert_array_equal(p['ends'], s['ends'])
            y = np.asarray(p['returns'], dtype=np.float64)
            D = np.asarray(p['D'], dtype=np.float64)
            n, d = y.shape
            C = float(d * .5 * np.log(2 * np.pi))
            for method in p.files:
                if method in ['returns', 'ends', 'D']:
                    continue
                raw = np.asarray(p[method], dtype=np.float64)
                sym = .5 * (raw + raw.transpose(0, 2, 1))
                w, q = np.linalg.eigh(sym)
                wc = np.maximum(w, FLOOR)
                active = np.any(w < FLOOR, axis=1)
                # Physical covariance = D Q diag(wc) Q^T D. No SDE h scaling.
                rotated = np.einsum('nji,nj->ni', q, y / D)
                Q = .5 * np.sum((rotated / np.sqrt(wc)) ** 2, axis=1)
                L = .5 * np.log(wc).sum(axis=1) + np.log(D).sum()
                stored = np.asarray(s[method], dtype=np.float64)
                np.testing.assert_allclose(L + Q + C, stored, rtol=RTOL, atol=ATOL)
                np.testing.assert_allclose(stored.mean(), summary['metrics'][split][method]['nll'], rtol=RTOL, atol=ATOL)
                np.testing.assert_allclose(active.mean(), summary['metrics'][split][method]['projection_frequency'], rtol=0, atol=0)
                validation.append(dict(split=split, method=method, count=n,
                                       max_abs_reconstruction_difference=float(np.max(np.abs(L + Q + C - stored))),
                                       floor_frequency_exact_match=True))
                for group, mask in [('all', np.ones(n, bool)), ('floor_active', active), ('no_floor', ~active)]:
                    count = int(mask.sum())
                    row = dict(split=split, method=method, group=group, count=count,
                               total_count=n, fraction=count/n,
                               raw_nonpositive_count=int(np.sum(w[mask, 0] <= 0)),
                               conditional_logdet_half=float(L[mask].mean()) if count else None,
                               conditional_quadratic_half=float(Q[mask].mean()) if count else None,
                               conditional_gaussian_constant=C if count else None,
                               conditional_archived_nll=float(stored[mask].mean()) if count else None,
                               overall_logdet_contribution=float(L[mask].sum()/n),
                               overall_quadratic_contribution=float(Q[mask].sum()/n),
                               overall_constant_contribution=float(C*count/n),
                               overall_archived_nll_contribution=float(stored[mask].sum()/n))
                    np.testing.assert_allclose(row['overall_logdet_contribution'] + row['overall_quadratic_contribution'] + row['overall_constant_contribution'], row['overall_archived_nll_contribution'], rtol=RTOL, atol=ATOL)
                    rows.append(row)
                a, b, c = rows[-3:]
                for name in ['count', 'overall_logdet_contribution', 'overall_quadratic_contribution', 'overall_constant_contribution', 'overall_archived_nll_contribution']:
                    np.testing.assert_allclose(b[name] + c[name], a[name], rtol=RTOL, atol=ATOL)
                quadratic_shares.append(dict(split=split, method=method, fraction_quadratic_from_floor=float(Q[active].sum()/Q.sum())))
    assert all(sha(ROOT / name) == h for name, h in before.items())
    with (OUT / 'components_by_floor.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    result = dict(floor_scaled=FLOOR, rtol=RTOL, atol=ATOL,
                  all_checks_passed=True, original_inputs_preserved=True,
                  original_arff_selected_iterations=[r['selected'] for r in arff],
                  rows=rows, score_reconstruction=validation, quadratic_shares=quadratic_shares,
                  input_sha256=before, script_sha256=sha(Path(__file__)),
                  python=sys.version, numpy=np.__version__,
                  execution='CPU NumPy arithmetic on saved raw forecasts/returns/scores only; no JAX, checkpoint loading, model inference, fitting or raw market archives')
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(all_checks_passed=True,
                         max_abs_difference=max(r['max_abs_reconstruction_difference'] for r in validation),
                         test_arff=[r for r in rows if r['split']=='test' and r['method'] in ['arff_s0','arff_s1','arff_s2']],
                         test_quadratic_shares=[r for r in quadratic_shares if r['split']=='test']), indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            (OUT / 'FAILURE.json').write_text(json.dumps(dict(error=repr(error)), indent=2) + '\n')
        raise
