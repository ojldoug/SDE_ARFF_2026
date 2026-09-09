#!/usr/bin/env python3
"""Run the fixed, validation-selected Experiment 8 ARFF production protocol.

Selection uses training-only internal holdouts. Canonical validation and test
metrics are computed only after all seven regressions have selected checkpoints.
See docs/ex8_arff_production.md for artifact and timing conventions.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.arff.covariance import covariance_targets, raw_covariance
from src.arff.evaluation import gaussian_nll, true_function_errors
from src.arff.regression import ARFFModel, make_compiled_adaptation_step, predict
from src.arff.two_stage import TwoStageARFFModel, make_folds
from src.arff.validation_selected import (
    ValidationSelectedResult, _split_indices, fit_validation_selected_arff,
)
from src.experiments.config import get_config
from src.experiments.dataset import load_dataset
from src.experiments.definitions import get_experiment
from src.experiments.timing import block_until_ready

METHOD = 'arff_validation_selected_crossfit'
ARTIFACT_VERSION = 2
K = 128
N_FOLDS = 5
FOLD_SEED = 2026
SPD_EPSILON = 1e-3
FIT_SETTINGS = dict(
    K=K, M_min=300, M_max=300, lambda_reg=1e-3, gamma=1.0, delta=0.2,
    resampling=False, metropolis_test=True, validation_fraction=0.1,
    moving_average_length=5, patience=5,
)


@dataclass(frozen=True)
class Stage:
    training: ValidationSelectedResult
    start_offset: float
    elapsed: float
    validation_seed: int


@dataclass(frozen=True)
class LearningResult:
    model: TwoStageARFFModel
    targets: jax.Array
    fold_id: np.ndarray
    stages: dict[str, Stage]
    crossfit_time: float
    algorithm_time: float
    final_key: jax.Array


def fit_selected(key, x, y, *, validation_seed, compiled_step, warmup=False):
    settings = dict(FIT_SETTINGS)
    if warmup:
        # Exercise the identical fitting/validation shapes without consuming the
        # production key or changing its 300-adaptation budget.
        settings.update(M_min=1, M_max=1)
    return fit_validation_selected_arff(
        key, x, y, **settings, validation_seed=validation_seed,
        compiled_adaptation_step=compiled_step,
    )


def learn(key, x, r, h, *, seed, diff_type, compiled_step, warmup=False):
    """Diagnostic's exact regression order and key flow, training arrays only.

The outer clock includes fold construction, setup, initialization, adaptation,
internal validation/selection and out-of-fold target construction. No final
metrics, metadata reconstruction, serialization or logging runs in this scope.
"""
    start = time.perf_counter()
    folds = make_folds(len(x), N_FOLDS, FOLD_SEED)
    d = r.shape[1]
    q = d if diff_type == 'diagonal' else d * (d + 1) // 2
    targets = jnp.zeros((len(x), q), dtype=x.dtype)
    fold_id = np.empty(len(x), dtype=np.int32)
    positions = np.arange(len(x))
    stages = {}

    def regression(name, key, xx, yy, validation_seed):
        stage_start = time.perf_counter()
        key, training = fit_selected(
            key, xx, yy, validation_seed=validation_seed,
            compiled_step=compiled_step, warmup=warmup,
        )
        block_until_ready((key, training.model.omega, training.model.amp))
        stages[name] = Stage(training, stage_start - start,
                             time.perf_counter() - stage_start, validation_seed)
        return key, training.model

    for number, holdout in enumerate(folds):
        mask = np.ones(len(x), dtype=bool)
        mask[holdout] = False
        fit_idx = positions[mask]
        key, model = regression(
            f'fold_{number}', key, x[fit_idx], r[fit_idx] / h[fit_idx],
            100000 + seed * 10000 + 1000 + number,
        )
        residual = r[holdout] - h[holdout] * predict(model, x[holdout])
        targets = targets.at[holdout].set(covariance_targets(residual, h[holdout], diff_type))
        fold_id[holdout] = number
    block_until_ready(targets)
    crossfit_time = time.perf_counter() - start
    key, drift = regression('final_drift', key, x, r / h, 200000 + seed)
    key, covariance = regression('covariance', key, x, targets, 300000 + seed)
    # TwoStageARFFModel is not registered as a pytree: synchronize its arrays.
    block_until_ready((key, drift.omega, drift.amp, covariance.omega, covariance.amp, targets))
    algorithm_time = time.perf_counter() - start
    return LearningResult(TwoStageARFFModel(drift, covariance, diff_type), targets,
                          fold_id, stages, crossfit_time, algorithm_time, key)


def prepare_compiled_functions(x, r, h, *, seed, diff_type):
    start = time.perf_counter()
    compiled_step = make_compiled_adaptation_step(
        **{k: FIT_SETTINGS[k] for k in (
            'delta', 'lambda_reg', 'gamma', 'resampling', 'metropolis_test')}
    )
    # Warm the whole path, including gather/scatter, ridge initialization,
    # internal holdout prediction/MSE and OOF prediction/target shapes.
    warm = learn(jax.random.PRNGKey(987654321), x, r, h, seed=seed,
                 diff_type=diff_type, compiled_step=compiled_step, warmup=True)
    block_until_ready((warm.final_key, warm.targets))
    return compiled_step, time.perf_counter() - start


def evaluate_split(model, x, r, h, definition):
    likelihood = gaussian_nll(model, x, r, h, spd_epsilon=SPD_EPSILON)
    drift, covariance = true_function_errors(
        model, x, true_drift=definition.drift,
        true_diffusion_factor=definition.diffusion_factor,
    )
    return dict(nll=likelihood.nll, drift_rmse=drift, covariance_rmse=covariance,
                raw_spd_violation_rate=likelihood.spd_violation_rate,
                min_raw_eigenvalue=likelihood.min_raw_eigenvalue,
                min_projected_eigenvalue=likelihood.min_projected_eigenvalue)


def evaluate_final(result, data, definition):
    # This function receives an already completed, frozen LearningResult. Test
    # observations are first indexed here, after every selection has finished.
    return {
        label: evaluate_split(result.model, data.x[idx], data.r[idx], data.h[idx], definition)
        for label, idx in [('train', data.train_idx), ('validation', data.validation_idx),
                           ('test', data.test_idx)]
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(dataset_path):
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip()
        status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO_ROOT, text=True)
    except (OSError, subprocess.SubprocessError):
        commit, status = 'unknown', 'unavailable'
    sources = [
        'scripts/run_ex8_arff_validation_selected_crossfit.py',
        'scripts/diagnose_ex8_validation_selected_crossfit.py',
        'src/arff/validation_selected.py', 'src/arff/regression.py',
        'src/arff/two_stage.py', 'src/arff/covariance.py', 'src/arff/evaluation.py',
        'src/experiments/config.py', 'src/experiments/dataset.py',
        'src/experiments/definitions.py', 'src/experiments/timing.py',
    ]
    packages = {}
    for name in ('numpy', 'jax', 'jaxlib', 'jax-cuda13-plugin', 'jax-cuda13-pjrt'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    env_names = ('JAX_PLATFORMS', 'JAX_ENABLE_X64', 'JAX_DEFAULT_MATMUL_PRECISION',
                 'JAX_COMPILATION_CACHE_DIR', 'CUDA_VISIBLE_DEVICES', 'XLA_FLAGS',
                 'XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_PYTHON_CLIENT_MEM_FRACTION',
                 'OMP_NUM_THREADS')
    return dict(
        git_commit=commit, git_dirty=bool(status.strip()), git_status=status,
        dataset_path=str(dataset_path), dataset_sha256=sha256_file(dataset_path),
        source_sha256_json=json.dumps({p: sha256_file(REPO_ROOT / p) for p in sources}, sort_keys=True),
        hostname=platform.node(), platform=platform.platform(),
        python_version=platform.python_version(), python_executable=sys.executable,
        numpy_version=np.__version__, jax_version=jax.__version__,
        package_versions_json=json.dumps(packages, sort_keys=True),
        jax_backend=jax.default_backend(), jax_enable_x64=bool(jax.config.jax_enable_x64),
        jax_default_matmul_precision=str(jax.config.jax_default_matmul_precision),
        devices_json=json.dumps([dict(platform=d.platform, device_kind=d.device_kind,
                                     id=d.id, process_index=d.process_index) for d in jax.devices()]),
        environment_json=json.dumps({k: os.environ[k] for k in env_names if k in os.environ}, sort_keys=True),
        command_json=json.dumps(sys.argv), recorded_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def build_artifact(result, data, metrics, *, seed, compilation_time, metadata, other_times):
    """Device transfers and deterministic metadata reconstruction are untimed."""
    config = get_config('ex8')
    config = replace(config, fourier_frequencies=K,
                     arff=replace(config.arff, **{k: FIT_SETTINGS[k] for k in (
                         'M_min', 'M_max', 'lambda_reg', 'gamma', 'delta',
                         'resampling', 'metropolis_test')}, n_folds=N_FOLDS),
                     evaluation=replace(config.evaluation, spd_epsilon=SPD_EPSILON))
    raw = dict(
        artifact_version=ARTIFACT_VERSION, method=METHOD, experiment='ex8', seed=seed,
        **FIT_SETTINGS, fourier_frequencies=K, n_folds=N_FOLDS, fold_seed=FOLD_SEED,
        arff_validation_fraction=FIT_SETTINGS['validation_fraction'], spd_epsilon=SPD_EPSILON,
        diff_type=result.model.diff_type, input_dimension=data.x.shape[1], output_dimension=data.r.shape[1],
        n_train=len(data.train_idx), n_validation=len(data.validation_idx), n_test=len(data.test_idx),
        train_idx=data.train_idx, validation_idx=data.validation_idx, test_idx=data.test_idx,
        dataset_x_dtype=str(data.x.dtype), dataset_r_dtype=str(data.r.dtype), dataset_h_dtype=str(data.h.dtype),
        training_dtype=str(result.targets.dtype), config_json=json.dumps(asdict(config), sort_keys=True),
        algorithm_time=result.algorithm_time, compilation_time=compilation_time,
        end_to_end_time=compilation_time + result.algorithm_time,
        crossfit_algorithm_time=result.crossfit_time, fold_id=result.fold_id,
        crossfit_covariance_targets=result.targets, final_prng_key=result.final_key,
        warmup_prng_seed=987654321, warmup_iterations_per_regression=1,
        weight_selection='earliest minimum raw internal validation MSE; no post-selection refit',
        iteration_convention='one-based, first adaptation is 1; initialization is not a candidate',
        history_clock_convention='unaltered fitter clock starts after splitting and initial amplitude fit; no exact global history clock is claimed',
        time_convention='synchronized two-stage learning after whole-path one-adaptation warm-up; includes internal validation and OOF target construction',
        end_to_end_scope='compilation/warm-up plus algorithm only; excludes loading, final metrics, provenance, artifact assembly and serialization',
        covariance_rmse_convention='raw symmetric covariance before projection',
        nll_convention='existing Gaussian Euler-Maruyama NLL with eigenvalue floor 1e-3',
        internal_index_convention='positions in canonical train_idx, preserving regression input order',
        **metadata, **other_times,
    )
    positions = np.arange(len(data.train_idx))
    for name, stage in result.stages.items():
        training = stage.training
        raw.update({f'{name}_{field}': getattr(training, field) for field in (
            'validation_mse', 'moving_average', 'cumulative_time', 'best_iteration',
            'best_validation_mse', 'best_time', 'stopped_iteration')})
        raw.update({f'{name}_omega': training.model.omega, f'{name}_amp': training.model.amp,
                    f'{name}_start_offset': stage.start_offset, f'{name}_algorithm_time': stage.elapsed,
                    f'{name}_end_offset': stage.start_offset + stage.elapsed,
                    f'{name}_validation_seed': stage.validation_seed})
        inputs = positions[result.fold_id != int(name[5:])] if name.startswith('fold_') else positions
        # Reuse the frozen fitter's deterministic helper; no RNG state is shared
        # with training. Save indices in canonical-training coordinates.
        fit, validation = _split_indices(len(inputs), validation_fraction=FIT_SETTINGS['validation_fraction'],
                                         seed=stage.validation_seed)
        raw[f'{name}_fit_train_positions'] = inputs[fit]
        raw[f'{name}_internal_validation_train_positions'] = inputs[validation]
    for prefix, model in [('drift', result.model.drift), ('covariance', result.model.covariance)]:
        raw[f'{prefix}_omega'], raw[f'{prefix}_amp'] = model.omega, model.amp
    raw['fold_algorithm_times'] = [result.stages[f'fold_{i}'].elapsed for i in range(N_FOLDS)]
    raw['fold_best_iterations'] = [result.stages[f'fold_{i}'].training.best_iteration for i in range(N_FOLDS)]
    for label, values in metrics.items():
        raw.update({f'{label}_{field}': value for field, value in values.items()})
    return {k: np.asarray(jax.device_get(v)) for k, v in raw.items()}


def reconstruct_model(arrays):
    return TwoStageARFFModel(
        ARFFModel(jnp.asarray(arrays['drift_omega']), jnp.asarray(arrays['drift_amp'])),
        ARFFModel(jnp.asarray(arrays['covariance_omega']), jnp.asarray(arrays['covariance_amp'])),
        str(arrays['diff_type'].item()),
    )


def validate_artifact(a):
    """Integrity checks only: raw non-SPD predictions never disqualify a run."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(int(a['artifact_version']) == ARTIFACT_VERSION and str(a['method']) == METHOD
            and str(a['experiment']) == 'ex8', 'Wrong artifact identity')
    for k, v in FIT_SETTINGS.items():
        require(a[k].item() == v, f'Changed fixed setting: {k}')
    require(int(a['n_folds']) == N_FOLDS and int(a['fold_seed']) == FOLD_SEED
            and float(a['spd_epsilon']) == SPD_EPSILON, 'Changed fold/SPD configuration')
    for k, v in a.items():
        require(not v.dtype.hasobject, f'Object array: {k}')
        if np.issubdtype(v.dtype, np.number):
            # The frozen fitter records nonfinite validation steps as +inf.
            # Preserve those histories if a finite checkpoint was selected.
            if k.endswith(('_validation_mse', '_moving_average')) and v.ndim == 1:
                require(not np.any(np.isnan(v) | np.isneginf(v)), f'Invalid history: {k}')
            else:
                require(np.all(np.isfinite(v)), f'Nonfinite field: {k}')
    n, d, inp = int(a['n_train']), int(a['output_dimension']), int(a['input_dimension'])
    q = d if str(a['diff_type']) == 'diagonal' else d * (d + 1) // 2
    require(a['crossfit_covariance_targets'].shape == (n, q), 'Invalid covariance targets')
    expected_fold_id = np.empty(n, dtype=np.int32)
    for i, holdout in enumerate(make_folds(n, N_FOLDS, FOLD_SEED)):
        expected_fold_id[holdout] = i
    require(np.array_equal(a['fold_id'], expected_fold_id), 'Invalid fold IDs')
    from src.experiments.dataset import validate_split_indices
    validate_split_indices(n + int(a['n_validation']) + int(a['n_test']),
                           a['train_idx'], a['validation_idx'], a['test_idx'])
    require(np.isclose(a['end_to_end_time'], a['algorithm_time'] + a['compilation_time']), 'Timing sum mismatch')
    for name in [f'fold_{i}' for i in range(N_FOLDS)] + ['final_drift', 'covariance']:
        history, clock = a[f'{name}_validation_mse'], a[f'{name}_cumulative_time']
        require(history.shape == clock.shape == a[f'{name}_moving_average'].shape == (300,), 'Invalid history shape')
        best = int(a[f'{name}_best_iteration'])
        require(best == int(np.argmin(history)) + 1 and int(a[f'{name}_stopped_iteration']) == 300,
                'Invalid selected/stopped iteration')
        require(np.isfinite(history[best - 1]) and history[best - 1] == a[f'{name}_best_validation_mse'],
                'Invalid selected MSE')
        require(clock[0] >= 0 and np.all(np.diff(clock) >= 0) and clock[best - 1] == a[f'{name}_best_time'],
                'Invalid history timing')
        start, elapsed, end = (float(a[f'{name}_{k}']) for k in ('start_offset', 'algorithm_time', 'end_offset'))
        require(start >= 0 and elapsed >= clock[-1] and np.isclose(end, start + elapsed)
                and end <= float(a['algorithm_time']) + 1e-7, 'Invalid stage timing')
        outputs = q if name == 'covariance' else d
        require(a[f'{name}_omega'].shape == (inp, K) and a[f'{name}_amp'].shape == (2 * K, outputs),
                'Invalid parameter shape')
        positions = np.arange(n)
        inputs = positions[expected_fold_id != int(name[5:])] if name.startswith('fold_') else positions
        expected_seed = (101000 + int(a['seed']) * 10000 + int(name[5:]) if name.startswith('fold_')
                         else (200000 if name == 'final_drift' else 300000) + int(a['seed']))
        require(int(a[f'{name}_validation_seed']) == expected_seed, 'Invalid internal seed')
        fit, val = _split_indices(len(inputs), validation_fraction=0.1, seed=expected_seed)
        require(np.array_equal(a[f'{name}_fit_train_positions'], inputs[fit])
                and np.array_equal(a[f'{name}_internal_validation_train_positions'], inputs[val]),
                'Invalid internal split')
    require(np.array_equal(a['drift_omega'], a['final_drift_omega'])
            and np.array_equal(a['drift_amp'], a['final_drift_amp']), 'Final drift mismatch')
    for label in ('train', 'validation', 'test'):
        for field in ('nll', 'drift_rmse', 'covariance_rmse', 'raw_spd_violation_rate',
                      'min_raw_eigenvalue', 'min_projected_eigenvalue'):
            require(a[f'{label}_{field}'].shape == () and np.isfinite(a[f'{label}_{field}']), 'Invalid split metric')
        require(0 <= a[f'{label}_raw_spd_violation_rate'] <= 1, 'Invalid violation rate')


def save_artifact(path, arrays, model, check_x):
    """Validate and round-trip a temporary archive, then publish without clobber."""
    validate_artifact(arrays)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite artifact: {path}')
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        with np.load(temporary, allow_pickle=False) as archive:
            restored_arrays = {k: archive[k] for k in archive.files}
        validate_artifact(restored_arrays)
        restored = reconstruct_model(restored_arrays)
        np.testing.assert_array_equal(np.asarray(predict(restored.drift, check_x)),
                                      np.asarray(predict(model.drift, check_x)))
        np.testing.assert_array_equal(np.asarray(raw_covariance(restored.covariance, check_x, restored.diff_type)),
                                      np.asarray(raw_covariance(model.covariance, check_x, model.diff_type)))
        # Same-filesystem hard link is atomic and fails if another writer won.
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def main(argv=None):
    wall_start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--artifact-path', type=Path)
    args = parser.parse_args(argv)
    if not 0 <= args.seed <= np.iinfo(np.uint32).max:
        parser.error('seed must fit an unsigned 32-bit PRNG seed')
    path = (args.artifact_path or REPO_ROOT / 'results' / 'production' /
            'arff_validation_selected_ex8' / f'seed_{args.seed}_artifacts.npz').expanduser().resolve()
    if path.suffix != '.npz' or path.exists():
        parser.error('artifact path must be a new .npz file')
    dataset_path = REPO_ROOT / 'data' / 'ex8.npz'
    definition = get_experiment('ex8')
    metadata_start = time.perf_counter()
    metadata = provenance(dataset_path)
    provenance_time = time.perf_counter() - metadata_start
    loading_start = time.perf_counter()
    data = load_dataset(dataset_path)
    x, r, h = (jnp.asarray(a[data.train_idx]) for a in (data.x, data.r, data.h))
    block_until_ready((x, r, h))
    loading_time = time.perf_counter() - loading_start
    print(f'Experiment : ex8\nseed       : {args.seed}\nbackend    : {jax.default_backend()}\n'
          f'train N    : {len(x)}\nK          : {K}\niterations : 300\n'
          'Warming the training path (one discarded adaptation per regression)...', flush=True)
    compiled_step, compilation_time = prepare_compiled_functions(x, r, h, seed=args.seed, diff_type=definition.diff_type)
    print('Fitting five cross-fit drift models, final drift and covariance...', flush=True)
    result = learn(jax.random.PRNGKey(args.seed), x, r, h, seed=args.seed,
                   diff_type=definition.diff_type, compiled_step=compiled_step)
    print(f'algorithm time       : {result.algorithm_time:.3f} s\n'
          f'first-call/JIT time  : {compilation_time:.3f} s\n'
          f'end-to-end time      : {compilation_time + result.algorithm_time:.3f} s', flush=True)
    for name, stage in result.stages.items():
        print(f'{name}: best iteration={stage.training.best_iteration}, '
              f'best internal MSE={stage.training.best_validation_mse:.8e}, time={stage.elapsed:.3f} s')
    evaluation_start = time.perf_counter()
    metrics = evaluate_final(result, data, definition)
    evaluation_time = time.perf_counter() - evaluation_start
    for label, values in metrics.items():
        print(f'\n{label}')
        for field, title in [('nll', 'NLL'), ('drift_rmse', 'drift RMSE'),
                             ('covariance_rmse', 'covariance RMSE'),
                             ('raw_spd_violation_rate', 'raw SPD violations'),
                             ('min_raw_eigenvalue', 'min raw eigenvalue')]:
            print(f'  {title:20s}: {values[field]:.8e}')
    assembly_start = time.perf_counter()
    arrays = build_artifact(result, data, metrics, seed=args.seed, compilation_time=compilation_time,
                            metadata=metadata, other_times=dict(provenance_time=provenance_time,
                            data_loading_time=loading_time, final_evaluation_time=evaluation_time))
    arrays['artifact_assembly_time'] = np.asarray(time.perf_counter() - assembly_start)
    arrays['wall_time_before_serialization'] = np.asarray(time.perf_counter() - wall_start)
    archive_start = time.perf_counter()
    save_artifact(path, arrays, result.model, x[:16])
    print(f'\nartifact   : {path}\narchive/round-trip wall time: {time.perf_counter() - archive_start:.3f} s\n'
          f'runner wall time: {time.perf_counter() - wall_start:.3f} s', flush=True)


if __name__ == '__main__':
    main()
