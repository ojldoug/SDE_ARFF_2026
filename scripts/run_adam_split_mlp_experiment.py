#!/usr/bin/env python3
"""Complete single-run MLP-split runner with archival-v4 output.

Uses the SAME capacity-matching helper and config.adam_mlp settings as the
uploaded joint MLP runner. Does not alter any existing production files.

Do not start this while the Fourier timing campaign is running. A machine-wide
NVIDIA idle check is made BEFORE creating JAX arrays. There is no busy bypass.
The --allow-cpu flag is for isolated development testing, not GPU benchmarks.

Examples AFTER the benchmark machine is available:
  python scripts/run_adam_split_mlp_experiment.py ex2 --seed 30 --validation-only
  python scripts/run_adam_split_mlp_experiment.py ex2 --seed 30 \
      --artifact-path results/smoke_mlp_split_ex2_seed30.npz
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
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

from src.adam.mlp import (
    AdamMLPModel, MLPParams, gaussian_nll, initialize_model, predict_covariance, predict_mlp,
)
from src.adam.split_mlp import (
    CovarianceMLPModel, covariance_from_split_model, covariance_gaussian_nll,
    drift_mse, fit_split_mlp_adam,
)
from src.adam.training import make_compiled_adam_functions
from src.arff.two_stage import make_folds
from src.experiments.config import get_config
from src.experiments.dataset import load_dataset
from src.experiments.definitions import get_experiment
from src.experiments.mlp_size import matched_two_layer_width
from src.experiments.model_size import covariance_output_dimension
from src.experiments.timing import TimingResult, block_until_ready, timed_call


def check_machine_idle(allow_cpu: bool) -> None:
    if allow_cpu:
        # This flag is intentionally explicit; avoid silently running a GPU job.
        if os.environ.get('JAX_PLATFORMS') != 'cpu':
            raise RuntimeError('--allow-cpu requires JAX_PLATFORMS=cpu')
        return

    result = subprocess.run(
        [
            'nvidia-smi',
            '--query-compute-apps=pid,process_name',
            '--format=csv,noheader',
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    own_pid = os.getpid()
    other_processes = []

    for line in result.stdout.splitlines():
        line = line.strip()

        if not line:
            continue

        pid_text = line.split(',', 1)[0].strip()

        try:
            pid = int(pid_text)
        except ValueError:
            other_processes.append(line)
            continue

        if pid != own_pid:
            other_processes.append(line)

    if other_processes:
        raise RuntimeError(
            'GPU machine is busy. Leave the current campaign running.\n'
            + '\n'.join(other_processes)
        )


def batch_shapes(sizes, batch_size):
    answer = set()
    for n in sizes:
        answer.add(min(n, batch_size))
        if n % batch_size:
            answer.add(n % batch_size)
    return sorted(answer)


def prepare_compiled_functions(x_train, r_train, h_train, x_val, r_val, h_val, *,
                               input_dimension, output_dimension, diff_type,
                               hidden_sizes, n_folds, fold_seed, batch_size, learning_rate):
    _, init_key = jax.random.split(jax.random.PRNGKey(987654321))
    initial = initialize_model(init_key, input_dimension=input_dimension,
                               output_dimension=output_dimension, diff_type=diff_type,
                               hidden_sizes=hidden_sizes)
    cov = CovarianceMLPModel(initial.covariance, diff_type, output_dimension)
    block_until_ready(initial)
    drift_functions = make_compiled_adam_functions(learning_rate, nll_fn=drift_mse)
    cov_functions = make_compiled_adam_functions(learning_rate, nll_fn=covariance_gaussian_nll)
    fold_sizes = [len(x_train)-len(f) for f in make_folds(len(x_train), n_folds, fold_seed)]
    first_call_seconds = 0.0
    for name, model, functions, sizes, val_target in [
        ('drift', initial.drift, drift_functions, fold_sizes + [len(x_train)], r_val / h_val),
        ('covariance', cov, cov_functions, [len(x_train)], jnp.zeros_like(r_val)),
    ]:
        optimizer, update, loss = functions
        state = optimizer.init(model)
        for size in batch_shapes(sizes, batch_size):
            target = jnp.zeros_like(r_train[:size])
            block_until_ready((state, target))
            _, elapsed = timed_call(update, model, state, x_train[:size], target, h_train[:size])
            first_call_seconds += elapsed
            print(f'  {name} update N={size}: {elapsed:.3f} s', flush=True)
        block_until_ready(val_target)
        _, elapsed = timed_call(loss, model, x_val, val_target, h_val)
        first_call_seconds += elapsed
        print(f'  {name} validation N={len(x_val)}: {elapsed:.3f} s', flush=True)
    return drift_functions, cov_functions, first_call_seconds


def evaluate_model(model, x, r, h, definition, *, chunk_size=8192):
    """Sample-weighted aggregate; bounded memory; outside benchmark timing."""
    sums = np.zeros(3, dtype=np.float64)
    lo, hi, nonpositive = np.inf, -np.inf, 0
    for start in range(0, len(x), chunk_size):
        xx, rr, hh = x[start:start+chunk_size], r[start:start+chunk_size], h[start:start+chunk_size]
        f = np.asarray(predict_mlp(model.drift, xx))
        a = np.asarray(predict_covariance(model, xx))
        ft = np.asarray(definition.drift(xx))
        sigma = np.asarray(definition.diffusion_factor(xx))
        at = sigma @ np.swapaxes(sigma, -1, -2)
        if f.shape != ft.shape or a.shape != at.shape:
            raise ValueError('Prediction/truth shapes differ; refusing implicit broadcasting')
        eigenvalues = np.linalg.eigvalsh(a)
        values = [float(gaussian_nll(model, xx, rr, hh)),
                  float(np.mean((f.astype(np.float64)-ft) ** 2)),
                  float(np.mean((a.astype(np.float64)-at) ** 2))]
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(eigenvalues)):
            raise RuntimeError('Nonfinite final evaluation')
        sums += len(xx) * np.asarray(values)
        lo, hi = min(lo, float(eigenvalues.min())), max(hi, float(eigenvalues.max()))
        nonpositive += int(np.sum(np.min(eigenvalues, axis=-1) <= 0))
    means = sums / len(x)
    return dict(nll=float(means[0]), drift_rmse=float(np.sqrt(means[1])),
                covariance_rmse=float(np.sqrt(means[2])), min_covariance_eig=lo,
                max_covariance_eig=hi, numerical_nonpositive_eig_rate=nonpositive/len(x))


def flatten_params(params, prefix):
    arrays = {f'{prefix}_weight_{i}': np.asarray(jax.device_get(w)) for i, w in enumerate(params.weights)}
    arrays.update({f'{prefix}_bias_{i}': np.asarray(jax.device_get(b)) for i, b in enumerate(params.biases)})
    return arrays


def reconstruct_model(arrays):
    layers = int(np.asarray(arrays['hidden_layers']).item()) + 1
    def params(prefix):
        return MLPParams(
            weights=tuple(jnp.asarray(arrays[f'{prefix}_weight_{i}']) for i in range(layers)),
            biases=tuple(jnp.asarray(arrays[f'{prefix}_bias_{i}']) for i in range(layers)),
        )
    return AdamMLPModel(drift=params('drift'), covariance=params('covariance'),
                        diff_type=str(np.asarray(arrays['diff_type']).item()))


def git_info():
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO_ROOT, text=True).strip()
        status = subprocess.check_output(['git','status','--porcelain'], cwd=REPO_ROOT, text=True)
        return commit, bool(status.strip()), status
    except (OSError, subprocess.SubprocessError):
        return 'unknown', True, 'unavailable'


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_artifact(arrays):
    """Schema/shape/value checks, not an accuracy threshold or seed filter."""
    if str(arrays['method'].item()) != 'adam_split_mlp' or int(arrays['artifact_version']) != 4:
        raise ValueError('Wrong artifact identity')
    epochs, folds = int(arrays['epochs_per_regression']), int(arrays['n_folds'])
    for stage, loss_name in [('final_drift','mse'), ('covariance','nll')]:
        train = arrays[f'{stage}_training_{loss_name}']
        val = arrays[f'{stage}_validation_{loss_name}']
        local = arrays[f'{stage}_cumulative_time']
        global_time = arrays[f'{stage}_global_cumulative_time']
        if any(v.shape != (epochs,) or not np.all(np.isfinite(v)) for v in [train,val,local,global_time]):
            raise ValueError(f'Invalid {stage} histories')
        if local[0] < 0 or np.any(np.diff(local) < 0):
            raise ValueError(f'Invalid {stage} local clock')
        if not np.allclose(global_time, float(arrays[f'{stage}_start_offset'])+local, rtol=1e-9, atol=1e-7):
            raise ValueError(f'Invalid {stage} global clock')
        if global_time[-1] > float(arrays['internal_algorithm_time']) + .1:
            raise ValueError('History exceeds total time')
        best = int(arrays[f'{stage}_best_epoch'])
        if best != int(np.argmin(val)) or not np.isclose(val[best], arrays[f'{stage}_best_validation_{loss_name}']):
            raise ValueError('Invalid selected checkpoint')
    fold_times = arrays['fold_algorithm_times']
    if fold_times.shape != (folds,) or np.any(fold_times <= 0) or not np.all(np.isfinite(fold_times)):
        raise ValueError('Invalid fold timing')
    fid = arrays['fold_id']
    if fid.shape != (int(arrays['n_train']),) or set(np.unique(fid)) != set(range(folds)):
        raise ValueError('Invalid fold IDs')
    if not np.isclose(arrays['end_to_end_time'], arrays['algorithm_time']+arrays['compilation_time']):
        raise ValueError('Timing sum mismatch')
    for k, v in arrays.items():
        if np.issubdtype(v.dtype, np.number) and not np.all(np.isfinite(v)):
            raise ValueError(f'Nonfinite field {k}')
    # Do not reject a scientific result solely for a numerical eigenvalue
    # diagnostic; preserve it explicitly instead of selecting successful seeds.


def save_artifact(path, *, name, seed, config, definition, result, timing, metrics,
                  hidden_width, declared_count, target_count, sizes, dataset_path, check_x):
    start = time.perf_counter()
    drift, cov, cf = result.final_drift_training, result.covariance_training, result.crossfit
    commit, dirty, status = git_info()
    raw = dict(
        artifact_version=4, method='adam_split_mlp', experiment=name, seed=seed,
        diff_type=definition.diff_type, hidden_layers=2, hidden_width=hidden_width,
        input_dimension=definition.state_dimension, output_dimension=definition.n_dimensions,
        activation='tanh', mlp_parameter_count=declared_count, fourier_parameter_count=target_count,
        fourier_frequencies=config.fourier_frequencies, epochs_per_regression=config.adam_mlp.epochs,
        batch_size=config.adam_mlp.batch_size, learning_rate=config.adam_mlp.learning_rate,
        n_folds=config.arff.n_folds, fold_seed=config.split.seed,
        n_train=sizes[0], n_validation=sizes[1], n_test=sizes[2],
        algorithm_time=timing.algorithm_seconds, compilation_time=timing.compilation_seconds,
        end_to_end_time=timing.end_to_end_seconds, internal_algorithm_time=result.internal_algorithm_time,
        fold_algorithm_times=cf.fold_algorithm_times, fold_best_epochs=cf.fold_best_epochs,
        fold_best_validation_mse=cf.fold_best_validation_losses, fold_id=cf.fold_id,
        crossfit_algorithm_time=cf.crossfit_algorithm_time,
        final_drift_start_offset=result.final_drift_start_offset,
        final_drift_algorithm_time=result.final_drift_algorithm_time,
        final_drift_end_offset=result.final_drift_start_offset+result.final_drift_algorithm_time,
        covariance_start_offset=result.covariance_start_offset,
        covariance_algorithm_time=result.covariance_algorithm_time,
        covariance_end_offset=result.covariance_start_offset+result.covariance_algorithm_time,
        git_commit=commit, git_dirty=dirty, git_status=status, hostname=platform.node(),
        python_version=platform.python_version(), numpy_version=np.__version__, jax_version=jax.__version__,
        jax_backend=jax.default_backend(), dataset_sha256=sha256_file(dataset_path),
        config_json=json.dumps(asdict(config), sort_keys=True),
        time_convention='Core update/validation first-call warm-up separated; auxiliary setup may include first-call work',
        history_clock_convention='stage_call_start_plus_generic_trainer_local_clock',
        end_to_end_scope='reported warm-up plus algorithm, excludes data loading, final evaluation and serialization',
        weight_selection='validation-selected checkpoints, not necessarily final epoch',
    )
    for stage, training, kind, offset in [
        ('final_drift', drift, 'mse', result.final_drift_start_offset),
        ('covariance', cov, 'nll', result.covariance_start_offset),
    ]:
        raw.update({f'{stage}_best_epoch': training.best_epoch,
                    f'{stage}_best_validation_{kind}': training.best_validation_nll,
                    f'{stage}_training_{kind}': training.training_nll,
                    f'{stage}_validation_{kind}': training.validation_nll,
                    f'{stage}_cumulative_time': training.cumulative_time,
                    f'{stage}_global_cumulative_time': offset+np.asarray(training.cumulative_time)})
    for label, values in metrics.items():
        raw.update({f'{label}_{key}': value for key, value in values.items()})
    raw.update(flatten_params(result.model.drift, 'drift'))
    raw.update(flatten_params(result.model.covariance, 'covariance'))
    raw['actual_mlp_parameter_count'] = sum(a.size for a in jax.tree_util.tree_leaves(result.model))
    source_paths = ['src/adam/mlp.py','src/adam/training.py','src/adam/split_mlp.py',
                    'scripts/run_adam_split_mlp_experiment.py','src/experiments/config.py']
    raw['source_sha256_json'] = json.dumps({p: sha256_file(REPO_ROOT/p) for p in source_paths}, sort_keys=True)
    arrays = {k: np.asarray(v) for k, v in raw.items()}
    validate_artifact(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite artifact: {path}')
    fd, tmp_name = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        with np.load(tmp_name, allow_pickle=False) as z:
            loaded = {k: z[k] for k in z.files}
        validate_artifact(loaded)
        restored = reconstruct_model(loaded)
        np.testing.assert_allclose(np.asarray(predict_mlp(restored.drift, check_x)),
                                   np.asarray(predict_mlp(result.model.drift, check_x)), rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(np.asarray(predict_covariance(restored, check_x)),
                                   np.asarray(predict_covariance(result.model, check_x)), rtol=1e-6, atol=1e-7)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    print(f'artifact   : {path}\narchive/round-trip wall time: {time.perf_counter()-start:.3f} s', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('experiment', choices=[f'ex{i}' for i in range(1,9)])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--artifact-path', type=Path)
    p.add_argument('--validation-only', action='store_true')
    p.add_argument('--allow-cpu', action='store_true')
    args = p.parse_args()
    if args.seed < 0:
        p.error('seed must be nonnegative')
    if args.validation_only and args.artifact_path:
        p.error('--validation-only cannot produce a production artifact')
    if args.artifact_path and args.artifact_path.expanduser().exists():
        p.error('Artifact path already exists; use a new smoke-test filename')
    check_machine_idle(args.allow_cpu)
    cfg, definition = get_config(args.experiment), get_experiment(args.experiment)
    if cfg.fourier_frequencies is None:
        raise ValueError('Capacity target is not configured')
    hp = cfg.adam_mlp
    if hp.epochs <= 0 or hp.batch_size <= 0 or hp.learning_rate <= 0:
        raise ValueError('Invalid MLP optimization configuration')
    width, declared_count, target_count = matched_two_layer_width(
        state_dimension=definition.n_dimensions,
        covariance_output_dimension=covariance_output_dimension(definition.n_dimensions, definition.diff_type),
        n_frequencies=cfg.fourier_frequencies,
    )
    hidden_sizes = (width, width)
    dataset_path = REPO_ROOT/'data'/f'{args.experiment}.npz'
    data = load_dataset(dataset_path)
    tr, va = data.train_idx, data.validation_idx
    arrays = tuple(jnp.asarray(a[idx]) for idx in [tr,va] for a in [data.x,data.r,data.h])
    block_until_ready(arrays)
    x, r, h, xv, rv, hv = arrays
    if not bool(jax.device_get(jnp.all(h > 0) & jnp.all(hv > 0))):
        raise ValueError('Observation lags must be positive')
    print(f'Experiment : {args.experiment}\nseed       : {args.seed}\nbackend    : {jax.default_backend()}')
    print(f'train N    : {len(tr)}\nvalidation : {len(va)}\nhidden     : {width}-{width}')
    print(f'MLP params : {declared_count}\nFourier target params: {target_count}')
    print(f'folds      : {cfg.arff.n_folds}\nepochs/regression: {hp.epochs}\nbatch size : {hp.batch_size}\nlearning rate: {hp.learning_rate:.8e}', flush=True)
    print('Split MLP first-call/JIT warm-up', flush=True)
    drift, cov, first_call = prepare_compiled_functions(
        *arrays, input_dimension=definition.state_dimension, output_dimension=definition.n_dimensions,
        diff_type=definition.diff_type, hidden_sizes=hidden_sizes, n_folds=cfg.arff.n_folds,
        fold_seed=cfg.split.seed, batch_size=hp.batch_size, learning_rate=hp.learning_rate,
    )
    print(f'first-call/JIT overhead: {first_call:.3f} s\nTraining split MLP...', flush=True)
    (_, result), algorithm_time = timed_call(
        fit_split_mlp_adam, jax.random.PRNGKey(args.seed), *arrays,
        input_dimension=definition.state_dimension, output_dimension=definition.n_dimensions,
        diff_type=definition.diff_type, hidden_sizes=hidden_sizes, n_folds=cfg.arff.n_folds,
        fold_seed=cfg.split.seed, epochs=hp.epochs, batch_size=hp.batch_size,
        drift_optimizer=drift[0], drift_compiled_train_step=drift[1], drift_compiled_loss=drift[2],
        covariance_optimizer=cov[0], covariance_compiled_train_step=cov[1], covariance_compiled_loss=cov[2],
    )
    timing = TimingResult(compilation_seconds=first_call, algorithm_seconds=algorithm_time)
    print(f'algorithm time       : {algorithm_time:.3f} s\nfirst-call/JIT time  : {first_call:.3f} s\nend-to-end time      : {timing.end_to_end_seconds:.3f} s')
    print(f'final drift best epoch: {result.final_drift_training.best_epoch}\nfinal drift best validation MSE: {result.final_drift_training.best_validation_nll:.8e}')
    print(f'covariance best epoch : {result.covariance_training.best_epoch}\ncovariance best validation NLL: {result.covariance_training.best_validation_nll:.8e}', flush=True)
    # Independent comparison against the real joint model, not L L^T vs itself.
    check_x = xv[:min(16,len(xv))]
    np.testing.assert_allclose(np.asarray(covariance_from_split_model(result.covariance_training.model, check_x)),
                               np.asarray(predict_covariance(result.model, check_x)), rtol=1e-5, atol=1e-7)
    residual = rv[:len(check_x)] - hv[:len(check_x)] * predict_mlp(result.model.drift, check_x)
    np.testing.assert_allclose(float(covariance_gaussian_nll(result.covariance_training.model, check_x, residual, hv[:len(check_x)])),
                               float(gaussian_nll(result.model, check_x, rv[:len(check_x)], hv[:len(check_x)])), rtol=1e-5, atol=1e-5)
    metrics = {}
    splits = [('train',tr), ('validation',va)]
    if not args.validation_only:
        splits.append(('test',data.test_idx))
    for label, idx in splits:
        values = evaluate_model(result.model, data.x[idx], data.r[idx], data.h[idx], definition)
        metrics[label] = values
        print('\n'+label)
        for field, log_label in [('nll','NLL'),('drift_rmse','drift RMSE'),('covariance_rmse','covariance RMSE'),
                                  ('min_covariance_eig','min covariance eig'),('max_covariance_eig','max covariance eig')]:
            print(f'  {log_label:20s}: {values[field]:.8e}')
    print(flush=True)
    if args.artifact_path:
        save_artifact(args.artifact_path.expanduser().resolve(), name=args.experiment, seed=args.seed,
                      config=cfg, definition=definition, result=result, timing=timing, metrics=metrics,
                      hidden_width=width, declared_count=declared_count, target_count=target_count,
                      sizes=(len(tr),len(va),len(data.test_idx)), dataset_path=dataset_path, check_x=check_x)


if __name__ == '__main__':
    main()
