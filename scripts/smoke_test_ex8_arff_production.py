#!/usr/bin/env python3
"""CPU-only checks using tiny synthetic data and fabricated archive fixtures.

Never loads data/ex8.npz or performs the full Experiment 8 fit.
"""
from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ['JAX_PLATFORMS'] = 'cpu'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import jax
import jax.numpy as jnp
import numpy as np

import diagnose_ex8_validation_selected_crossfit as diagnostic
import run_ex8_arff_validation_selected_crossfit as runner
from src.arff.regression import ARFFModel
from src.arff.two_stage import TwoStageARFFModel, make_folds
from src.arff.validation_selected import ValidationSelectedResult, fit_validation_selected_arff
from src.experiments.dataset import ExperimentDataset
from src.experiments.definitions import get_experiment


def fixture():
    """No training: deliberately indefinite covariance plus 300 synthetic losses."""
    x = np.linspace(-1, 1, 200, dtype=np.float32).reshape(100, 2)
    data = ExperimentDataset(x, np.zeros_like(x), np.full((100, 1), .01, dtype=np.float32),
                             np.arange(80), np.arange(80, 90), np.arange(90, 100))
    omega = jnp.zeros((2, 128))
    drift = ARFFModel(omega, jnp.zeros((256, 2)))
    covariance = ARFFModel(omega, jnp.zeros((256, 3)).at[0].set(jnp.array([1., 2., 1.])))
    fid = np.empty(80, dtype=np.int32)
    for i, holdout in enumerate(make_folds(80, 5, 2026)):
        fid[holdout] = i
    stages = {}
    for i, name in enumerate([f'fold_{i}' for i in range(5)] + ['final_drift', 'covariance']):
        losses = np.ones(300)
        clock = np.arange(1, 301) / 1000
        training = ValidationSelectedResult(covariance if name == 'covariance' else drift,
                                             losses, losses.copy(), clock, 1, 1., clock[0], 300)
        seed = 101000 + i if i < 5 else (200000 if i == 5 else 300000)
        stages[name] = runner.Stage(training, float(i), 1., seed)
    result = runner.LearningResult(TwoStageARFFModel(drift, covariance, 'symmetric'),
                                    jnp.zeros((80, 3)), fid, stages, 5., 7., jax.random.PRNGKey(0))
    metadata = dict(git_commit='fixture', git_dirty=True, git_status='fixture', dataset_sha256='a' * 64,
                    source_sha256_json='{}', hostname='fixture', python_version='fixture',
                    numpy_version=np.__version__, jax_version=jax.__version__, jax_backend='cpu')
    return data, result, metadata


def tiny_fit(key, x, y, *, validation_seed, compiled_step, warmup=False):
    settings = dict(runner.FIT_SETTINGS, K=4, M_min=1 if warmup else 3, M_max=1 if warmup else 3)
    return fit_validation_selected_arff(key, x, y, **settings, validation_seed=validation_seed,
                                        compiled_adaptation_step=compiled_step)


class ProductionTests(unittest.TestCase):
    def test_fixed_settings_match_diagnostic(self):
        mapping = dict(K='K', M_min='M_MIN', M_max='M_MAX', lambda_reg='LAMBDA_REG', gamma='GAMMA',
                       delta='DELTA', resampling='RESAMPLING', metropolis_test='METROPOLIS_TEST',
                       validation_fraction='ARFF_VALIDATION_FRACTION', moving_average_length='MOVING_AVERAGE_LENGTH',
                       patience='PATIENCE')
        for key, attribute in mapping.items():
            self.assertEqual(runner.FIT_SETTINGS[key], getattr(diagnostic, attribute))
        self.assertEqual(runner.FOLD_SEED, runner.get_config('ex8').split.seed)
        self.assertEqual(runner.SPD_EPSILON, runner.get_config('ex8').evaluation.spd_epsilon)

    def test_tiny_numerical_parity_and_warmup_key_isolation(self):
        # Uneven folds exercise both fold shapes. Actual numerical operations,
        # only 31 synthetic observations, K=4 and 3 adaptations per regression.
        x = jnp.linspace(-1, 1, 62).reshape(31, 2)
        h = jnp.full((31, 1), .01)
        r = .02 * jnp.sin(x * 3)
        kernel = runner.make_compiled_adaptation_step(delta=.2, lambda_reg=.001, gamma=1.,
                                                     resampling=False, metropolis_test=True)
        key = jax.random.PRNGKey(7)
        with patch.object(diagnostic, 'fit_selected', side_effect=tiny_fit), contextlib.redirect_stdout(io.StringIO()):
            expected_key, cf, fold_results = diagnostic.build_cross_fitted_targets(
                key, x, r, h, diff_type='symmetric', fold_seed=2026,
                internal_validation_seed=170000, compiled_step=kernel)
            expected_key, drift = tiny_fit(expected_key, x, r / h, validation_seed=200007, compiled_step=kernel)
            expected_key, cov = tiny_fit(expected_key, x, cf.covariance_targets, validation_seed=300007, compiled_step=kernel)
        with patch.object(runner, 'fit_selected', side_effect=tiny_fit):
            warmed_kernel, _ = runner.prepare_compiled_functions(x, r, h, seed=7, diff_type='symmetric')
            cache_size = warmed_kernel._cache_size()
            got = runner.learn(key, x, r, h, seed=7, diff_type='symmetric', compiled_step=warmed_kernel)
            self.assertEqual(cache_size, warmed_kernel._cache_size(), 'Adaptation recompiled after warm-up')
        np.testing.assert_array_equal(got.final_key, expected_key)
        np.testing.assert_array_equal(got.targets, cf.covariance_targets)
        np.testing.assert_array_equal(got.fold_id, cf.fold_id)
        expected_stages = {f'fold_{i}': v[0] for i, v in enumerate(fold_results)}
        expected_stages.update(final_drift=drift, covariance=cov)
        for name, expected in expected_stages.items():
            actual = got.stages[name].training
            for field in ('validation_mse', 'moving_average', 'best_iteration', 'best_validation_mse', 'stopped_iteration'):
                np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
            np.testing.assert_array_equal(actual.model.omega, expected.model.omega)
            np.testing.assert_array_equal(actual.model.amp, expected.model.amp)

    def test_frozen_fitter_ties_indexing_and_no_refit(self):
        # A scripted adaptation generates losses 4, 1, 1, 9. Initialization
        # would have zero loss; it must not enter the selection candidates.
        values = iter([2., 1., 1., 3.])
        calls = []
        def step(key, model, x, y):
            calls.append(1)
            value = next(values)
            return key, ARFFModel(model.omega, jnp.array([[value], [0.]]))
        _, selected = fit_validation_selected_arff(
            jax.random.PRNGKey(0), jnp.zeros((20, 1)), jnp.zeros((20, 1)), K=1,
            M_min=4, M_max=4, lambda_reg=.001, gamma=1., delta=.2, resampling=False,
            metropolis_test=True, compiled_adaptation_step=step)
        self.assertEqual(len(calls), 4)
        self.assertEqual(selected.best_iteration, 2)
        np.testing.assert_array_equal(selected.validation_mse, [4., 1., 1., 9.])
        np.testing.assert_array_equal(selected.model.amp, [[1.], [0.]])

    def make_archive(self):
        data, result, metadata = fixture()
        metrics = runner.evaluate_final(result, data, get_experiment('ex8'))
        return data, result, runner.build_artifact(result, data, metrics, seed=0, compilation_time=2.,
                                                   metadata=metadata, other_times={})

    def test_non_spd_metrics_and_archive_roundtrip(self):
        data, result, arrays = self.make_archive()
        for label in ('train', 'validation', 'test'):
            self.assertEqual(float(arrays[f'{label}_raw_spd_violation_rate']), 1.)
            self.assertLess(float(arrays[f'{label}_min_raw_eigenvalue']), 0.)
            self.assertGreater(float(arrays[f'{label}_min_projected_eigenvalue']), 0.)
        runner.validate_artifact(arrays)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.npz'
            runner.save_artifact(path, arrays, result.model, data.x[:4])
            with np.load(path, allow_pickle=False) as saved:
                for key, value in arrays.items():
                    np.testing.assert_array_equal(saved[key], value)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                runner.save_artifact(path, arrays, result.model, data.x[:4])
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual([p.name for p in Path(directory).iterdir()], ['result.npz'])

    def test_archive_rejects_corrupt_selection_splits_and_clocks(self):
        _, _, arrays = self.make_archive()
        mutations = [('final_drift_best_iteration', np.asarray(2)),
                     ('fold_0_validation_seed', np.asarray(999)),
                     ('fold_id', np.zeros(80, dtype=np.int32)),
                     ('covariance_cumulative_time', np.arange(300, 0, -1).astype(float)),
                     ('K', np.asarray(512))]
        for key, value in mutations:
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_artifact(dict(arrays, **{key: value}))

    def test_infinite_unselected_history_is_preserved(self):
        _, _, arrays = self.make_archive()
        arrays['final_drift_validation_mse'][10] = np.inf
        arrays['final_drift_moving_average'][10:15] = np.inf
        runner.validate_artifact(arrays)

    def test_main_freezes_models_before_test_access_and_keeps_bookkeeping_outside_fit(self):
        data, result, metadata = fixture()
        events = []
        class GuardedArray:
            def __init__(self, array):
                self.array, self.shape, self.dtype = array, array.shape, array.dtype
            def __getitem__(self, idx):
                if np.any(np.asarray(idx) >= 90):
                    self_test.assertIn('fit_completed', events)
                    events.append('test_access')
                return self.array[idx]
        self_test = self
        guarded = SimpleNamespace(x=GuardedArray(data.x), r=GuardedArray(data.r), h=GuardedArray(data.h),
                                  train_idx=data.train_idx, validation_idx=data.validation_idx, test_idx=data.test_idx)
        def fit(key, x, r, h, **kwargs):
            self.assertEqual(events, ['metadata', 'warmup'])
            self.assertEqual(len(x), 80)
            events.append('fit_completed')
            return result
        def warm(*args, **kwargs):
            events.append('warmup')
            return None, 2.
        def meta(*args):
            events.append('metadata')
            return metadata
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
                patch.object(runner, 'provenance', side_effect=meta), \
                patch.object(runner, 'load_dataset', return_value=guarded), \
                patch.object(runner, 'prepare_compiled_functions', side_effect=warm), \
                patch.object(runner, 'learn', side_effect=fit) as fit_mock:
            path = Path(directory) / 'production.npz'
            runner.main(['--seed', '0', '--artifact-path', str(path)])
            self.assertEqual(fit_mock.call_count, 1)
            self.assertIn('test_access', events)
            with np.load(path, allow_pickle=False) as z:
                self.assertEqual(float(z['algorithm_time']), 7.)
                self.assertEqual(float(z['end_to_end_time']), 9.)

    def test_cli_rejects_bad_seed_and_existing_path_before_work(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()), \
                patch.object(runner, 'load_dataset') as load, patch.object(runner, 'provenance') as metadata:
            path = Path(directory) / 'existing.npz'
            path.write_bytes(b'preserve')
            for argv in (['--seed', '-1'], ['--seed', str(2**32)], ['--artifact-path', str(path)],
                         ['--artifact-path', str(Path(directory) / 'bad.txt')]):
                with self.assertRaises(SystemExit) as raised:
                    runner.main(argv)
                self.assertEqual(raised.exception.code, 2)
            load.assert_not_called()
            metadata.assert_not_called()
            self.assertEqual(path.read_bytes(), b'preserve')


if __name__ == '__main__':
    unittest.main(verbosity=2)
