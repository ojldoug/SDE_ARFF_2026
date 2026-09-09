#!/usr/bin/env python3
"""Non-training checks: actual runner main/config/serialization, mocked updates."""
import os
os.environ['JAX_PLATFORMS']='cpu'
from contextlib import ExitStack,redirect_stdout,redirect_stderr
from dataclasses import asdict
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_ex8_fourier_width as wrapper
import run_adam_fourier_experiment as joint
import run_adam_split_fourier_experiment as split
import run_ex8_k128_campaigns as campaign
from src.adam.fourier import initialize_model
from src.experiments.config import get_config


def dataset():
    return NS(x=np.zeros((400,2),dtype=np.float32),r=np.zeros((400,2),dtype=np.float32),
        h=np.full((400,1),.001,dtype=np.float32),train_idx=np.arange(320),validation_idx=np.arange(320,360),test_idx=np.arange(360,400))


def history(model):
    return NS(model=model,best_epoch=0,best_validation_nll=1.,training_nll=np.ones(300),
              validation_nll=np.ones(300),cumulative_time=np.linspace(.01,3.,300))


class WidthTests(unittest.TestCase):
    def test_only_width_changes_and_inspect_does_not_import_runner(self):
        base=asdict(get_config('ex8'));effective=asdict(wrapper.effective_config('ex8'))
        self.assertEqual({k for k in base if base[k]!=effective[k]},{'fourier_frequencies'})
        self.assertEqual(effective['fourier_frequencies'],128)
        with patch.object(wrapper.importlib,'import_module') as imp,redirect_stdout(io.StringIO()):
            wrapper.main(['joint','--inspect'])
            imp.assert_not_called()
        self.assertEqual(get_config('ex8').fourier_frequencies,512)

    def test_joint_width_reaches_initialization_warmup_fit_and_real_archive(self):
        seen=[];original=joint.get_config
        def init(*args,**kwargs):
            self.assertEqual(kwargs['n_frequencies'],128);seen.append('initialization')
            return initialize_model(*args,**kwargs)
        def timed(fn,*args,**kwargs):
            if fn is joint.fit_adam_fourier:
                self.assertEqual(args[1].drift.omega.shape,(2,128));seen.append('fit')
                self.assertEqual((kwargs['epochs'],kwargs['batch_size']),(300,256))
                return (args[0],history(args[1])),3.
            self.assertEqual(args[0].drift.omega.shape,(2,128));seen.append('warmup')
            return None,1.
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack,redirect_stdout(io.StringIO()):
            for name,value in [('load_dataset',lambda p:dataset()),('initialize_model',init),('timed_call',timed),
                ('make_compiled_adam_functions',lambda *a,**k:(NS(init=lambda m:None),object(),object())),
                ('gaussian_nll',lambda *a:1.),('true_function_errors',lambda *a,**k:(1.,1.)),
                ('predict_covariance',lambda m,x:np.tile(np.eye(2),(len(x),1,1)))]:
                stack.enter_context(patch.object(joint,name,side_effect=value))
            path=Path(directory)/'joint.npz'
            wrapper.main(['joint','--seed','9','--artifact-path',str(path)])
            with np.load(path,allow_pickle=False) as z:
                self.assertEqual(int(z['fourier_frequencies']),128)
                self.assertEqual(z['drift_omega'].shape,(2,128))
                self.assertEqual(z['covariance_amp'].shape,(256,3))
                self.assertEqual(int(z['seed']),9)
        self.assertEqual(seen,['initialization','warmup','warmup','warmup','fit'])
        self.assertIs(joint.get_config,original)
        self.assertEqual(get_config('ex8').fourier_frequencies,512)

    def test_split_width_reaches_warmup_fit_and_real_archive(self):
        import jax
        model=initialize_model(jax.random.PRNGKey(1),input_dimension=2,output_dimension=2,n_frequencies=128,diff_type='symmetric')
        seen=[];original=split.get_config
        def warm(**kwargs):
            self.assertEqual(kwargs['n_frequencies'],128);seen.append('warmup')
            # Inspect the real warm-up initializer, stopping before any update.
            class StopInspection(Exception):pass
            def spy(*args,**kw):
                self.assertEqual(kw['n_frequencies'],128);seen.append('warmup_initialization');raise StopInspection()
            with patch.object(split,'initialize_model',side_effect=spy):
                with self.assertRaises(StopInspection):original_warm(**kwargs)
            return (None,)*6+(3.,)
        def timed(fn,*args,**kwargs):
            self.assertEqual(kwargs['n_frequencies'],128)
            self.assertEqual((kwargs['epochs'],kwargs['batch_size'],kwargs['n_folds'],kwargs['fold_seed']),(300,256,5,2026))
            seen.append('fit')
            import src.adam.split_fourier as core
            class StopInspection(Exception):pass
            def spy(*a,**kw):
                self.assertEqual(kw['n_frequencies'],128);seen.append('fit_initialization');raise StopInspection()
            with patch.object(core,'initialize_model',side_effect=spy):
                with self.assertRaises(StopInspection):fn(*args,**kwargs)
            cf=NS(fold_algorithm_times=np.ones(5),crossfit_algorithm_time=5.,fold_best_epochs=np.zeros(5,dtype=int),
                  fold_best_validation_losses=np.ones(5),fold_id=np.arange(320)%5)
            result=NS(model=model,final_drift_training=history(model.drift),covariance_training=history(model),crossfit=cf,
                      final_drift_start_offset=5.,final_drift_algorithm_time=3.,covariance_start_offset=8.,
                      covariance_algorithm_time=3.,internal_algorithm_time=11.)
            return (args[0],result),11.
        original_warm=split.prepare_compiled_functions
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack,redirect_stdout(io.StringIO()):
            stack.enter_context(patch.object(split,'load_dataset',return_value=dataset()))
            stack.enter_context(patch.object(split,'prepare_compiled_functions',side_effect=warm))
            stack.enter_context(patch.object(split,'timed_call',side_effect=timed))
            stack.enter_context(patch.object(split,'evaluate_split',return_value=dict(nll=1.,drift_rmse=1.,covariance_rmse=1.,min_covariance_eig=1.,max_covariance_eig=1.)))
            path=Path(directory)/'split.npz'
            wrapper.main(['split','--seed','9','--artifact-path',str(path)])
            with np.load(path,allow_pickle=False) as z:
                self.assertEqual(int(z['fourier_frequencies']),128)
                self.assertEqual(z['drift_omega'].shape,(2,128))
                self.assertEqual(z['covariance_amp'].shape,(256,3))
                self.assertEqual(int(z['seed']),9)
        self.assertEqual(seen,['warmup','warmup_initialization','fit','fit_initialization'])
        self.assertIs(split.get_config,original)

    def test_refuse_existing_artifact_and_wrong_width(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stderr(io.StringIO()),patch.object(wrapper.importlib,'import_module') as imp:
            p=Path(directory)/'existing.npz';p.write_bytes(b'keep')
            for argv in [['joint','--artifact-path',str(p)],['split','--fourier-frequencies','512','--inspect']]:
                with self.assertRaises(SystemExit):wrapper.main(argv)
            self.assertEqual(p.read_bytes(),b'keep');imp.assert_not_called()

    def test_campaign_validator_rejects_both_old_widths(self):
        root=Path(__file__).resolve().parents[1]
        for method,directory in [('joint','adam_ex8'),('split','adam_split_ex8')]:
            p=root/'results/production'/directory
            with self.assertRaisesRegex(ValueError,'fourier_frequencies'):
                campaign.validate_pair(method,0,p/'seed_0_artifacts.npz',p/'seed_0.txt')


if __name__=='__main__':unittest.main(verbosity=2)
