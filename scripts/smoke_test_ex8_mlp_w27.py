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
import run_ex8_mlp_w27 as wrapper
import run_adam_mlp_experiment as joint
import run_ex8_mlp_w27_campaign as campaign
from src.adam.mlp import initialize_model
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
            wrapper.main(['--inspect'])
            imp.assert_not_called()
        self.assertEqual(get_config('ex8').fourier_frequencies,512)

    def test_joint_width_reaches_initialization_warmup_fit_and_real_archive(self):
        seen=[];original=joint.get_config
        def init(*args,**kwargs):
            self.assertEqual(kwargs['hidden_sizes'],(27,27));seen.append('initialization')
            return initialize_model(*args,**kwargs)
        def timed(fn,*args,**kwargs):
            if fn is joint.fit_adam:
                self.assertEqual(args[1].drift.weights[0].shape,(2,27));seen.append('fit')
                self.assertEqual((kwargs['epochs'],kwargs['batch_size']),(300,256))
                return (args[0],history(args[1])),3.
            self.assertEqual(args[0].drift.weights[0].shape,(2,27));seen.append('warmup')
            return None,1.
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack,redirect_stdout(io.StringIO()):
            for name,value in [('load_dataset',lambda p:dataset()),('initialize_model',init),('timed_call',timed),
                ('make_compiled_adam_functions',lambda *a,**k:(NS(init=lambda m:None),object(),object())),
                ('gaussian_nll',lambda *a:1.),('true_function_errors',lambda *a,**k:(1.,1.)),
                ('predict_covariance',lambda m,x:np.tile(np.eye(2),(len(x),1,1)))]:
                stack.enter_context(patch.object(joint,name,side_effect=value))
            path=Path(directory)/'joint.npz'
            wrapper.main(['--seed','9','--artifact-path',str(path)])
            with np.load(path,allow_pickle=False) as z:
                self.assertEqual(int(z['hidden_width']),27)
                self.assertEqual(z['drift_weight_0'].shape,(2,27))
                self.assertEqual(z['covariance_weight_2'].shape,(27,3))
                self.assertEqual(int(z['seed']),9)
                self.assertEqual(int(z['mlp_parameter_count']),1814)
                self.assertEqual(sum(z[k].size for k in z.files if '_weight_' in k or '_bias_' in k),1814)
        self.assertEqual(seen,['initialization','warmup','warmup','warmup','fit'])
        self.assertIs(joint.get_config,original)
        self.assertEqual(get_config('ex8').fourier_frequencies,512)

    def test_refuse_existing_artifact_and_wrong_width(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stderr(io.StringIO()),patch.object(wrapper.importlib,'import_module') as imp:
            p=Path(directory)/'existing.npz';p.write_bytes(b'keep')
            for argv in [['--artifact-path',str(p)],['--hidden-width','57','--inspect']]:
                with self.assertRaises(SystemExit):wrapper.main(argv)
            self.assertEqual(p.read_bytes(),b'keep');imp.assert_not_called()

    def test_campaign_validator_rejects_old_width(self):
        p=Path(__file__).resolve().parents[1]/'results/production/mlp_ex8'
        with self.assertRaisesRegex(ValueError,'hidden_width'):
            campaign.validate_pair('mlp',0,p/'seed_0_artifacts.npz',p/'seed_0.txt')

if __name__=='__main__':unittest.main(verbosity=2)
