#!/usr/bin/env python3
"""Supervisor-only tests. All benchmark subprocess launches are mocked."""
import os
os.environ['JAX_PLATFORMS']='cpu'
import sys
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_ex8_campaigns as c


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.dirs={m:self.root/m for m in ('arff','fourier')}
        for p in self.dirs.values():p.mkdir()
        self.control=self.root/'control'
        self.control.mkdir()
        self.patches=[patch.object(c,'DIRECTORIES',self.dirs),patch.object(c,'CONTROL',self.control)]
        for p in self.patches:p.start()
        for m in self.dirs:c.write_json(self.dirs[m]/'campaign_state.json',dict(status='prepared',seeds={}))

    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()

    def test_exclusive_copy_and_manifest(self):
        src=self.root/'source';dst=self.root/'dest'
        src.write_bytes(b'original')
        c.copy_exclusive(src,dst)
        self.assertEqual(src.read_bytes(),dst.read_bytes())
        c.copy_exclusive(src,dst)
        src.write_bytes(b'different')
        with self.assertRaises(FileExistsError):c.copy_exclusive(src,dst)
        self.assertEqual(dst.read_bytes(),b'original')
        p=self.root/'manifest.json'
        c.write_json(p,{'a':1},exclusive=True)
        with self.assertRaises(FileExistsError):c.write_json(p,{'a':2},exclusive=True)
        self.assertEqual(c.read_json(p),{'a':1})

    def test_gpu_lock_excludes_second_worker(self):
        with c.lock_file(self.control/'gpu_0.lock'):
            with self.assertRaises(BlockingIOError):
                with c.lock_file(self.control/'gpu_0.lock'):pass
        with c.lock_file(self.control/'gpu_0.lock'):pass

    def test_busy_gpu_does_not_dispatch(self):
        stop=threading.Event();stop.set()
        with patch.object(c,'gpu_uuid',return_value='gpu0'),patch.object(c,'gpu_processes',return_value=[['gpu0','9','other']]),patch.object(c.subprocess,'Popen') as launch:
            c.run_job('arff',0,0,stop)
            launch.assert_not_called()
        self.assertFalse(c.paths('arff',0)[1].exists())

    def test_failed_job_preserves_log_and_reservation(self):
        class Failed:
            pid=123
            def wait(self):return 2
        def launch(*args,**kwargs):
            kwargs['stdout'].write('failure evidence\n')
            return Failed()
        with patch.object(c,'gpu_uuid',return_value='gpu0'),patch.object(c,'gpu_processes',return_value=[]),patch.object(c,'verify_snapshot'),patch.object(c.subprocess,'Popen',side_effect=launch) as started:
            with self.assertRaises(RuntimeError):c.run_job('arff',0,0,threading.Event())
            with self.assertRaises(FileExistsError):c.run_job('arff',0,0,threading.Event())
            self.assertEqual(started.call_count,1)
        artifact,log=c.paths('arff',0)
        self.assertEqual(log.read_text(),'failure evidence\n')
        self.assertFalse(artifact.exists())
        self.assertTrue((self.dirs['arff']/'state/seed_0/exit.json').exists())

    def test_success_validates_then_publishes_without_overwrite(self):
        calls=[]
        class Finished:
            pid=234
            def wait(self):return 0
        def launch(command,**kwargs):
            calls.append('launch')
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'2')
            self.assertEqual(kwargs['env']['JAX_PLATFORMS'],'cuda')
            self.assertIn('--seed',command)
            Path(command[-1]).write_bytes(b'fixture artifact')
            kwargs['stdout'].write('completed fixture log\n')
            return Finished()
        def validate(*args):
            calls.append('validate')
            self.assertFalse(c.paths('arff',2)[0].exists())
            return dict(artifact_sha256='fixture',log_sha256='fixture',algorithm_time=1.)
        with patch.object(c,'gpu_uuid',return_value='gpu2'),patch.object(c,'gpu_processes',return_value=[]),patch.object(c,'verify_snapshot'),patch.object(c,'validate_pair',side_effect=validate),patch.object(c.subprocess,'Popen',side_effect=launch):
            c.run_job('arff',2,2,threading.Event())
        self.assertEqual(calls,['launch','validate'])
        self.assertEqual(c.paths('arff',2)[0].read_bytes(),b'fixture artifact')
        self.assertEqual(c.read_json(self.dirs['arff']/'campaign_state.json')['seeds']['2']['status'],'complete')

    def test_fourier_is_blocked_by_arff_failure(self):
        c.update('arff',status='failed')
        with self.assertRaises(RuntimeError),patch.object(c.subprocess,'Popen') as launch:
            c.wait_for_arff()
        launch.assert_not_called()


if __name__=='__main__':unittest.main(verbosity=2)
