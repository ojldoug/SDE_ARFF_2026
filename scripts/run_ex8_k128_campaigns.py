#!/usr/bin/env python3
"""Persistent four-GPU K=128 Fourier campaigns; numerical runners are unchanged.

prepare freezes provenance and reserves two fresh K=128 campaign directories.
run joint / run split are intended for separate persistent tmux sessions.
Fourier waits for a validated complete Joint Fourier campaign. status is read-only.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
# Supervisor imports only: no device initialization or numerical fitting.
os.environ['JAX_PLATFORMS'] = 'cpu'
import run_ex8_arff_validation_selected_crossfit as arff
import run_production_batch as batch
from src.arff.two_stage import make_folds

BASE = ROOT / 'results/production'
CONTROL = BASE / 'ex8_k128_campaign_control'
DIRECTORIES = {'joint': BASE / 'adam_fourier_ex8_k128', 'split': BASE / 'adam_split_fourier_ex8_k128'}
RUNNERS = {'joint': 'scripts/run_ex8_fourier_width.py', 'split': 'scripts/run_ex8_fourier_width.py'}
ACCEPTED = {'joint': 'scripts/run_adam_fourier_experiment.py', 'split': 'scripts/run_adam_split_fourier_experiment.py'}
SESSIONS = {'joint': 'joint_fourier_ex8_k128_campaign', 'split': 'split_fourier_ex8_k128_campaign'}
PYTHON = '/home/kammonaa/miniconda3/envs/arff-sde/bin/python'
STATE_LOCK = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return arff.sha256_file(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, *, exclusive=False):
    """Atomic state replacement; exclusive publication for immutable manifests."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(tmp, path)
        else:
            os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def paths(method, seed):
    directory = DIRECTORIES[method]
    return directory / f'seed_{seed}_artifacts.npz', directory / f'seed_{seed}.txt'


def validate_pair(method, seed, artifact, log):
    if not artifact.is_file() or not log.is_file():
        raise ValueError(f'Missing artifact/log pair: {artifact}, {log}')
    bm = 'adam' if method == 'joint' else 'adam_split'
    if not batch.seed_is_complete(log_path=log,artifact_path=artifact,method=bm,experiment='ex8',seed=seed):
        raise ValueError('Production artifact/log validator failed')
    with np.load(artifact,allow_pickle=False) as z:
        a={k:z[k] for k in z.files}
    tree=ast.parse((ROOT/ACCEPTED[method]).read_text())
    schema=next({k.arg for k in n.keywords} for n in ast.walk(tree)
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='savez_compressed')
    if set(a)!=schema:
        raise ValueError('Artifact schema differs from the accepted runner')
    expected=dict(artifact_version=1 if method=='joint' else 4,
        method='adam_fourier' if method=='joint' else 'adam_split_fourier',
        seed=seed,experiment='ex8',fourier_frequencies=128,batch_size=256,learning_rate=1e-4,diff_type='symmetric')
    expected['epochs' if method=='joint' else 'epochs_per_regression']=300
    if method=='split':
        expected.update(n_folds=5,fold_seed=2026,n_train=80000,n_validation=10000,n_test=10000,jax_backend='gpu')
    for k,v in expected.items():
        if a[k].item()!=v:raise ValueError(f'Configuration mismatch: {k}')
    for k,shape in dict(drift_omega=(2,128),drift_amp=(256,2),covariance_omega=(2,128),covariance_amp=(256,3)).items():
        if a[k].shape!=shape:raise ValueError(f'Width/shape mismatch: {k}')
    for k,v in a.items():
        if v.dtype.hasobject or (np.issubdtype(v.dtype,np.number) and not np.all(np.isfinite(v))):
            raise ValueError(f'Invalid array: {k}')
    if sum(a[k].size for k in ['drift_omega','drift_amp','covariance_omega','covariance_amp'])!=1792:
        raise ValueError('Final trainable parameter count is not 1792')
    stages=[('', 'nll')] if method=='joint' else [('final_drift_','mse'),('covariance_','nll')]
    for prefix,loss in stages:
        h=a[prefix+'validation_'+loss]; best=int(a[prefix+'best_epoch'])
        if h.shape!=(300,) or best!=int(np.argmin(h)) or h[best]!=a[prefix+'best_validation_'+loss]:
            raise ValueError('Checkpoint selection/history mismatch')
    if method=='split':
        fid=np.empty(80000,dtype=np.int32)
        for i,fold in enumerate(make_folds(80000,5,2026)):fid[fold]=i
        np.testing.assert_array_equal(a['fold_id'],fid)
    import re
    text=log.read_text()
    for label,value in [('backend','gpu'),('train N','80000'),('validation','10000'),('test','10000'),('frequencies','128')]:
        if not re.search(r'^'+re.escape(label)+r'\s*:\s*'+value+r'\s*$',text,re.M):
            raise ValueError(f'Log configuration mismatch: {label}')
    return dict(artifact_sha256=digest(artifact),log_sha256=digest(log),
        algorithm_time=float(a['algorithm_time']),compilation_time=float(a['compilation_time']),end_to_end_time=float(a['end_to_end_time']))


def copy_exclusive(source, destination):
    """Publish a byte-identical copy; never overwrite or relabel the original."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        if digest(source) != digest(destination):
            raise FileExistsError(f'Different existing output: {destination}')
        return
    fd, temp = tempfile.mkstemp(dir=destination.parent, prefix='.reuse-')
    try:
        with source.open('rb') as src, os.fdopen(fd, 'wb') as dst:
            for chunk in iter(lambda: src.read(1024 * 1024), b''):
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        if digest(temp) != digest(source):
            raise RuntimeError('Reuse copy checksum mismatch')
        os.link(temp, destination)
    finally:
        os.unlink(temp)


def verify_snapshot():
    manifest = read_json(CONTROL / 'manifest.json')
    for name, sha in manifest['source_sha256'].items():
        if digest(ROOT / name) != sha:
            raise RuntimeError(f'Frozen campaign source changed: {name}')
    if digest(ROOT / 'data/ex8.npz') != manifest['dataset_sha256']:
        raise RuntimeError('Canonical dataset changed')
    return manifest


def prepare():
    if (CONTROL / 'manifest.json').exists():
        verify_snapshot()
        print('Existing K=128 manifest verified; outputs unchanged.')
        return
    for directory in DIRECTORIES.values():
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f'Unexpected existing campaign directory: {directory}')
    for directory in DIRECTORIES.values():
        directory.mkdir(parents=True, exist_ok=True)
    source_paths = sorted(set([str(p.relative_to(ROOT)) for p in (ROOT/'src').rglob('*.py')]
        + list(ACCEPTED.values()) + ['scripts/run_ex8_fourier_width.py', 'scripts/run_ex8_k128_campaigns.py',
            'scripts/summarize_ex8_k128_campaigns.py', 'scripts/run_production_batch.py']))
    manifest = dict(created_at=now(), purpose='width-controlled accuracy/reproducibility',
        fourier_frequencies=128, sole_scientific_change='fourier_frequencies: 512 -> 128',
        seeds=list(range(30)), gpus=[0,1,2,3], sessions=SESSIONS, python=PYTHON,
        assignment='seed modulo 4; Joint completes and validates before Split dispatch',
        reused={m:dict(seed=-1) for m in DIRECTORIES},
        timing_context='New jobs run concurrently across four GPUs; shared CPU/memory/PCIe contention. Not isolated publication timing.',
        dataset_sha256=digest(ROOT/'data/ex8.npz'), source_sha256={p:digest(ROOT/p) for p in source_paths})
    write_json(CONTROL/'manifest.json', manifest, exclusive=True)
    for method in DIRECTORIES:
        write_json(DIRECTORIES[method]/'campaign_state.json',
            dict(method=method,status='prepared',updated_at=now(),seeds={}),exclusive=True)
    print('Prepared two new K=128 campaigns; no K=512 artifact reused.')


def update(method, *, status=None, seed=None, record=None):
    with STATE_LOCK:
        path = DIRECTORIES[method]/'campaign_state.json'
        state = read_json(path)
        state['updated_at'] = now()
        if status is not None:
            state['status'] = status
        if seed is not None:
            state['seeds'][str(seed)] = record
        write_json(path, state)


@contextmanager
def lock_file(path):
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield fd
    finally:
        # Do not explicitly unlock: an inherited descriptor keeps the lock
        # alive if the supervisor dies while its benchmark child is running.
        os.close(fd)


def gpu_processes():
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                    '--format=csv,noheader,nounits'], text=True)
    return [line.strip().split(',', 2) for line in text.splitlines() if line.strip()]


def gpu_uuid(gpu):
    return subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=uuid',
                                    '--format=csv,noheader'], text=True).strip()


def run_job(method, seed, gpu, stop):
    artifact, log = paths(method, seed)
    reservation = DIRECTORIES[method]/'state'/f'seed_{seed}'
    # Whole-process GPU lock shared by both campaigns, inherited by the child.
    with lock_file(CONTROL/f'gpu_{gpu}.lock') as gpu_lock:
        uuid = gpu_uuid(gpu)
        while any(row[0].strip() == uuid for row in gpu_processes()):
            update(method, seed=seed, record=dict(status='waiting_gpu', gpu=gpu, updated_at=now()))
            if stop.wait(15):
                return
        if stop.is_set():
            return
        verify_snapshot()
        if artifact.exists() or log.exists() or reservation.exists():
            raise FileExistsError(f'Incomplete/suspicious output or reservation for {method} seed {seed}; refusing retry')
        reservation.parent.mkdir(exist_ok=True)
        reservation.mkdir()  # Exclusive durable reservation; never automatically removed.
        staged = reservation/f'seed_{seed}_artifacts.npz'
        command = [PYTHON, '-B', '-u', str(ROOT/RUNNERS[method])]
        command += [method, '--fourier-frequencies', '128']
        command += ['--seed', str(seed), '--artifact-path', str(staged)]
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), JAX_PLATFORMS='cuda',
                   CONDA_PREFIX=str(Path(PYTHON).parent.parent), CONDA_DEFAULT_ENV='arff-sde')
        env['PATH'] = str(Path(PYTHON).parent) + os.pathsep + env.get('PATH','')
        record = dict(status='reserved', seed=seed, gpu=gpu, gpu_uuid=uuid, started_at=now(),
                      command=command, final_artifact=str(artifact), staged_artifact=str(staged), log=str(log),
                      timing_context='parallel four-GPU accuracy campaign; not isolated publication timing')
        write_json(reservation/'job.json', record, exclusive=True)
        # Check again after bookkeeping, immediately before the only launch.
        while any(row[0].strip() == uuid for row in gpu_processes()):
            if stop.wait(15):
                raise RuntimeError('Stopped before launch; reservation preserved')
        with log.open('x') as handle:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                       pass_fds=(gpu_lock,))
            record.update(status='running', pid=process.pid)
            write_json(reservation/'job.json', record)
            update(method, seed=seed, record=record)
            print(f'{now()} START {method} seed={seed} gpu={gpu} pid={process.pid}', flush=True)
            code = process.wait()
        record.update(returncode=code, finished_at=now())
        write_json(reservation/'exit.json', record, exclusive=True)
        if code:
            raise RuntimeError(f'{method} seed {seed} exited {code}; log preserved at {log}')
        checked = validate_pair(method, seed, staged, log)
        verify_snapshot()
        os.link(staged, artifact)  # Atomic no-clobber publication after validation.
        record.update(status='complete', **checked)
        write_json(reservation/'job.json', record)
        update(method, seed=seed, record=record)
        print(f'{now()} COMPLETE {method} seed={seed} gpu={gpu} algorithm={checked["algorithm_time"]:.3f}s', flush=True)


def completed_existing(method, seed):
    artifact, log = paths(method, seed)
    if not artifact.exists() and not log.exists():
        return False
    checked = validate_pair(method, seed, artifact, log)
    manifest = read_json(CONTROL/'manifest.json')
    reuse = manifest['reused'][method]
    if seed == reuse['seed']:
        if any(checked[k] != reuse[k] for k in ('artifact_sha256','log_sha256')):
            raise ValueError('Approved reuse pair was changed')
        record = dict(status='reused', timing_context=reuse['timing_context'], **checked)
    else:
        job = DIRECTORIES[method]/'state'/f'seed_{seed}'/'job.json'
        if not job.exists():
            raise ValueError('Unexpected unreserved completed artifact')
        record = read_json(job)
        for k in ('artifact_sha256','log_sha256'):
            if k in record and record[k] != checked[k]:
                raise ValueError('Completed artifact/log changed')
        record.update(status='complete', **checked)
    update(method, seed=seed, record=record)
    return True


def wait_for_arff():
    while True:
        state = read_json(DIRECTORIES['joint']/'campaign_state.json')
        if state['status'] in ('failed','blocked'):
            raise RuntimeError('Joint Fourier campaign failed/blocked; Fourier will not launch')
        if state['status'] == 'complete':
            for seed in range(30):
                validate_pair('joint', seed, *paths('joint', seed))
            return
        time.sleep(15)


def final_report():
    command=[PYTHON,'-B','-u',str(ROOT/'scripts/summarize_ex8_k128_campaigns.py')]
    with (CONTROL/'summary_console.txt').open('x') as handle:
        completed=subprocess.run(command,cwd=ROOT,env=dict(os.environ,JAX_PLATFORMS='cpu',MPLBACKEND='Agg'),
            stdout=handle,stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError('Summary failed; inspect summary_console.txt; campaign artifacts preserved')
    print('Validated width-controlled summary and new RMSE figures saved.',flush=True)


def run(method):
    verify_snapshot()
    with lock_file(CONTROL/f'{method}.supervisor.lock'):
        update(method, status='waiting_for_joint' if method=='split' else 'starting')
        try:
            if method=='split':
                print('Waiting for all 30 Joint Fourier seeds to validate before Split Fourier dispatch.', flush=True)
                wait_for_arff()
            verify_snapshot()
            pending = [s for s in range(30) if not completed_existing(method,s)]
            # Initial whole-machine-idle requirement; subsequent dispatch checks
            # the affected GPU and permits other approved workers on other GPUs.
            while gpu_processes():
                update(method,status='waiting_machine_idle')
                time.sleep(15)
            update(method,status='running')
            stop = threading.Event()
            errors = []
            def worker(gpu):
                for seed in pending:
                    if seed % 4 != gpu or stop.is_set():
                        continue
                    try:
                        run_job(method,seed,gpu,stop)
                    except Exception as exc:
                        stop.set()
                        errors.append(str(exc))
                        update(method,seed=seed,record=dict(status='failed',gpu=gpu,error=str(exc),updated_at=now()))
                        print(f'{now()} FAILED {method} seed={seed}: {exc}',flush=True)
                        return
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(worker,range(4)))
            if errors:
                raise RuntimeError('; '.join(errors))
            for seed in range(30):
                if not completed_existing(method,seed):
                    raise RuntimeError(f'Missing completed seed {seed}')
            update(method,status='complete')
            print(f'{now()} CAMPAIGN COMPLETE: {method}, 30 validated seeds',flush=True)
            if method=='split':
                final_report()
        except Exception:
            update(method,status='failed')
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','status'])
    parser.add_argument('method',nargs='?',choices=['joint','split'])
    args=parser.parse_args()
    if args.action=='prepare':
        prepare()
    elif args.action=='run':
        if not args.method:
            parser.error('run requires arff or fourier')
        run(args.method)
    else:
        for method in DIRECTORIES:
            p=DIRECTORIES[method]/'campaign_state.json'
            if p.exists():
                state=read_json(p)
                counts={}
                for v in state['seeds'].values():counts[v['status']]=counts.get(v['status'],0)+1
                print(method,state['status'],counts,'updated',state['updated_at'])
            else:
                print(method,'not prepared')


if __name__=='__main__':
    main()
