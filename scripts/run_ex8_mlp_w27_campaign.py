#!/usr/bin/env python3
"""Persistent four-GPU width-27 Joint MLP campaign; accepted numerical runner unchanged."""
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
CONTROL = BASE / 'ex8_mlp_w27_campaign_control'
DIRECTORIES = {'mlp': BASE / 'adam_mlp_ex8_w27'}
RUNNERS = {'mlp': 'scripts/run_ex8_mlp_w27.py'}
ACCEPTED = {'mlp': 'scripts/run_adam_mlp_experiment.py'}
SESSIONS = {'mlp': 'mlp_ex8_w27_campaign'}
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
    if not batch.seed_is_complete(log_path=log,artifact_path=artifact,method='mlp',experiment='ex8',seed=seed):
        raise ValueError('Production artifact/log validator failed')
    with np.load(artifact,allow_pickle=False) as z:
        a={k:z[k] for k in z.files}
    with np.load(BASE/'mlp_ex8/seed_0_artifacts.npz',allow_pickle=False) as z:
        if set(a)!=set(z.files):raise ValueError('Accepted MLP artifact schema mismatch')
    expected=dict(artifact_version=1,method='adam_mlp',seed=seed,experiment='ex8',
        hidden_width=27,hidden_layers=2,mlp_parameter_count=1814,fourier_parameter_count=1792,
        batch_size=256,learning_rate=1e-3,epochs=300,diff_type='symmetric')
    for k,v in expected.items():
        if a[k].item()!=v:raise ValueError(f'Configuration mismatch: {k}')
    counts=[]
    for prefix,q in [('drift',2),('covariance',3)]:
        shapes=[(2,27),(27,27),(27,q)]
        for i,shape in enumerate(shapes):
            if a[f'{prefix}_weight_{i}'].shape!=shape or a[f'{prefix}_bias_{i}'].shape!=(shape[1],):
                raise ValueError('Architecture mismatch')
        counts.append(sum(a[f'{prefix}_{kind}_{i}'].size for kind in ('weight','bias') for i in range(3)))
    if counts!=[893,921]:raise ValueError('Actual parameter count mismatch')
    for k,v in a.items():
        if v.dtype.hasobject or (np.issubdtype(v.dtype,np.number) and not np.all(np.isfinite(v))):
            raise ValueError(f'Invalid array: {k}')
    for k in ('training_nll','validation_nll','cumulative_time'):
        if a[k].shape!=(300,):raise ValueError('History length mismatch')
    best=int(a['best_epoch']);h=a['validation_nll']
    if best!=int(np.argmin(h)) or h[best]!=a['best_validation_nll']:
        raise ValueError('Checkpoint mismatch')
    if np.any(np.diff(a['cumulative_time'])<0):raise ValueError('Timing history mismatch')
    import re
    for label,value in [('backend','gpu'),('train N','80000'),('validation','10000'),('test','10000'),('hidden','27-27'),('MLP params','1814')]:
        if not re.search(r'^'+re.escape(label)+r'\s*:\s*'+value+r'\s*$',log.read_text(),re.M):
            raise ValueError(f'Log mismatch: {label}')
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
        print('Existing width-27 manifest verified; outputs unchanged.')
        return
    for directory in DIRECTORIES.values():
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f'Unexpected existing campaign directory: {directory}')
    for directory in DIRECTORIES.values():
        directory.mkdir(parents=True, exist_ok=True)
    source_paths = sorted(set([str(p.relative_to(ROOT)) for p in (ROOT/'src').rglob('*.py')]
        + list(ACCEPTED.values()) + ['scripts/run_ex8_mlp_w27.py', 'scripts/run_ex8_mlp_w27_campaign.py',
            'scripts/summarize_ex8_capacity_matched.py', 'scripts/run_production_batch.py']))
    protected = [p for d in ('mlp_ex8','adam_fourier_ex8_k128','adam_split_fourier_ex8_k128','arff_ex8_corrected') for p in (BASE/d).glob('seed_*') if p.is_file()]
    manifest = dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
        protected_sha256={str(p.relative_to(ROOT)):digest(p) for p in protected}, created_at=now(), purpose='width-controlled accuracy/reproducibility',
        hidden_width=27, total_parameters=1814, sole_scientific_change='hidden width: 57 -> 27; budget metadata 7168 -> 1792',
        seeds=list(range(30)), gpus=[0,1,2,3], sessions=SESSIONS, python=PYTHON,
        assignment='seed modulo 4; one benchmark per GPU',
        reused={m:dict(seed=-1) for m in DIRECTORIES},
        timing_context='New jobs run concurrently across four GPUs; shared CPU/memory/PCIe contention. Not isolated publication timing.',
        dataset_sha256=digest(ROOT/'data/ex8.npz'), source_sha256={p:digest(ROOT/p) for p in source_paths})
    write_json(CONTROL/'manifest.json', manifest, exclusive=True)
    for method in DIRECTORIES:
        write_json(DIRECTORIES[method]/'campaign_state.json',
            dict(method=method,status='prepared',updated_at=now(),seeds={}),exclusive=True)
    print('Prepared new width-27 MLP campaign; no historical artifacts reused.')


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
        command += ['--hidden-width', '27']
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


def final_report():
    command=[PYTHON,'-B','-u',str(ROOT/'scripts/summarize_ex8_capacity_matched.py')]
    with (CONTROL/'summary_console.txt').open('x') as handle:
        completed=subprocess.run(command,cwd=ROOT,env=dict(os.environ,JAX_PLATFORMS='cpu',MPLBACKEND='Agg'),
            stdout=handle,stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError('Summary failed; inspect summary_console.txt; campaign artifacts preserved')
    print('Validated width-controlled summary and new RMSE figures saved.',flush=True)


def run(method):
    verify_snapshot()
    with lock_file(CONTROL/f'{method}.supervisor.lock'):
        update(method, status='starting')
        try:
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
            final_report()
        except Exception:
            update(method,status='failed')
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','status'])
    parser.add_argument('method',nargs='?',choices=['mlp'])
    args=parser.parse_args()
    if args.action=='prepare':
        prepare()
    elif args.action=='run':
        if not args.method:
            parser.error('run requires mlp')
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
