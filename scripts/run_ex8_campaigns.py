#!/usr/bin/env python3
"""Persistent four-GPU accuracy campaigns; numerical runners are unchanged.

prepare validates/copies the two approved reuse pairs and freezes provenance.
run arff / run fourier are intended for separate persistent tmux sessions.
Fourier waits for a validated complete ARFF campaign. status is read-only.
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
CONTROL = BASE / 'ex8_campaign_control'
DIRECTORIES = {'arff': BASE / 'arff_ex8_corrected', 'fourier': BASE / 'adam_split_ex8'}
RUNNERS = {'arff': 'scripts/run_ex8_arff_validation_selected_crossfit.py',
           'fourier': 'scripts/run_adam_split_fourier_experiment.py'}
SESSIONS = {'arff': 'arff_ex8_corrected_campaign', 'fourier': 'split_fourier_ex8_campaign'}
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
    with np.load(artifact, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    if int(a['seed']) != seed or str(a['experiment']) != 'ex8' or str(a['jax_backend']) != 'gpu':
        raise ValueError('Wrong seed/experiment/backend')
    if method == 'arff':
        arff.validate_artifact(a)
        text = log.read_text()
        if not all(v in text for v in ('runner wall time:', 'artifact   :', '\ntest\n')):
            raise ValueError('ARFF log is incomplete')
        if a['dataset_sha256'].item() != digest(ROOT / 'data/ex8.npz'):
            raise ValueError('ARFF dataset checksum mismatch')
        for source, sha in json.loads(a['source_sha256_json'].item()).items():
            if digest(ROOT / source) != sha:
                raise ValueError(f'ARFF source checksum mismatch: {source}')
    else:
        if not batch.seed_is_complete(log_path=log, artifact_path=artifact, method='adam_split',
                                      experiment='ex8', seed=seed):
            raise ValueError('Fourier production validator rejected pair')
        tree = ast.parse((ROOT / RUNNERS['fourier']).read_text())
        schema = next({k.arg for k in n.keywords} for n in ast.walk(tree)
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                      and n.func.attr == 'savez_compressed')
        if set(a) != schema:
            raise ValueError('Fourier schema is not exactly the accepted v4 schema')
        expected = dict(artifact_version=4, method='adam_split_fourier', fourier_frequencies=512,
                        n_folds=5, fold_seed=2026, epochs_per_regression=300, batch_size=256,
                        learning_rate=1e-4, n_train=80000, n_validation=10000, n_test=10000,
                        diff_type='symmetric')
        for key, value in expected.items():
            if a[key].item() != value:
                raise ValueError(f'Fourier configuration mismatch: {key}')
        expected_folds = np.empty(80000, dtype=np.int32)
        for i, fold in enumerate(make_folds(80000, 5, 2026)):
            expected_folds[fold] = i
        np.testing.assert_array_equal(a['fold_id'], expected_folds)
        for key, shape in dict(drift_omega=(2,512), drift_amp=(1024,2),
                               covariance_omega=(2,512), covariance_amp=(1024,3)).items():
            if a[key].shape != shape:
                raise ValueError(f'Invalid shape: {key}')
        for stage, loss in [('final_drift','mse'), ('covariance','nll')]:
            history = a[f'{stage}_validation_{loss}']
            best = int(a[f'{stage}_best_epoch'])
            if history.shape != (300,) or best != int(np.argmin(history)):
                raise ValueError('Invalid Fourier checkpoint selection')
            if history[best] != a[f'{stage}_best_validation_{loss}']:
                raise ValueError('Invalid Fourier best loss')
    return {'artifact_sha256': digest(artifact), 'log_sha256': digest(log),
            'algorithm_time': float(a['algorithm_time']), 'compilation_time': float(a['compilation_time']),
            'end_to_end_time': float(a['end_to_end_time'])}


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
        print('Existing campaign manifest verified; no outputs changed.')
        return
    # Validate both originals and reject unexpected outputs before any copying.
    originals = {'arff': paths('arff', 1), 'fourier': (
        ROOT / 'results/adam_split_ex8_seed0_finalcheck.npz', ROOT / 'results/adam_split_ex8_seed0_finalcheck.txt')}
    reused = {}
    for method, seed in [('arff',1), ('fourier',0)]:
        reused[method] = dict(seed=seed, original_artifact=str(originals[method][0]),
                              original_log=str(originals[method][1]), timing_context='preexisting isolated validated run',
                              **validate_pair(method, seed, *originals[method]))
        for s in range(30):
            for p, original in zip(paths(method, s), originals[method]):
                if p.exists() and (s != seed or digest(p) != digest(original)):
                    raise FileExistsError(f'Unexpected existing output, left untouched: {p}')
    for directory in DIRECTORIES.values():
        directory.mkdir(parents=True, exist_ok=True)
    for source, destination in zip(originals['fourier'], paths('fourier', 0)):
        copy_exclusive(source, destination)
    source_paths = sorted(set([str(p.relative_to(ROOT)) for p in (ROOT / 'src').rglob('*.py')]
                             + list(RUNNERS.values()) + ['scripts/run_ex8_campaigns.py',
                               'scripts/run_production_batch.py', 'scripts/diagnose_ex8_validation_selected_crossfit.py']))
    manifest = dict(created_at=now(), purpose='accuracy/reproducibility', seeds=list(range(30)),
                    gpus=[0,1,2,3], assignment='seed modulo 4; ARFF completes before Fourier starts',
                    sessions=SESSIONS, python=PYTHON, reused=reused,
                    timing_context='New jobs may execute concurrently across four GPUs; shared CPU/memory/PCIe contention. Not publication-quality isolated timing.',
                    numerical_runners_frozen=True, dataset_sha256=digest(ROOT/'data/ex8.npz'),
                    source_sha256={p:digest(ROOT/p) for p in source_paths})
    write_json(CONTROL/'manifest.json', manifest, exclusive=True)
    for method in DIRECTORIES:
        write_json(DIRECTORIES[method]/'campaign_state.json',
                   dict(method=method, status='prepared', updated_at=now(), seeds={}), exclusive=True)
    print('Prepared campaign manifest; ARFF seed 1 retained and Fourier seed 0 copied byte-for-byte.')


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
        if method == 'fourier':
            command.append('ex8')
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
        state = read_json(DIRECTORIES['arff']/'campaign_state.json')
        if state['status'] in ('failed','blocked'):
            raise RuntimeError('ARFF campaign failed/blocked; Fourier will not launch')
        if state['status'] == 'complete':
            for seed in range(30):
                validate_pair('arff', seed, *paths('arff', seed))
            return
        time.sleep(15)


def final_report():
    summary = {}
    for method in DIRECTORIES:
        state = read_json(DIRECTORIES[method]/'campaign_state.json')
        times = [v['algorithm_time'] for v in state['seeds'].values() if v['status']=='complete']
        summary[method] = dict(new_parallel_runs=len(times), mean_algorithm_time=float(np.mean(times)),
                               std_algorithm_time=float(np.std(times,ddof=1)), min_algorithm_time=min(times),
                               max_algorithm_time=max(times), reused_seed=read_json(CONTROL/'manifest.json')['reused'][method]['seed'])
    write_json(CONTROL/'parallel_timing_summary.json', summary)
    proposal = ('# Proposed isolated-runtime protocol — NOT EXECUTED\n\n'
                'The completed accuracy campaigns used four concurrent GPUs. Their timings are preserved, '
                'but shared-resource contention prevents direct publication comparisons with isolated benchmarks.\n\n'
                'Propose 30 fresh runs per method, seeds 0–29, sequentially on physical GPU 0 with the entire '
                'machine otherwise idle. Use the frozen runners, data, arff-sde environment and normal per-process '
                'warm-up. Save to separate isolated-runtime directories; do not replace accuracy artifacts. '
                'Predeclare all seeds and report mean, sample SD, median and range of algorithm and warm-up time. '
                'Check all GPUs and CPU occupancy before every run; flag external interference without silently '
                'dropping results. Record GPU clocks/temperature and driver/runtime environment. '
                'Use a fixed, balanced alternating method order across seeds.\n\n'
                'Baseline estimate from accepted isolated runs: approximately 5–6 hours including warm-up and '
                'artifact evaluation. Current parallel timing variability is recorded below for context, not '
                'as an isolated-runtime estimate. This protocol requires approval before execution.\n\n'
                '```json\n' + json.dumps(summary,indent=2) + '\n```\n')
    destination = CONTROL/'isolated_runtime_proposal.md'
    with destination.open('x') as handle:
        handle.write(proposal)


def run(method):
    verify_snapshot()
    with lock_file(CONTROL/f'{method}.supervisor.lock'):
        update(method, status='waiting_for_arff' if method=='fourier' else 'starting')
        try:
            if method=='fourier':
                print('Waiting for all 30 ARFF seeds to validate before Fourier dispatch.', flush=True)
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
            if method=='fourier':
                final_report()
        except Exception:
            update(method,status='failed')
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','status'])
    parser.add_argument('method',nargs='?',choices=['arff','fourier'])
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
