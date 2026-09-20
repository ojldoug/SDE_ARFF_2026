#!/usr/bin/env python3
"""Versioned data adapter; retain the accepted native training precision and code."""
from pathlib import Path
import os
import run_controlled_ex8 as base
base.STUDY = base.ROOT/'results/controlled_study_2026/float64_v2'
ROOT, STUDY = base.ROOT, base.STUDY
RUNNERS, digest, configuration = base.RUNNERS, base.digest, base.configuration

def verify_registration():
    m = base.verify_registration()
    import json
    v = json.loads((ROOT/'results/controlled_study_2026/float64_v2_validation/validation.json').read_text())
    assert v['status'] == 'passed'
    assert digest(ROOT/m['dataset']['path']) == m['dataset']['sha256'] == v['corrected_sha256']
    assert digest(ROOT/v['original_dataset']) == v['original_sha256']
    return m

def run_job(job, path, allow_cpu=False):
    verify_registration()
    import jax
    assert not jax.config.jax_enable_x64, 'Preserve accepted float32 estimator arithmetic'
    base.run_job(job, path, allow_cpu=allow_cpu)
    verify_registration()

def main():
    import argparse, json
    from dataclasses import asdict
    p=argparse.ArgumentParser()
    p.add_argument('--job-json',type=Path,required=True)
    p.add_argument('--artifact-path',type=Path,required=True)
    p.add_argument('--inspect',action='store_true')
    a=p.parse_args();job=json.loads(a.job_json.read_text())
    if a.inspect:
        verify_registration()
        print(json.dumps(dict(job=job,effective_config=asdict(configuration(job))),indent=2))
    else:run_job(job,a.artifact_path.resolve())
if __name__=='__main__':main()
