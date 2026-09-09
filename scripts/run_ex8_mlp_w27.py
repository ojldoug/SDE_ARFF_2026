#!/usr/bin/env python3
"""Width-only adapter around the unchanged Experiment 8 Joint MLP runner."""
from dataclasses import asdict, replace
import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from src.experiments.config import get_config

RUNNERS = {'mlp': 'run_adam_mlp_experiment'}


def effective_config(name):
    if name != 'ex8':
        raise ValueError('This adapter is exclusively for Experiment 8')
    return replace(get_config(name), fourier_frequencies=128)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.set_defaults(method='mlp')
    p.add_argument('--hidden-width', type=int, choices=[27], default=27)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--artifact-path', type=Path)
    p.add_argument('--inspect', action='store_true', help='Print the sole configuration change; no JAX or training')
    args = p.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        p.error('seed must fit an unsigned 32-bit PRNG seed')
    base, effective = asdict(get_config('ex8')), asdict(effective_config('ex8'))
    differences = {k: [base[k], effective[k]] for k in base if base[k] != effective[k]}
    if differences != {'fourier_frequencies': [512,128]}:
        raise RuntimeError(f'Unexpected baseline/configuration change: {differences}')
    if args.inspect:
        print(json.dumps(dict(method=args.method, runner=RUNNERS[args.method], changes=differences,
                              effective_config=effective, hidden_width=27, total_parameters=1814), indent=2))
        return
    if args.artifact_path is None or args.artifact_path.suffix != '.npz':
        p.error('a new .npz --artifact-path is required')
    path = args.artifact_path.expanduser().resolve()
    if path.exists():
        p.error(f'refusing to overwrite existing artifact: {path}')
    module = importlib.import_module(RUNNERS[args.method])
    original_get_config, original_argv = module.get_config, sys.argv
    try:
        # Only this runner module's binding changes, before its main reads it.
        # Global CONFIGS, optimizer and numerical functions remain untouched.
        # K here is only a parameter-budget input to matched_two_layer_width.
        # The existing matcher selects (27,27), 1814 parameters against 1792.
        module.get_config = effective_config
        sys.argv = [module.__file__, 'ex8', '--seed', str(args.seed), '--artifact-path', str(path)]
        module.main()
    finally:
        module.get_config = original_get_config
        sys.argv = original_argv


if __name__ == '__main__':
    main()
