#!/usr/bin/env python3
"""Versioned, matched-noise Ex8 re-integration; never edits the original dataset."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cuda')
from pathlib import Path
import sys, json, hashlib, platform, subprocess, time
import numpy as np
import jax
import jax.numpy as jnp
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments.definitions import ex8_drift, ex8_diffusion_factor
from src.experiments.dataset import load_dataset
OUT = ROOT / 'results/controlled_study_2026/float64_v2_validation'
SOURCE = ROOT / 'data/ex8.npz'
TARGET = ROOT / 'data/ex8_float64_v2.npz'
SOURCE_SHA = 'a95f5483184d0e8dad39231fc30b18779d600be59c38d5464e22dca03db707c7'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rms(v): return float(np.sqrt(np.mean(np.asarray(v, dtype=np.float64)**2)))
def write(p, v):
    with p.open('x') as f: json.dump(v, f, indent=2)
def integrate(x, noise, dt, zero=False):
    def step(state, z):
        increment = jnp.zeros_like(state) if zero else jnp.einsum('nij,nj->ni', ex8_diffusion_factor(state), jnp.sqrt(dt)*z)
        return state + dt*ex8_drift(state) + increment, None
    endpoint, _ = jax.lax.scan(step, x, noise)
    return endpoint-x

def main():
    start = time.time()
    assert sha(SOURCE) == SOURCE_SHA and not TARGET.exists()
    OUT.mkdir(parents=True, exist_ok=False)
    with np.load(SOURCE, allow_pickle=False) as z: original = {k: z[k] for k in z.files}
    x = original['x_data']; assert x.shape == (100000, 2)
    jax.config.update('jax_enable_x64', False)
    _, key_noise = jax.random.split(jax.random.PRNGKey(0))
    print('Generate original float32 full-shape normal draws', flush=True)
    noise = jax.random.normal(key_noise, (1000, 100000, 2), dtype=jnp.float32)
    noise.block_until_ready()
    # Replay the full dataset, rather than infer full-data fidelity from a subset.
    replay32 = np.asarray(integrate(jnp.asarray(x), noise, jnp.asarray(1e-7, dtype=jnp.float32)))
    np.testing.assert_array_equal(replay32, original['r_data'])
    noise_sha = hashlib.sha256(np.asarray(noise).tobytes()).hexdigest()
    print('Full float32 replay is bit-identical; integrate float64', flush=True)
    jax.config.update('jax_enable_x64', True)
    increments = np.asarray(integrate(jnp.asarray(x, dtype=jnp.float64), noise.astype(jnp.float64), jnp.asarray(1e-7, dtype=jnp.float64)))
    assert increments.dtype == np.float64 and np.isfinite(increments).all()
    n = 4096
    previous = ROOT/'results/controlled_study_2026/preflight/matched_noise_precision_gpu.npz'
    with np.load(previous) as z: np.testing.assert_array_equal(increments[:n], z['cpu64'])
    xs = jnp.asarray(x[:n], dtype=jnp.float64)
    z0 = noise[:, :n, :].astype(jnp.float64)
    print('Validate zero-noise and coupled fine-step refinement', flush=True)
    zero = np.asarray(integrate(xs, jnp.zeros_like(z0), jnp.asarray(1e-7, dtype=jnp.float64), zero=True))
    # expm1/log1p avoids cancellation in the reference increment.
    exact = np.asarray(xs)*np.expm1(1000*np.log1p(-1e-7))
    zero_error = rms((zero-exact)/1e-4)
    assert zero_error < 1e-8
    # Conditional Brownian bridges preserve each original coarse increment.
    levels = [increments[:n]]; closure = []
    for level in (1, 2):
        bridge = jax.random.normal(jax.random.fold_in(key_noise, 64000+level), z0.shape, dtype=jnp.float64)
        fine = jnp.stack(((z0+bridge)/jnp.sqrt(2.), (z0-bridge)/jnp.sqrt(2.)), axis=1).reshape((-1,n,2))
        closure.append(float(np.max(np.abs(np.asarray((fine[0::2]+fine[1::2])/jnp.sqrt(2.)-z0)))))
        z0 = fine
        levels.append(np.asarray(integrate(xs, z0, jnp.asarray(1e-7/(2**level), dtype=jnp.float64))))
    refinement = [rms((levels[i]-levels[i+1])/1e-4) for i in (0,1)]
    assert refinement[1] < refinement[0], refinement
    cast_error = rms((increments.astype(np.float32).astype(np.float64)-increments)/1e-4)
    assert cast_error < 1e-5
    # Keep original state/lag/split representations exactly. Only r_data changes.
    corrected = dict(original, r_data=increments)
    staged = OUT/'validated_dataset.npz'
    with staged.open('xb') as f: np.savez_compressed(f, **corrected)
    d = load_dataset(staged)
    assert [len(d.train_idx), len(d.validation_idx), len(d.test_idx)] == [80000,10000,10000]
    with np.load(staged) as z:
        for k, v in original.items():
            if k != 'r_data':
                assert z[k].dtype == v.dtype and z[k].tobytes() == v.tobytes()
        assert z['r_data'].dtype == np.float64
        np.testing.assert_array_equal(z['r_data'], increments)
    changed = np.any(increments != original['r_data'].astype(np.float64), axis=1)
    with (OUT/'changed_rows.npz').open('xb') as f:
        np.savez_compressed(f, r_data_row_ids=np.flatnonzero(changed), fields=np.asarray(['r_data']))
    with (OUT/'refinement_evidence.npz').open('xb') as f:
        np.savez_compressed(f, original_row_ids=np.arange(n), dt_1e7=levels[0], dt_5e8=levels[1], dt_25e9=levels[2], deterministic_increment=zero, deterministic_reference=exact)
    result = dict(status='passed', original_dataset=str(SOURCE), original_sha256=SOURCE_SHA,
        corrected_dataset=str(TARGET), corrected_sha256=sha(staged), changed_fields=['r_data'], changed_rows=int(changed.sum()),
        unchanged_fields=[k for k in original if k != 'r_data'], sample_counts=dict(total=100000,train=80000,validation=10000,test=10000),
        h=1e-4, fine_steps=1000, delta=1e-7, seed=0, sde='unchanged ex8_drift and ex8_diffusion_factor',
        construction='same original full-shape float32 normal draws, cast to float64; float64 state updates and endpoint differences; r_data stored float64',
        noise_shape=[1000,100000,2], noise_sha256=noise_sha, full_float32_replay_bit_identical=True,
        prior_4096_float64_GPU_replay_bit_identical=True, deterministic_identity='r=((1-delta)^1000-1)*x; evaluated with expm1(1000*log1p(-delta))',
        deterministic_time_normalized_rmse=zero_error, refinement_time_normalized_rms=refinement,
        refinement_definition='4096 fixed initial points; nested conditional Brownian bridge refinements delta, delta/2, delta/4; RMS successive increment differences divided by h',
        refinement_bridge_keys=[64001,64002], refinement_closure_max=closure,
        float64_to_existing_float32_training_increment_conversion_time_normalized_rmse=cast_error,
        original_to_corrected_time_normalized_rmse=rms((original['r_data'].astype(np.float64)-increments)/1e-4),
        stored_dtypes={k:str(v.dtype) for k,v in corrected.items()}, training_precision='unchanged accepted runners; only integration and archived increments use float64',
        source_sha256={p:sha(ROOT/p) for p in ['src/experiments/em_data.py','src/experiments/definitions.py','scripts/create_ex8_float64_v2.py']},
        environment=dict(python=platform.python_version(),jax=jax.__version__,numpy=np.__version__,devices=[str(v) for v in jax.devices()]),
        end_to_end_seconds=time.time()-start)
    assert sha(SOURCE) == SOURCE_SHA
    write(OUT/'validation.json', result)
    # Exclusive publication, only after all checks pass. Preserve staging evidence.
    os.link(staged, TARGET)
    write(TARGET.with_suffix('.provenance.json'), result)
    print(json.dumps(result, indent=2), flush=True)
if __name__ == '__main__': main()
