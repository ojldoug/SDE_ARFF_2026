from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dill
import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from lib.lib_ARFF_folds2 import (
    ARFFHyperparameters,
    ARFFTrain,
)


ex_name = "ex8"

with open(
    HERE / "true_functions" / f"{ex_name}.pkl",
    "rb",
) as f:
    true_functions = dill.load(f)

true_drift = true_functions["drift"]
true_diffusion = true_functions["diffusion"]

training_data = np.load(
    HERE / "training_data" / f"{ex_name}.npz"
)

x = training_data["x_data"]
r = training_data["r_data"]
h = training_data["step_sizes"]
diff_type = str(training_data["diff_type"])

print("raw shapes:", x.shape, r.shape, h.shape)

if x.ndim == 3:
    sample_rate = 100

    x = (
        x[:, :, ::sample_rate]
        .reshape(-1, x.shape[1])
    )

    r = (
        r.reshape(
            r.shape[0],
            r.shape[1],
            r.shape[2] // sample_rate,
            sample_rate,
        )
        .sum(axis=3)
        .reshape(-1, r.shape[1])
    )

    h = (
        h.reshape(
            h.shape[0],
            h.shape[1] // sample_rate,
            sample_rate,
        )
        .sum(axis=2)
    )

print("training shapes:", x.shape, r.shape, h.shape)

val_split = 0.1
ARFF_val_split = 0.1

drift_hyperparam = ARFFHyperparameters(
    K=2**7,
    M_min=300,
    M_max=300,
    lambda_reg=1e-3,
    gamma=1,
    delta=0.2,
    name="drift",
)

diff_hyperparam = ARFFHyperparameters(
    K=drift_hyperparam.K,
    M_min=drift_hyperparam.M_min,
    M_max=drift_hyperparam.M_max,
    lambda_reg=drift_hyperparam.lambda_reg,
    gamma=drift_hyperparam.gamma,
    delta=drift_hyperparam.delta,
    name="diffusion",
)

key = random.PRNGKey(0)

ARFF = ARFFTrain(
    resampling=False,
    metropolis_test=True,
)

(
    drift_param,
    diff_param,
    training_time,
    loss,
    val_loss,
    drift_err,
    diff_err,
    z,
    Sigma,
) = ARFF.train_model(
    key,
    drift_hyperparam,
    diff_hyperparam,
    x,
    r,
    h,
    val_split=val_split,
    ARFF_val_split=ARFF_val_split,
    diff_type=diff_type,
    plot=False,
    true_drift=true_drift,
    true_diffusion=true_diffusion,
    n_folds=1,
    enforce_spd=True,
)

jax.block_until_ready(
    (
        drift_param,
        diff_param,
    )
)

print()
print("================================================")
print("LEGACY EX8 SEED 0")
print("================================================")
print(f"time       : {training_time:.8e}")
print(f"train NLL  : {loss:.8e}")
print(f"val NLL    : {val_loss:.8e}")
print(f"drift RMSE : {drift_err:.8e}")
print(f"diff RMSE  : {diff_err:.8e}")
print("================================================")
