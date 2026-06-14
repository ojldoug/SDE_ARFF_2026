import jax
import jax.numpy as jnp
import optax
from jax import jit, vmap
from jax import random, grad, value_and_grad
from jax import nn
from functools import partial
from matplotlib import pyplot as plt

import numpy as np
import time
import math

jax.config.update("jax_enable_x64", False)
DTYPE = jnp.float32


def split_data(key, val_split, *inputs):
    key, subkey = random.split(key)

    # obtain split indices
    N = inputs[0].shape[0]
    val_sample_size = int(N * val_split)
    permuted_idx = random.permutation(subkey, N)
    val_idx = permuted_idx[:val_sample_size]
    mask = jnp.ones(N, dtype=bool).at[val_idx].set(False)

    # split the data
    inputs_train = tuple(data[mask] for data in inputs)
    inputs_valid = tuple(data[~mask] for data in inputs)

    return inputs_train, inputs_valid, key


def shuffle_data(key, *inputs):
    """Randomly permute aligned arrays."""
    key, subkey = random.split(key)
    N = inputs[0].shape[0]
    perm = random.permutation(subkey, N)
    return tuple(data[perm] for data in inputs), key


def make_folds(N, n_folds):
    """Return approximately equal fold index arrays."""
    indices = np.arange(N)
    return np.array_split(indices, n_folds)


class Functions:
    @jit
    def S(omega, x):
        x_proj = x @ omega
        return jnp.concatenate([jnp.cos(x_proj), jnp.sin(x_proj)], axis=-1)
        # return jnp.exp(1j * x_proj)
    
    @jit
    def beta(params, x):
        return (Functions.S(params["omega"], x) @ params["amp"])
    
    def drift(params, x):
        drift_ = Functions.beta(params, x)
        return drift_

    def diffusion_cov(params, x, diff_type, spd=True, eps=1e-8):
        x = jnp.atleast_2d(x)
        cov_vectors = Functions.beta(params, x)

        violation = False

        if diff_type == "diagonal":
            if spd:
                violation = jnp.any(cov_vectors <= 0.0)
                cov_vectors = jnp.maximum(cov_vectors, eps)
            else:
                cov_vectors = jnp.abs(cov_vectors)

            cov = vmap(jnp.diag)(cov_vectors)
            return cov, violation
        
        T = params["amp"].shape[1]
        D = int((math.sqrt(1 + 8 * T) - 1) / 2)
        
        lt_i, lt_j = np.tril_indices(D)
        LT_idx = (jnp.array(lt_i), jnp.array(lt_j))
        
        def make_symmetric_matrix(vec):
            mat = jnp.zeros((D, D))
            mat = mat.at[LT_idx].set(vec)
            sym = mat + mat.T - jnp.diag(jnp.diag(mat))
            return sym
    
        cov = vmap(make_symmetric_matrix)(cov_vectors)

        if spd:
            vals, vecs = jnp.linalg.eigh(cov)
            print(vals)
            violation = vals[:, 0] <= 0.0
            vals = jnp.maximum(vals, eps)
            cov = (vecs * vals[:, None, :]) @ jnp.swapaxes(vecs, -1, -2)

        return cov, violation

    def diffusion(params, x, diff_type, spd=True, eps=1e-8):        
        cov, _ = Functions.diffusion_cov(params, x, diff_type, spd=spd, eps=eps)

        if diff_type == "diagonal":
            return jnp.sqrt(cov)
        elif diff_type == "triangular":
            return jax.vmap(jnp.linalg.cholesky)(cov)

        def matrix_sqrtm(mat):
            vals, vecs = jnp.linalg.eigh(mat)
            sqrt_vals = jnp.sqrt(jnp.clip(vals, a_min=0.0))
            return (vecs * sqrt_vals) @ vecs.T

        return jax.vmap(matrix_sqrtm, in_axes=0, out_axes=0)(cov)


def nll_loss(drift_param, diff_param, x, r, h, diff_type, spd=True, eps=1e-8):
    D = r.shape[1]
    f = Functions.drift(drift_param, x)
    delta = r - h * f

    cov, violation = Functions.diffusion_cov(diff_param, x, diff_type, spd=spd, eps=eps)
    var = cov * h[:, :, None] 

    if diff_type == "diagonal":
        var_vec = jnp.diagonal(var, axis1=1, axis2=2)
        quad = jnp.sum((delta ** 2) / var_vec, axis=1)
        logdet = jnp.sum(jnp.log(var_vec), axis=1)
    else:
        def get_quad_and_logdet(S, d):
            sol = jnp.linalg.solve(S, d)
            quad = jnp.dot(d, sol)
            _, ld = jnp.linalg.slogdet(S)
            return quad, ld

        quad, logdet = jax.vmap(get_quad_and_logdet)(var, delta)
        
    losses = 0.5 * quad + 0.5 * logdet + 0.5 * D * jnp.log(2.0 * jnp.pi)
    return jnp.mean(losses), jnp.mean(violation) 


def function_errors(drift_param, diff_param, x, true_drift, true_diffusion, diff_type):
    """
    Computes empirical L2 errors on x:
      ||f - f_hat||_L2(rho) and ||Sigma - Sigma_hat||_L2(rho)

    Assumptions:
      - learned drift: Functions.beta(drift_param, x)  -> (N, d)
      - learned diffusion: Functions.beta(diff_param, x)
            * diagonal diff_type -> (N, d)  (diagonal of Sigma)
            * full diff_type     -> (N, d(d+1)/2)  (lower-tri of Sigma)
      - true_diffusion(x) may return:
            * (N, d)        : diagonal sigma
            * (N, d, d)     : sigma matrix
            * (N, d, d)     : Sigma matrix (if it already returns covariance)
    """

    # ----- drift -----
    f_hat = Functions.beta(drift_param, x)      # (N,d)
    f_true = true_drift(x)                     # (N,d)
    drift_err = jnp.sqrt(jnp.mean(jnp.sum((f_true - f_hat) ** 2, axis=1)))

    # ----- diffusion -----
    Sigma_hat = Functions.beta(diff_param, x)  # learned representation

    td = true_diffusion(x)
    # Convert true diffusion to Sigma (covariance) in FULL matrix form first
    if td.ndim == 2:
        # td is (N,d): interpret as diagonal sigma -> Sigma diag = sigma^2
        Sigma_full = jnp.einsum('nd,nd->nd', td, td)  # (N,d) diag only
    elif td.ndim == 3:
        # td is (N,d,d): could be sigma or Sigma. We need to decide.
        # Heuristic: if it's (approximately) PSD but not triangular? too hard.
        # We'll assume it's sigma and form Sigma = sigma sigma^T, which is typical.
        Sigma_full = jnp.einsum('nij,nkj->nik', td, td)  # (N,d,d)
    else:
        raise ValueError(f"true_diffusion(x) returned unexpected shape: {td.shape}")

    # Now match the learned representation
    if diff_type == "diagonal":
        if Sigma_full.ndim == 3:
            Sigma_true = jnp.diagonal(Sigma_full, axis1=1, axis2=2)  # (N,d)
        else:
            # Sigma_full already (N,d) diag
            Sigma_true = Sigma_full
    else:
        # full case uses lower-tri vectorization like your training target
        if Sigma_full.ndim != 3:
            # if we only had diag, embed into matrix
            d = Sigma_full.shape[1]
            Sigma_full = jnp.zeros((Sigma_full.shape[0], d, d)).at[:, jnp.arange(d), jnp.arange(d)].set(Sigma_full)
        LT_i, LT_j = jnp.tril_indices(Sigma_full.shape[1])
        Sigma_true = Sigma_full[:, LT_i, LT_j]  # (N, d(d+1)/2)

    diff_err = jnp.sqrt(jnp.mean(jnp.sum((Sigma_true - Sigma_hat) ** 2, axis=1)))

    return drift_err, diff_err




class ARFFHyperparameters:
    def __init__(self, K=2**6, M_min=0, M_max=100, lambda_reg=2e-3, gamma=1, delta=0.1, name=None):
        self.K = K
        self.M_min = M_min
        self.M_max = M_max
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.delta = delta
        self.name = name


class ARFFTrain:
    def __init__(self, resampling=True, metropolis_test=True):
        self.resampling = resampling
        self.metropolis_test = metropolis_test
       
    @staticmethod
    def get_Sigma(drift_param, x, r, h, diff_type):
        f = r - h * Functions.drift(drift_param, x)
        if diff_type == "diagonal":
            Sigma = f ** 2 / h
        else:
            Sigma = jnp.matmul(f[:, :, None], f[:, :, None].transpose(0, 2, 1)) / h[:, :, None]
            LT_idx_i, LT_idx_j = jnp.tril_indices(f.shape[1])
            Sigma = Sigma[:, LT_idx_i, LT_idx_j]
        return Sigma

    def get_cross_fitted_Sigma(
        self,
        key,
        drift_hyperparam,
        x,
        r,
        h,
        z,
        n_folds,
        ARFF_val_split,
        diff_type,
    ):
        """
        Cross-fitted residual covariance targets.

        For each fold I_l:
          train drift on all data except I_l,
          compute residual covariance targets on I_l.
        """
        N = x.shape[0]

        if n_folds < 2:
            Sigma_start = time.time()
            drift_param, drift_ve, drift_moving_avg, drift_time, key = ARFFTrain.ARFF_loop(
                self, key, drift_hyperparam, x, z, ARFF_val_split
            )
            Sigma = ARFFTrain.get_Sigma(drift_param, x, r, h, diff_type)
            Sigma_time = time.time() - Sigma_start
            return drift_param, drift_ve, drift_moving_avg, drift_time, Sigma, Sigma_time, key

        if n_folds > N:
            raise ValueError(f"n_folds={n_folds} cannot exceed number of training samples N={N}.")

        folds = make_folds(N, n_folds)

        Sigma_parts = [None] * n_folds
        x_parts = [None] * n_folds
        crossfit_time = 0.0

        for ell, val_idx_np in enumerate(folds):
            print(f"\nCross-fit drift fold {ell + 1}/{n_folds}")

            val_idx = jnp.array(val_idx_np)
            train_mask_np = np.ones(N, dtype=bool)
            train_mask_np[val_idx_np] = False
            train_idx = jnp.array(np.where(train_mask_np)[0])

            x_train_fold = x[train_idx]
            z_train_fold = z[train_idx]

            x_holdout = x[val_idx]
            r_holdout = r[val_idx]
            h_holdout = h[val_idx]

            t0 = time.time()
            drift_param_fold, _, _, fold_time, key = ARFFTrain.ARFF_loop(
                self,
                key,
                drift_hyperparam,
                x_train_fold,
                z_train_fold,
                ARFF_val_split,
            )
            crossfit_time += fold_time + (time.time() - t0 - fold_time)

            Sigma_holdout = ARFFTrain.get_Sigma(
                drift_param_fold,
                x_holdout,
                r_holdout,
                h_holdout,
                diff_type,
            )

            Sigma_parts[ell] = Sigma_holdout
            x_parts[ell] = x_holdout

        # These are shuffled fold-concatenated targets. That is fine because x and Sigma stay aligned.
        x_cf = jnp.concatenate(x_parts, axis=0)
        Sigma_cf = jnp.concatenate(Sigma_parts, axis=0)

        print("\nTraining final drift model on full training data")
        drift_param, drift_ve, drift_moving_avg, drift_time, key = ARFFTrain.ARFF_loop(
            self,
            key,
            drift_hyperparam,
            x,
            z,
            ARFF_val_split,
        )

        total_drift_time = crossfit_time + drift_time
        Sigma_time = crossfit_time

        return drift_param, drift_ve, drift_moving_avg, total_drift_time, x_cf, Sigma_cf, Sigma_time, key
        
    @staticmethod
    @jit
    def get_amp(x, y, lambda_reg, omega):
        S_ = Functions.S(omega, x) 
        A = jnp.einsum('nk,nl->kl', jnp.conj(S_), S_) + x.shape[0] * lambda_reg * jnp.eye(S_.shape[1])
        b = jnp.matmul(jnp.conj(jnp.transpose(S_)), y)
        
        amp = jnp.linalg.solve(A, b)
        #amp, _ = jax.scipy.sparse.linalg.cg(A, b, tol=1e-6, maxiter=20000)
        return amp
    

    @staticmethod
    @partial(jit, static_argnames=['RESAMPLING', 'METROPOLIS_TEST'])
    def ARFF_one_step(key, omega, amp, x, y, delta, lambda_reg, gamma, RESAMPLING=True, METROPOLIS_TEST=True):

        amp_norm = jnp.linalg.norm(jnp.reshape(amp, (-1, omega.shape[1])), axis=0)

        if RESAMPLING:
            amp_pmf = amp_norm / jnp.sum(amp_norm)
            key, subkey = random.split(key)
            omega = omega[:, random.choice(subkey, amp_norm.shape[0], shape=(omega.shape[1],), p=amp_pmf)]

        if METROPOLIS_TEST:
            key, subkey = random.split(key)
            dw = random.normal(subkey, omega.shape)
            omega_prime = omega + delta * dw
            
            amp_prime_norm = jnp.linalg.norm(jnp.reshape(ARFFTrain.get_amp(x, y, lambda_reg, omega_prime), (-1, omega.shape[1])), axis=0)

            key, subkey = random.split(key)
            omega = jnp.where((amp_prime_norm / amp_norm) ** gamma >= random.uniform(subkey, omega.shape[1]), omega_prime, omega)
               
        else:
            key, subkey = random.split(key)
            dw = random.normal(subkey, omega.shape)
            omega = omega + delta * dw

        amp = ARFFTrain.get_amp(x, y, lambda_reg, omega)

        return omega, amp, key

    def ARFF_loop(self, key, hyperparam, x, y, val_split):
        start_time = time.time()
        (x, y), (x_valid, y_valid), key = split_data(key, val_split, x, y)

        omega = jnp.zeros((x.shape[1], hyperparam.K))
        amp = ARFFTrain.get_amp(x, y, hyperparam.lambda_reg, omega)

        val_errors = jnp.zeros(hyperparam.M_max)
        val_error_min = jnp.inf
        moving_sum = 0
        moving_avg = jnp.zeros(hyperparam.M_max)
        min_moving_avg = jnp.inf
        moving_avg_len = 5
        min_index = 0
        break_iterations = 5
        
        for i in range(hyperparam.M_max):
            omega, amp, key = ARFFTrain.ARFF_one_step(key, omega, amp, x, y, 
                                                      hyperparam.delta, hyperparam.lambda_reg, hyperparam.gamma, 
                                                      RESAMPLING=self.resampling,
                                                      METROPOLIS_TEST=self.metropolis_test)


            val_error = jnp.mean(jnp.abs(Functions.beta({"omega": omega, "amp": amp}, x_valid) - y_valid) ** 2)
            val_errors = val_errors.at[i].set(val_error)
            
            # Update moving sum: add current error
            moving_sum += val_error
            
            # Subtract the error that's no longer in the window if window is full
            if i >= moving_avg_len:
                moving_sum -= val_errors[i - moving_avg_len]

            # Compute moving average
            window_size = i + 1 if i < moving_avg_len else moving_avg_len
            moving_avg = moving_avg.at[i].set(moving_sum / window_size)
  
            if moving_avg[i] < min_moving_avg:
                min_moving_avg = moving_avg[i]
                min_index = i
 
            if min_index + break_iterations < i and i > hyperparam.M_min:
                break

            if jnp.isnan(val_error):
                val_error = 1e100
                
            if val_error < val_error_min:
                end_time = time.time()
                val_errors_min = val_error
                param = {"omega": omega, "amp": amp}
 
            if i % 1 == 0 or i == hyperparam.M_max - 1:
                print(f"\r{hyperparam.name} epoch: {i}", end='')

        print()
        
        return param, val_errors[:i], moving_avg[:i], end_time-start_time, key

    def train_model(self, key, drift_hyperparam, diff_hyperparam, x, r, h, val_split=0.1, ARFF_val_split=0.1, diff_type="diagonal", plot=True, true_drift=None, true_diffusion=None, n_folds=1, enforce_spd=False):
            
        # First split off validation data.
        (x, r, h), (x_valid, r_valid, h_valid), key = split_data(key, val_split, x, r, h)

        # Then randomize the remaining training data before fold construction.
        (x, r, h), key = shuffle_data(key, x, r, h)

        # Scaled increments for drift training.
        z = r / h

        if n_folds is None or n_folds < 2:
            # Original non-cross-fitted behavior.
            drift_param, drift_ve, drift_moving_avg, drift_time, key = ARFFTrain.ARFF_loop(
                self, key, drift_hyperparam, x, z, ARFF_val_split
            )
            plot and plot_loss(drift_ve, drift_moving_avg)

            Sigma_start = time.time()
            Sigma = ARFFTrain.get_Sigma(drift_param, x, r, h, diff_type)
            Sigma_time = time.time() - Sigma_start
            x_for_diff = x

        else:
            # Cross-fitted diffusion targets.
            (
                drift_param,
                drift_ve,
                drift_moving_avg,
                drift_time,
                x_for_diff,
                Sigma,
                Sigma_time,
                key,
            ) = self.get_cross_fitted_Sigma(
                key=key,
                drift_hyperparam=drift_hyperparam,
                x=x,
                r=r,
                h=h,
                z=z,
                n_folds=n_folds,
                ARFF_val_split=ARFF_val_split,
                diff_type=diff_type,
            )
            plot and plot_loss(drift_ve, drift_moving_avg)

        # Train diffusion model on either ordinary or cross-fitted covariance targets.
        diff_param, diff_ve, diff_moving_avg, diff_time, key = ARFFTrain.ARFF_loop(
            self, key, diff_hyperparam, x_for_diff, Sigma, ARFF_val_split
        )
        plot and plot_loss(diff_ve, diff_moving_avg)

        # Evaluate final drift and diffusion on original training/validation split.
        
        spd_eps = 1e-20
        loss, train_viol = nll_loss(drift_param, diff_param, x, r, h, diff_type, spd=enforce_spd, eps=spd_eps)
        val_loss, val_viol = nll_loss(drift_param, diff_param, x_valid, r_valid, h_valid, diff_type, spd=enforce_spd, eps=spd_eps)

        drift_err, diff_err = function_errors(
            drift_param,
            diff_param,
            x_valid,
            true_drift,
            true_diffusion,
            diff_type,
        )

        training_time = drift_time + Sigma_time + diff_time

        print(f"loss = {loss:.4f}, val_loss = {val_loss:.4f}, train_viol_rate = {train_viol:.4f}, val_viol_rate = {val_viol:.4f}, time = {training_time:.4f}s")
        print(f"drift_err = {drift_err:.4e}, diff_err = {diff_err:.4e}")

        return drift_param, diff_param, training_time, loss, val_loss, drift_err, diff_err, z, Sigma


def plot_loss(ve, moving_avg):
    plt.semilogy(ve, label="Validation Error")
    plt.semilogy(moving_avg, label="Moving Average")

    plt.title('ARFF Loss', fontsize=12)
    plt.xlabel(r'$M$', fontsize=12)
    plt.legend()
    plt.show()
    