from functools import partial
from typing import Callable, Optional, Sequence, Type
import flax.linen as nn
import jax.numpy as jnp
import jax

beta_1 = 20.0
beta_0 = 0.1

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = jnp.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = jnp.cos((t + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0, 0.999)

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    betas = jnp.linspace(
        beta_start, beta_end, timesteps
    )
    return betas

def vp_beta_schedule(timesteps):
    """Discret VP noise schedule
    """
    t = jnp.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.
    b_min = 0.1
    alpha = jnp.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return betas

def vp_sde_schedule(t):
    """Continous VPSDE schedule. Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.
    """
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    alpha_t = jnp.exp(log_mean_coeff)
    std = jnp.sqrt(1. - jnp.exp(2. * log_mean_coeff))
    return alpha_t, std

class FourierFeatures(nn.Module):
    output_size: int
    learnable: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            w = self.param('kernel', nn.initializers.normal(0.2),
                           (self.output_size // 2, x.shape[-1]), jnp.float32)
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)

class DDPM(nn.Module):
    cond_encoder_cls: Type[nn.Module]
    reverse_encoder_cls: Type[nn.Module]
    time_preprocess_cls: Type[nn.Module]

    @nn.compact
    def __call__(self,
                 s: jnp.ndarray,
                 a: jnp.ndarray,
                 time: jnp.ndarray,
                 training: bool = False):

        time = time.reshape(a.shape[0], -1)  # (B, 1) shape check, make sure that time has the same dimension as s and a
        t_ff = self.time_preprocess_cls()(time)
        cond = self.cond_encoder_cls()(t_ff, training=training)
        reverse_input = jnp.concatenate([a, s, cond], axis=-1)

        return self.reverse_encoder_cls()(reverse_input, training=training)

@partial(jax.jit, static_argnames=('actor_apply_fn', 'act_dim', 'T', 'repeat_last_step', 'clip_sampler', 'training'))
def ddpm_sampler(actor_apply_fn, actor_params, T, rng, act_dim, observations, alphas, alpha_hats, betas, sample_temperature, repeat_last_step, clip_sampler, training = False):

    batch_size = observations.shape[0]
    
    def fn(input_tuple, time):
        current_x, rng = input_tuple
        
        input_time = jnp.expand_dims(jnp.array([time]).repeat(current_x.shape[0]), axis = 1)
        eps_pred = actor_apply_fn({"params": actor_params}, observations, current_x, input_time, training = training)

        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng, key = jax.random.split(rng, 2)
        z = jax.random.normal(key,
                            shape=(observations.shape[0], current_x.shape[1]),)
        z_scaled = sample_temperature * z
        current_x = current_x + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)

        if clip_sampler:
            current_x = jnp.clip(current_x, -1, 1)

        return (current_x, rng), ()

    key, rng = jax.random.split(rng, 2)
    input_tuple, () = jax.lax.scan(fn, (jax.random.normal(key, (batch_size, act_dim)), rng), jnp.arange(T-1, -1, -1), unroll = 5)

    for _ in range(repeat_last_step):
        input_tuple, () = fn(input_tuple, 0)
    
    action_0, rng = input_tuple
    action_0 = jnp.clip(action_0, -1, 1)

    return action_0, rng

@partial(jax.jit, static_argnames=('actor_apply_fn', 'act_dim', 'T', 'repeat_last_step', 'clip_sampler', 'training'))
def dpm_solver_sampler_1st(actor_apply_fn, actor_params, T, rng, act_dim, observations, alphas, alpha_hats, betas, sample_temperature, repeat_last_step, clip_sampler, training = False):
    batch_size = observations.shape[0]
    t_T = 1.
    t_0 = 1e-3
    time_steps = jnp.linspace(t_T, t_0, T + 1)
    orders = [1,] * T  # first order solver
        
    def singlestep_dpm_solver_update(input_tuple, time_index):
        current_x, rng = input_tuple
        vec_s = jnp.expand_dims(jnp.array([time_steps[time_index]]).repeat(current_x.shape[0]), axis = 1)
        vec_t = jnp.expand_dims(jnp.array([time_steps[time_index+1]]).repeat(current_x.shape[0]), axis = 1)

        eps_pred = actor_apply_fn({"params": actor_params}, observations, current_x, vec_s, training = training)
        
        current_x = dpm_solver_first_update(current_x, vec_s, vec_t, eps_pred)
        
        if clip_sampler:
            current_x = jnp.clip(current_x, -1, 1)
        
        rng, key = jax.random.split(rng, 2)
        return (current_x, rng), ()
    
    key, rng = jax.random.split(rng, 2)
    input_tuple, () = jax.lax.scan(singlestep_dpm_solver_update, (jax.random.normal(key, (batch_size, act_dim)) * 0, rng) , jnp.arange(0, T, 1), unroll = 5)
    
    for _ in range(repeat_last_step):
        input_tuple, () = singlestep_dpm_solver_update(input_tuple, T)
        
    action_0, rng = input_tuple
    action_0 = jnp.clip(action_0, -1, 1)
    return action_0, rng

def dpm_solver_first_update(x, s, t, eps_pred):
    # dims = x.dim()
    lambda_s, lambda_t = marginal_lambda(s), marginal_lambda(t)
    h = lambda_t - lambda_s
    log_alpha_s, log_alpha_t = marginal_log_mean_coeff(s), marginal_log_mean_coeff(t)
    sigma_t = marginal_std(t)

    phi_1 = jnp.expm1(h)
    
    # equation 3.7
    x_t = (
        (jnp.exp(log_alpha_t - log_alpha_s)) * x
        - (sigma_t * phi_1) * eps_pred
    )
    return x_t

def get_time_steps(t_T, t_0, N):
    """Uniform time sample

    Args:
        t_T (float): start time stamp
        t_0 (float): end time stamp
        N (int): sample steps

    Returns:
        A jnp.DeviceArray of the time steps, with the shape (N + 1,).
    """
    return jnp.linspace(t_T, t_0, N + 1)

def marginal_lambda(t):
    """
    Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
    """
    log_mean_coeff = marginal_log_mean_coeff(t)
    log_std = 0.5 * jnp.log(1. - jnp.exp(2. * log_mean_coeff))
    return log_mean_coeff - log_std

def marginal_log_mean_coeff(t):
    return -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0

def marginal_std(t):
    """
    Compute sigma_t of a given continuous-time label t in [0, T].
    """
    return jnp.sqrt(1. - jnp.exp(2. * marginal_log_mean_coeff(t))) 

def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a jnp.DeviceArray with shape [N].
        `dim`: a `int`.
    Returns:
        a jnp.DeviceArray with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,)*(dims - 1)]