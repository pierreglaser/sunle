import jax
import jax.numpy as jnp
from jax import random, vmap
from numpyro import distributions as np_distributions
from unle.samplers.distributions import (
    DoublyIntractableLogDensity,
    maybe_wrap,
    maybe_wrap_log_l,
)
from unle.samplers.pytypes import Array


def get_likelihood_dist(sigma_sq):
    mean = 0.0
    return np_distributions.MultivariateNormal(
        jnp.array(mean)[None], covariance_matrix=sigma_sq * jnp.eye(1)  # type: ignore
    )


def log_likelihood(sigma_sq: Array, x: Array):
    # for now, sigma is assumed to be 1d
    mu = jnp.zeros((1,))

    if len(x.shape) == 2:
        n, _ = x.shape[0], x.shape[1]
    else:
        n, _ = 1, x.shape[0]

    return -1 / (2 * jnp.sum(sigma_sq)) * jnp.sum(
        vmap(lambda y: jnp.dot(y - mu, y - mu))(x)
    ) - n / 2 * jnp.log(jnp.prod(2 * jnp.pi * sigma_sq))


def unormalized_log_likelihood(sigma_sq: Array, x: Array):
    # for now, sigma is assumed to be 1d
    mu = jnp.zeros((1,))

    return (
        -1
        / (2 * jnp.sum(sigma_sq))
        * jnp.sum(vmap(lambda y: jnp.dot(y - mu, y - mu))(x))
    )


def get_prior_dist() -> np_distributions.Distribution:
    alpha = 1.0
    beta = 1.0
    return np_distributions.InverseGamma(
        jnp.array(alpha)[None], jnp.array(beta)[None]  # type: ignore
    ).to_event()


def get_posterior_dist(x) -> np_distributions.Distribution:
    mu = jnp.zeros((1,))

    n = x.shape[0]

    alpha = 1.0
    beta = 1.0
    return np_distributions.InverseGamma(
        jnp.array(alpha)[None] + n / 2,
        jnp.array(beta)[None]
        + jnp.sum((x - mu.reshape(-1, 1)) ** 2) / 2,  # type: ignore
    ).to_event()


key = random.PRNGKey(0)
key, key_sigma, key_x = random.split(key, num=3)
sigma_sq = get_prior_dist().sample(key_sigma) + 10.0

x_obs: Array
x_obs = get_likelihood_dist(sigma_sq).sample(
    key_x, sample_shape=(1,)
)  # pyright: ignore[reportGeneralTypeIssues]

posterior = get_posterior_dist(x_obs)


doubly_intractable_log_prob = DoublyIntractableLogDensity(
    log_prior=maybe_wrap(get_prior_dist().log_prob),  # type: ignore
    log_likelihood=maybe_wrap_log_l(unormalized_log_likelihood),
    x_obs=x_obs[0, :],
)


def posterior(sigma_sq):
    return log_likelihood(sigma_sq, x_obs) + get_prior_dist().log_prob(sigma_sq)


def test_initialization():
    from unle.samplers.inference_algorithms.mcmc.base import adam_initialize

    init = adam_initialize(
        3.0, posterior, num_steps=3000, learning_rate=0.01  # type: ignore
    )

    assert jnp.abs(jax.grad(posterior)(init)) < 0.001


def test_initialization_doubly_intractable():
    from unle.samplers.inference_algorithms.mcmc.base import (
        adam_initialize_doubly_intractable,
    )

    init = adam_initialize_doubly_intractable(
        jnp.array(3.0),
        doubly_intractable_log_prob,
        num_steps=3000,
        key=random.PRNGKey(1),  # type: ignore
        learning_rate=0.001,  # type: ignore
        num_likelihood_sampler_steps=1000,
    )
    assert jnp.abs(jax.grad(posterior)(init)) < 0.005
