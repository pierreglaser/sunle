from typing import Callable, Literal, cast

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as np_distributions
import pytest
from jax import Array, random
from unle.samplers.kernels.adaptive_mala import (
    AdaptiveMALAConfig,
    AdaptiveMALAKernelFactory,
)
from unle.unle import UNLE


def unle_one_round(
    prior: np_distributions.Distribution,
    simulator: Callable,
    x_obs: Array,
    use_warm_start: bool,
    num_samples: int,
    ebm_model_type: Literal["likelihood", "joint_tilted"],
    estimate_log_normalizer: bool,
    key: random.KeyArray,
):
    unle = UNLE.create()

    key, subkey = random.split(key)
    parameters = prior.sample(subkey, (num_samples,))
    observations = simulator(parameters)

    unle = unle.append_simulations(parameters, observations, prior)

    key, subkey = random.split(key)
    _, unle = unle.train_likelihood(
        subkey,
        ebm_model_type,
        max_iter=10,
        num_frozen_steps=10,
        num_smc_steps=5,
        num_mala_steps=3,
        ess_threshold=0.8,
        num_particles=100,
        use_warm_start=use_warm_start,
        batch_size=10,
        learning_rate=0.01,
        weight_decay=0.1,
    )

    if ebm_model_type == "likelihood" and estimate_log_normalizer:
        key, subkey = random.split(key)
        unle = unle.train_lznet(subkey)

    unle, current_posterior = unle.build_posterior(
        prior, x_obs, sampler="mcmc", num_warmup_steps=5
    )
    return current_posterior


@pytest.mark.parametrize(
    "use_warm_start",
    [False, True],
)
def test_standalone_prior_simulator(use_warm_start):
    np.random.seed(42)

    # Using numpy because the current expectation is that simulator
    # mutate global random state, and thus do not explicitly take a
    # random key as input -- numpy does that, jax does not.
    def simulator(params) -> Array:
        return params + np.random.normal(0, 1, size=params.shape)

    prior_dist = np_distributions.Normal(
        jnp.zeros((2,)), jnp.ones((2,))  # type: ignore
    ).to_event(1)

    key, subkey = random.split(random.PRNGKey(0))
    theta = prior_dist.sample(subkey)
    x_obs = simulator(theta)

    AdaptiveMALAKernelFactory(
        AdaptiveMALAConfig(0.1, update_cov=False, use_dense_cov=False)
    )

    _, subkey = random.split(key)
    # TODO(pierreglaser): use KeyArray instead of PRNGKeyArray
    posterior = unle_one_round(
        prior_dist,
        simulator,
        x_obs,
        use_warm_start,
        100,
        "joint_tilted",
        False,
        key=cast(random.KeyArray, subkey),
    )
    key, subkey = random.split(key)

    # TODO: these tests should have more elaborate testing
    # harnesses to check for warm starting for MCMC
    # and cold starting for SMC, as well as test for the
    # actual quality of posterior + warm starting
    # test that different values of num_samples work
    # as expected
    samples_one = posterior.sample(subkey, (10,), return_updated_dist=False)
    assert isinstance(samples_one, jnp.ndarray)
    assert samples_one.shape == (10, 2)

    samples_two = posterior.sample(subkey, (20,), return_updated_dist=False)
    assert isinstance(samples_two, jnp.ndarray)
    assert samples_two.shape == (20, 2)

    # posterior.sample should not have any side-effect:
    # two consecutive posterior.sample calls should return
    # the same samples
    samples_one = posterior.sample(subkey, (10,), return_updated_dist=False)
    assert isinstance(samples_one, jnp.ndarray)
    assert samples_one.shape == (10, 2)

    samples_two = posterior.sample(subkey, (10,), return_updated_dist=False)
    assert jnp.allclose(samples_one, samples_two)

    # Test posterior sampling algorithm updating
    # feature.
    samples_one, posterior_one = posterior.sample(
        subkey, (10,), return_updated_dist=True
    )
    samples_two, posterior_two = posterior_one.sample(
        subkey, (10,), return_updated_dist=True
    )
    samples_three, _ = posterior_two.sample(subkey, (10,), return_updated_dist=True)
    assert isinstance(samples_two, jnp.ndarray)
    assert not jnp.allclose(samples_one, samples_two)
    assert not jnp.allclose(samples_two, samples_three)
