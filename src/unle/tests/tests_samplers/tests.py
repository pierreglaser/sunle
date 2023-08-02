import jax.numpy as jnp
from jax import random
from unle.samplers.distributions import maybe_wrap
from unle.samplers.inference_algorithms.mcmc.base import (
    MCMCAlgorithm,
    MCMCConfig,
)
from unle.samplers.kernels.mala import (
    MALAConfig,
    MALAInfo,
    MALAKernelFactory,
    MALAState,
)


def make_basic_mcmc_config(
    num_samples: int = 1000,
) -> MCMCConfig[MALAConfig, MALAState, MALAInfo]:
    kf = MALAKernelFactory(MALAConfig(step_size=0.1))
    config = MCMCConfig(
        num_samples=num_samples,
        kernel_factory=kf,
        num_chains=100,
        thinning_factor=1,
        record_trajectory=False,
        num_warmup_steps=1000,
        adapt_step_size=True,
        adapt_mass_matrix=False,
        progress_bar=False,
        warmup_method="jax_samplers",
        init_using_log_l_mode=False,
    )
    return config


def make_basic_mcmc_alg(
    num_samples: int = 1000,
) -> MCMCAlgorithm[MALAConfig, MALAState, MALAInfo]:
    config = make_basic_mcmc_config(num_samples=num_samples)
    log_prob = maybe_wrap(lambda x: -0.5 * jnp.sum(x**2))
    alg = MCMCAlgorithm.create(config, log_prob)
    return alg


def test_online_equals_offline():
    alg = make_basic_mcmc_alg()

    init = jnp.zeros((alg.config.num_chains, 2))

    alg = alg.init_from_particles(init)

    def f(x):
        return x

    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    _, results = alg.run(subkey)
    avg_offline = jnp.mean(
        results.info.single_chain_results.chain.x.reshape(-1, 2), axis=0
    )

    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    _, avg_online, _ = alg.estimate_expectation_of(f, subkey)
    assert jnp.allclose(avg_offline, avg_online)
