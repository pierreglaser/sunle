from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, cast

from jax import Array
from typing_extensions import Unpack
from unle.ebm.base import BaseEnergyFnWrapper, BaseTrainer
from unle.utils.vmapped_mcmc import VmappedMCMC, VmappedMCMCFactory

if TYPE_CHECKING:
    from jax.random import KeyArray
    from unle.samplers.inference_algorithms.base import InferenceAlgorithm
    from unle.typing import PyTreeNode

import jax.numpy as jnp
from jax import grad, random, vmap
from jax.tree_util import tree_map
from numpyro import distributions as np_distributions
from unle.ebm.train_ebm import Batch, TrainingConfig
from unle.samplers.distributions import ThetaConditionalLogDensity
from unle.samplers.inference_algorithms.mcmc.base import MCMCAlgorithmFactory
from unle.samplers.particle_aproximation import ParticleApproximation


def maybe_reshape(x):
    import jax.numpy as jnp

    if len(x.shape) >= 3:
        return jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:]))
    elif len(x.shape) >= 2:
        return jnp.reshape(x, (x.shape[0] * x.shape[1],))
    else:
        raise ValueError("Can't reshape")


class ConditionalEnergyFnWrapper(BaseEnergyFnWrapper[Unpack[Tuple[Array, Array]]]):
    def __call__(self, theta, x) -> Array:
        return -self.energy_fn(self.params, theta, x)


class LikelihoodTrainer(BaseTrainer[Unpack[Tuple[Array, Array]]]):
    def make_log_prob(self, energy_fn, params) -> ConditionalEnergyFnWrapper:
        return ConditionalEnergyFnWrapper(energy_fn, params)

    def estimate_log_likelihood_gradient(
        self,
        params: PyTreeNode,
        true_samples: Batch[Tuple[Array, Array]],
        ebm_samples: ParticleApproximation,
        noise_injection_val: float,
        key: KeyArray,
        log_joint: BaseEnergyFnWrapper[Unpack[Tuple[Array, Array]]],
    ) -> PyTreeNode:
        num_particles = min(ebm_samples.num_samples, len(true_samples.batch[0]))
        print(f"using {num_particles} particles to estimate log likelihood gradient")

        noise: Array = noise_injection_val * random.normal(
            key, jnp.concatenate(true_samples.batch, axis=1).shape
        )

        energy_true_samples = grad(
            lambda p: jnp.average(
                vmap(log_joint.energy_fn, in_axes=(None, 0, 0))(
                    p,
                    true_samples.batch[0],
                    true_samples.batch[1] + noise[:, self.dim_z :],
                ),
                axis=0,
            )
        )(params)

        energy_ebm_samples = grad(
            lambda p: jnp.average(
                vmap(log_joint.energy_fn, in_axes=(None, 0, 0))(
                    p,
                    true_samples.batch[0][:num_particles],
                    ebm_samples.particles,
                ),
                weights=ebm_samples.normalized_ws,
                axis=0,
            )
        )(params)

        return tree_map(lambda x, y: x - y, energy_true_samples, energy_ebm_samples)

    def _initialize_sampling_alg(
        self,
        log_prob: BaseEnergyFnWrapper[Unpack[Tuple[Array, Array]]],
        config: TrainingConfig,
        dataset: Tuple[Array, Array],
        params: PyTreeNode,
        key: KeyArray,
        use_first_iter_cfg: bool = False,
    ) -> InferenceAlgorithm:
        assert config.sampling_init_dist is not None  # type narrowing
        if isinstance(config.sampling_init_dist, np_distributions.Distribution):
            # for likelihood-based training, we keep track of a particle xⁱ
            # per training point Θⁱ (sampled from p(x|Θⁱ; ψ)), but update
            # only `num_particles` per iteration.
            key, subkey = random.split(key)
            particles = config.sampling_init_dist.sample(
                key=subkey, sample_shape=len(dataset)
            )
            thetas = dataset[0]
        else:
            thetas, particles = dataset[0], dataset[1]

        likelihoods = ThetaConditionalLogDensity(
            log_prob.replace(params=params), thetas
        )
        assert isinstance(config.sampling_cfg, MCMCAlgorithmFactory)

        if use_first_iter_cfg:
            factory = config.sampling_cfg_first_iter
        else:
            factory = config.sampling_cfg

        vmapped_factory = VmappedMCMCFactory.from_mcmc_factory(
            config.sampling_cfg.replace(
                config=factory.config.replace(num_samples=1, num_chains=1)
            ),
            likelihoods.vmap_axes(likelihoods, None, 0),
        )
        algs = vmapped_factory.build_algorithm(likelihoods)
        algs = algs.init_from_particles(particles[:, None, :])
        return algs

    def compute_ebm_approx(
        self,
        alg: InferenceAlgorithm,
        params: PyTreeNode,
        key: KeyArray,
        true_samples: Batch[Tuple[Array, Array]],
        config: TrainingConfig,
    ) -> Tuple[InferenceAlgorithm, ParticleApproximation]:
        assert isinstance(alg, VmappedMCMC)
        num_particles = min(config.num_particles, len(true_samples.batch[0]))
        print(f"Using {num_particles} particles for EBM approximation")

        assert isinstance(alg.log_prob, ThetaConditionalLogDensity)

        log_prob = alg.log_prob.replace(
            log_prob=alg.log_prob.log_prob.replace(params=params)
        )
        alg = alg.set_log_prob(log_prob=log_prob)
        this_iter_algs = alg.get_slice(
            true_samples.indices[:num_particles],
            excluded_node_types=(ConditionalEnergyFnWrapper,),
        )

        key, subkey = random.split(key)
        nta, results = this_iter_algs.run_and_update_init(
            random.split(subkey, this_iter_algs.num_algs)
        )

        # mcmc algorithm only has one sample per likelihood p(x|Θⁱ; ψ))
        ebm_samples_xs = cast(
            ParticleApproximation, tree_map(lambda x: x[:, 0, ...], results.samples)
        )

        updated_alg = alg.set_slice(
            true_samples.indices[:num_particles],
            nta,
            excluded_node_types=(ConditionalEnergyFnWrapper,),
        )
        return updated_alg, ebm_samples_xs
