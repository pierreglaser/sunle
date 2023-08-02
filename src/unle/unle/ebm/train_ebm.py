from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple, cast

from typing_extensions import Unpack
from unle.ebm.base import BaseEnergyFnWrapper, BaseTrainer, Batch, TrainingConfig

if TYPE_CHECKING:
    from jax.random import KeyArray
    from unle.samplers.inference_algorithms.base import (
        InferenceAlgorithm,
    )
    from unle.samplers.particle_aproximation import ParticleApproximation

import jax.numpy as jnp
from jax import Array, grad, random, vmap
from jax.random import fold_in
from jax.tree_util import tree_map
from numpyro import distributions as np_distributions
from unle.samplers.inference_algorithms.mcmc.base import MCMCAlgorithm
from unle.typing import PyTreeNode


class EnergyFnWrapper(BaseEnergyFnWrapper[Unpack[Tuple[Array]]]):
    def __call__(self, x: Array) -> Any:
        return -self.energy_fn(self.params, x)


class Trainer(BaseTrainer[Array]):
    dim_z: int

    def __init__(self, dim_z: int):
        self.dim_z = dim_z

    def compute_ebm_approx(
        self,
        alg: InferenceAlgorithm,
        params: PyTreeNode,
        key: KeyArray,
        true_samples: Batch,
        config: TrainingConfig,
    ) -> Tuple[InferenceAlgorithm, ParticleApproximation]:
        alg = alg.set_log_prob(
            cast(EnergyFnWrapper, alg.log_prob).replace(params=params)
        )
        # call the class method to prevent spurious recompilations.
        key, subkey = random.split(key)
        alg, results = type(alg).run_and_update_init(alg, subkey)
        return alg, results.samples

    def estimate_log_likelihood_gradient(
        self,
        params: PyTreeNode,
        true_samples: Batch[Tuple[Unpack[Tuple[Array]]]],
        ebm_samples: ParticleApproximation,
        noise_injection_val: float,
        key: KeyArray,
        log_joint: BaseEnergyFnWrapper[Unpack[Tuple[Array]]],
    ) -> PyTreeNode:
        noise: Array = noise_injection_val * random.normal(
            key, true_samples.batch[0].shape
        )

        energy_true_samples = grad(
            lambda p: jnp.average(
                vmap(log_joint.energy_fn, in_axes=(None, 0))(
                    p,
                    jnp.concatenate(
                        [
                            true_samples.batch[0][:, : self.dim_z],
                            true_samples.batch[0][:, self.dim_z :]
                            + noise[:, self.dim_z :],
                        ],
                        axis=1,
                    ),
                ),
                axis=0,
            )
        )(params)

        energy_ebm_samples = grad(
            lambda p: jnp.average(
                vmap(log_joint.energy_fn, in_axes=(None, 0))(p, ebm_samples.xs),
                weights=ebm_samples.normalized_ws,
                axis=0,
            )
        )(params)

        return tree_map(lambda x, y: x - y, energy_true_samples, energy_ebm_samples)

    def _resolve_proposal_distribution(
        self, config: TrainingConfig
    ) -> np_distributions.Distribution:
        assert isinstance(config.sampling_init_dist, np_distributions.Distribution)
        return config.sampling_init_dist

    def _resolve_proposal_particles(
        self, config: TrainingConfig, dataset: Tuple[Array], key: KeyArray
    ) -> Array:
        key, key_zx0s = random.split(key)

        indexes = random.choice(key_zx0s[0], len(dataset), (config.num_particles,))
        return dataset[indexes]

    def _initialize_sampling_alg(
        self,
        log_prob: BaseEnergyFnWrapper[Array],
        config: TrainingConfig,
        dataset: Tuple[Array],
        params: PyTreeNode,
        key: KeyArray,
        use_first_iter_cfg: bool = False,
    ) -> InferenceAlgorithm:
        assert config.num_particles is not None  # type narrowing
        assert config.sampling_init_dist is not None  # type narrowing

        if use_first_iter_cfg:
            algs = config.sampling_cfg_first_iter.build_algorithm(
                log_prob=log_prob.replace(params=params)
            )
        else:
            algs = config.sampling_cfg.build_algorithm(
                log_prob=log_prob.replace(params=params)
            )

        if isinstance(config.sampling_init_dist, np_distributions.Distribution):
            dists = self._resolve_proposal_distribution(config)
            key, subkey = random.split(key)
            algs = algs.init(fold_in(subkey, 0), dists)  # type: ignore
        else:
            particles = self._resolve_proposal_particles(config, dataset, key)
            assert isinstance(algs, MCMCAlgorithm)
            algs = algs.init_from_particles(particles)
        return algs

    def make_log_prob(self, energy_fn, params) -> EnergyFnWrapper:
        self.log_joint = EnergyFnWrapper(
            energy_fn=energy_fn,
            params=params,
        )
        return self.log_joint
