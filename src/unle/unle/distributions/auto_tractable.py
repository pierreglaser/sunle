from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    Generic,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

if TYPE_CHECKING:
    from jax.random import KeyArray
    from typing_extensions import Self

import copy

import jax.numpy as jnp
from jax import Array, jit, random
from jax.tree_util import tree_leaves, tree_map
from numpyro import distributions as np_distributions
from numpyro.distributions import TransformedDistribution
from unle.distributions.base import (
    ConditionalDistributionBase,
    np_distribution_unflatten,
)
from unle.samplers.inference_algorithms.base import InferenceAlgorithmFactory
from unle.samplers.inference_algorithms.importance_sampling.smc import (
    SMCConfig,
    SMCFactory,
)
from unle.samplers.inference_algorithms.mcmc.base import (
    MCMCAlgorithm,
    MCMCAlgorithmFactory,
    MCMCConfig,
    MCMCResults,
)
from unle.samplers.kernels.mala import MALAConfig, MALAKernelFactory
from unle.samplers.kernels.rwmh import RWConfig, RWKernelFactory
from unle.samplers.kernels.savm import SAVMConfig, SAVMKernelFactory
from unle.typing import Numeric, PyTreeNode


def tree_all(function: Callable[[PyTreeNode], Numeric], tree: PyTreeNode) -> Numeric:
    mapped_tree = tree_map(function, tree)
    return jnp.all(jnp.array(tree_leaves(mapped_tree)))


def _get_default_inference_mala_config() -> InferenceAlgorithmFactory:
    config = MCMCAlgorithmFactory(
        config=MCMCConfig(
            kernel_factory=MALAKernelFactory(config=MALAConfig(0.01)),
            num_samples=100,
            num_chains=100,
            thinning_factor=20,
            num_warmup_steps=500,
            adapt_mass_matrix=False,
            adapt_step_size=True,
            progress_bar=True,
            target_accept_rate=0.2,
            init_using_log_l_mode=False,
        )
    )
    return config


def _get_default_inference_smc_config() -> InferenceAlgorithmFactory:
    inner_kernel_factory = MALAKernelFactory(MALAConfig(0.01))
    config = SMCFactory(
        config=SMCConfig(
            num_samples=1,
            ess_threshold=0.8,
            inner_kernel_factory=inner_kernel_factory,
            num_steps=3000,
            inner_kernel_steps=3,
            record_trajectory=False,
        )
    )
    return config


def _get_default_inference_savm_config() -> InferenceAlgorithmFactory:
    config = MCMCAlgorithmFactory(
        config=MCMCConfig(
            kernel_factory=SAVMKernelFactory(
                config=SAVMConfig(
                    aux_var_kernel_factory=MALAKernelFactory(MALAConfig(0.1)),
                    aux_var_num_inner_steps=100,
                    base_var_kernel_factory=RWKernelFactory(config=RWConfig(0.1, None)),
                    aux_var_init_strategy="x_obs",
                )
            ),
            num_samples=100,
            num_chains=100,
            thinning_factor=20,
            num_warmup_steps=2000,
            adapt_mass_matrix=False,
            adapt_step_size=True,
            progress_bar=True,
            target_accept_rate=0.2,
            init_using_log_l_mode=False,
        )
    )
    return config


def make_inference_config(
    conditional_distribution: ConditionalDistributionBase,
    type_: str = "mcmc",
    num_warmup_steps: Optional[int] = None,
    exchange_mcmc_inner_sampler_num_steps: Optional[int] = None,
):
    if type_ == "mcmc":
        if conditional_distribution.has_theta_dependent_normalizer:
            inference_config = _get_default_inference_savm_config()
        else:
            inference_config = _get_default_inference_mala_config()

        assert isinstance(inference_config, MCMCAlgorithmFactory)
        inference_config = inference_config.replace(
            config=inference_config.config.replace(
                num_warmup_steps=num_warmup_steps,
            )
        )
        if isinstance(inference_config.config.kernel_factory, SAVMKernelFactory):
            inference_config = inference_config.replace(
                config=inference_config.config.replace(
                    kernel_factory=inference_config.config.kernel_factory.replace(
                        config=inference_config.config.kernel_factory.config.replace(
                            aux_var_num_inner_steps=exchange_mcmc_inner_sampler_num_steps  # noqa: E501
                        )
                    )
                )
            )
        return inference_config
    else:
        assert type_ == "smc"
        return _get_default_inference_smc_config()


D = TypeVar("D", bound=np_distributions.Distribution)


class AutoTractableDistributionBase(np_distributions.Distribution, Generic[D]):
    dist: D

    def __init__(
        self,
        dist: D,
        config: Optional[InferenceAlgorithmFactory] = None,
        init_distribution: Optional[np_distributions.Distribution] = None,
        sample_in_base_space: bool = True,
    ):
        """
        Wrapper around intractable distribution which implements a `sample`
        method that uses an approximate algorithm under the hood
        (either MCMC or SMC). To amortize the possible warmup
        period induced by such algorithm across `sample` calls,
        `sample` can be set to keep track of the internal state of the MCMC
        algorithm, see the `AutoTractableDistributionBase.sample`
        documentation.

        Parameters
        ----------
        dist : np_distribution.Distribution
            Intractable distribution to wrap.
        config : InferenceAlgorithmFactory, optional
            Instance containing the configuration for the MCMC algorithm used
            to perform posterior sampling.
        init_distribution : np_distribution.Distribution, optional
            Distribution used to initialize the MCMC algorithm.
        """

        assert config is not None
        self.inference_config = config
        # self.support = self.z_scorer.get_transform("params").domain
        # assumes jittability using jittable_method_descriptor
        self.sample_in_base_space = sample_in_base_space
        if sample_in_base_space and isinstance(dist, TransformedDistribution):
            self.sampling_alg = self.inference_config.build_algorithm(
                dist.base_dist.log_prob
            )  # type: ignore
        else:
            self.sampling_alg = self.inference_config.build_algorithm(
                dist.log_prob
            )  # type: ignore

        self.dist = dist
        assert init_distribution is not None
        self.init_distribution = init_distribution

        np_distributions.Distribution.__init__(self, dist.batch_shape, dist.event_shape)

    def tree_flatten(self):  # pyright: ignore [reportIncompatibleMethodOverride]
        return (
            self.dist,
            self.inference_config,
            self.sampling_alg,
            self.sample_in_base_space,
        ), ()

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        (dist, inference_config, alg, sample_in_base_space) = params
        obj = np_distribution_unflatten(cls, (dist.batch_shape, dist.event_shape), ())
        obj.inference_config = inference_config
        obj.sampling_alg = alg
        obj.sample_in_base_space = sample_in_base_space
        return obj

    @property
    def log_prob(self):  # pyright: ignore [reportIncompatibleMethodOverride]
        return self.dist.log_prob

    @overload
    def sample(
        self,
        key: KeyArray,
        sample_shape: tuple = (),
        return_updated_dist: Literal[False] = False,
    ) -> Array:
        ...

    @overload
    def sample(
        self,
        key: KeyArray,
        sample_shape: tuple = (),
        return_updated_dist: Literal[True] = True,
    ) -> Tuple[Array, Self]:
        ...

    @overload
    def sample(
        self,
        key: KeyArray,
        sample_shape: tuple = (),
        return_updated_dist: Literal[True, False] = False,
    ) -> Union[Array, Tuple[Array, Self]]:
        ...

    def sample(
        self,
        key: KeyArray,
        sample_shape: tuple = (),
        return_updated_dist: Literal[True, False] = False,
    ) -> Union[Array, Tuple[Array, Self]]:
        """
        (Approximately) Sample from an intractable distribution.

        MCMC sampling is warm-started across consecutive `self.sample` calls

        Parameters
        ----------
        num_samples : int
            number of posterior samples to return.
        key : jax.random.KeyArray
        return_updated_dist : bool
            if True, a new `AutoTractableDistributionBase` object is returned,
            which represents the same distribution as `self`,
            but contains an updated version of `self`'s sampling
            state manipulated by the approximate sampling algorithms.
            This flag only has an effect if the internal sampling
            state can be re-used across `sample` calls. This is the
            case for MCMC algorithms (in which case the state is
            the current position and step sizes of the MCMC chains),
            but not (naively) the case for SMC algorithms.

        Returns
        -------
        posterior samples : jax.Array
            approximate posterior samples
        dist : AutoTractableDistributionBase
            posterior distribution with an updated internal sampling state
        """
        assert self.sampling_alg is not None
        alg = self.sampling_alg
        assert len(sample_shape) == 1
        num_samples = sample_shape[0]

        if alg.can_set_num_samples:
            alg = alg.set_num_samples(num_samples)
        else:
            config = self.inference_config.replace(
                config=self.inference_config.config.replace(num_samples=num_samples)
            )

            alg = config.build_algorithm(self.dist.base_dist.log_prob)  # type: ignore

        if not alg.initialized:
            key, subkey = random.split(key)
            alg = alg.init(key=subkey, dist=self.init_distribution)

        key, subkey = random.split(key)

        all_finite_before = tree_all(lambda x: jnp.all(jnp.isfinite(x)), (alg))
        print(f"all finite before sampling: {all_finite_before}")

        alg, results = jit(alg.run)(subkey)
        # alg, results = alg.run(subkey)

        all_finite = tree_all(lambda x: jnp.all(jnp.isfinite(x)), (alg, results))
        print(f"all finite after sampling: {all_finite}")
        posterior_samples = results.samples.xs

        if isinstance(results, MCMCResults):
            assert isinstance(alg, MCMCAlgorithm)
            alg = alg.reset_at_final_state(
                results.info.single_chain_results.final_state
            )

        if self.sample_in_base_space:
            assert isinstance(self.dist, TransformedDistribution)
            for transform in self.dist.transforms:
                posterior_samples = cast(Array, transform(posterior_samples))

        if return_updated_dist:
            new_posterior = copy.copy(self)
            new_posterior.sampling_alg = alg
            return posterior_samples, new_posterior
        else:
            return posterior_samples


class AutoTractableConditionalDistribution(  # pyright: ignore [reportIncompatibleMethodOverride]  # noqa: E501
    AutoTractableDistributionBase[ConditionalDistributionBase],
    ConditionalDistributionBase,
):
    @property
    def condition(self):
        return self.dist.condition

    @property
    def has_theta_dependent_normalizer(self):
        return self.dist.has_theta_dependent_normalizer

    @property
    def conditioned_event_shape(self):
        return self.dist.conditioned_event_shape
