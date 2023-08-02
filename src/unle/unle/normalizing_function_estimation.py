# See issue #620.
# pytype: disable=wrong-keyword-args
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from jax import Array
    from jax.random import KeyArray

    from unle.typing import PyTreeNode

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from jax import grad, random
from jax.tree_util import tree_map
from numpyro.distributions.transforms import AffineTransform
from typing_extensions import Self

from unle.distributions.base import ConditionalDistributionBase
from unle.neural_networks.neural_networks import MLP, MLPConfig
from unle.neural_networks.regression import RegressionTrainingConfig, train_regressor
from unle.samplers.inference_algorithms.mcmc.base import (
    MCMCAlgorithmFactory,
    MCMCConfig,
)
from unle.samplers.kernels.mala import MALAConfig, MALAKernelFactory
from unle.utils.reparametrization import compose_dense_and_componentwise_transform
from unle.utils.vmapped_mcmc import VmappedMCMC, VmappedMCMCFactory


def _get_default_config():
    sampling_factory = MCMCAlgorithmFactory(
        config=MCMCConfig(
            kernel_factory=MALAKernelFactory(config=MALAConfig(0.01)),
            num_samples=1000,
            num_chains=1,
            thinning_factor=1,
            num_warmup_steps=300,
            adapt_mass_matrix=False,
            adapt_step_size=True,
            init_using_log_l_mode=False,
            target_accept_rate=0.5,
        )
    )
    return sampling_factory


class LogZNet(struct.PyTreeNode):
    likelihood: ConditionalDistributionBase
    params: Optional[PyTreeNode] = None
    config: MLPConfig = MLPConfig(100, 4)
    scale: float = struct.field(pytree_node=False, default=1.0)
    bias: Optional[Array] = struct.field(pytree_node=False, default=0.0)
    all_lz_net_sampling_algs: Optional[VmappedMCMC] = None
    sampling_factory: MCMCAlgorithmFactory = _get_default_config()
    train_thetas: Optional[Array] = None
    train_grad_energy: Optional[Array] = None

    @classmethod
    def create(
        cls,
        likelihood: ConditionalDistributionBase,
        config: MLPConfig = MLPConfig(100, 4),
        sampling_factory: Optional[MCMCAlgorithmFactory] = None,
    ):
        if sampling_factory is None:
            sampling_factory = _get_default_config()

        return cls(
            config=config,
            likelihood=likelihood,
            sampling_factory=sampling_factory,
        )

    def __call__(self, theta: Array) -> Array:
        vals = MLP(width=self.config.width, depth=self.config.depth).apply(
            {"params": self.get_params()}, theta
        )[0]
        # pyright type narrowing: in spite of its type signature,
        # `flax.linen.Module.apply` doesn't return a Tuple
        # when mutable=False.
        assert not isinstance(vals, Tuple)

        if self.bias is None:
            return self.scale * vals
        else:
            return self.scale * vals + jnp.dot(self.bias, theta)

    def get_algs(self):
        assert self.all_lz_net_sampling_algs is not None
        return self.all_lz_net_sampling_algs

    def get_params(self):
        assert self.params is not None
        return self.params

    def get_thetas(self):
        assert self.train_thetas is not None
        return self.train_thetas

    def set_likelihood(self, likelihood: ConditionalDistributionBase):
        self = self.replace(likelihood=likelihood)
        return self

    def _likelihood_vmap_axes(self):
        return self.likelihood.log_prob.vmap_axes(obj=self.likelihood.vmap_axes(0))

    # Training Data Creation ---------------------------------------------------

    def add_new_points(
        self,
        params: Array,
        observations: Array,
    ) -> Self:
        print("(building new chains): thetas_this_round.shape", params.shape)

        vmapped_factory = VmappedMCMCFactory.from_mcmc_factory(
            self.sampling_factory,
            log_prob_vmap_axes=self._likelihood_vmap_axes(),
        )

        algs = vmapped_factory.build_algorithm(
            self.likelihood.set_condition(params).log_prob
        )

        algs = algs.init_from_particles(observations[:, None, :])

        if self.all_lz_net_sampling_algs is not None:
            algs = self._concatenate_algs(self.all_lz_net_sampling_algs, algs)
            params = jnp.concatenate([self.get_thetas(), params], axis=0)

        return self.replace(all_lz_net_sampling_algs=algs, train_thetas=params)

    # Reparametrization --------------------------------------------------------

    def reparametrize(
        self,
        param_map: AffineTransform,
        observation_map: AffineTransform,
    ):
        """
        Given map :math:`f_𝒳`,  :math:`f_𝚯 and a LogZNet of a conditional distribution
        p(x | Θ), return a new LogZNet for the conditional distribution
        :math:`(f_𝒳)_# p(x | f_𝚯(Θ))`. Useful for cross-round adaptation.
        """
        self = self._reparametrize_net(param_map)
        self = self._reparametrize_algs(
            param_map,
            observation_map,
        )
        # update existing training set. grad_energies stays the same since
        # ∇_Θ log (f_𝒳)_# p(y|f_𝚯(Θ)) = ∇_Θ (log p(f^{-1}(y)|Θ) + C(x))
        #                             = ∇_Θ log p(x|Θ)
        thetas = self.get_thetas()
        new_thetas = param_map(thetas)
        return self.replace(train_thetas=new_thetas)

    def _reparametrize_net(
        self,
        map: AffineTransform,
    ):
        """
        Given a map :math:`f`, and a LogZNet of a conditional distribution
        p(x | theta), return a new LogZNet for the conditional distribution
        p(x | f(theta)). Useful for cross-round adaptation.
        """
        params = self.get_params()
        outermost_scale = params["layers_0"]["kernel"]
        outermost_bias = params["layers_0"]["bias"]

        new_kernel, new_bias = compose_dense_and_componentwise_transform(
            outermost_scale,
            outermost_bias,
            1 / map.scale,
            -map.loc / map.scale,
        )
        new_params = params.unfreeze().copy()

        new_params["layers_0"]["kernel"] = new_kernel
        new_params["layers_0"]["bias"] = new_bias

        return self.replace(params=FrozenDict(new_params))

    def _reparametrize_algs(
        self,
        param_map: AffineTransform,
        observation_map: AffineTransform,
    ) -> Self:
        all_algs = self.get_algs()
        thetas = self.get_thetas()
        new_thetas = param_map(thetas)

        # update the conditioned variable value of each targeted
        # conditional probability: p(x|Θ) -> p(x|f_𝚯(Θ))
        new_all_algs = all_algs.set_log_prob(
            self.likelihood.set_condition(new_thetas).log_prob
        )
        # update the algorithm's state such that they target
        # (f_𝒳)_# p(x | f_𝚯(Θ)) instead of p(x | f_𝚯(Θ))
        new_all_algs = new_all_algs.reparametrize(observation_map)
        return self.replace(
            all_lz_net_sampling_algs=new_all_algs,
        )

    def _concatenate_algs(self, alg1: VmappedMCMC, alg2: VmappedMCMC) -> VmappedMCMC:
        all_algs_concatenated: VmappedMCMC = tree_map(
            lambda x1, x2: (
                x1.set_condition(jnp.concatenate((x1.condition, x2.condition)))
                if isinstance(x1, ConditionalDistributionBase)
                else jnp.concatenate((x1, x2), axis=0)
            ),
            alg1,
            alg2,
            is_leaf=lambda x: isinstance(x, ConditionalDistributionBase),
        )
        return all_algs_concatenated

    def energy(self, x, theta):
        return -self.likelihood.log_prob_override_conditionned(theta, x)

    def create_training_data(self, key: KeyArray) -> Self:
        key, subkey = random.split(key)
        key, subkey = random.split(key)
        print("computing log-z net expectaions in an online fashion")
        algs = self.get_algs()

        (
            algs,
            mean_grad_energy,
            results,
        ) = algs.estimate_expectation_of(
            grad(self.energy, argnums=1),
            random.split(subkey, algs.num_algs),
            self.get_thetas(),
        )
        algs = algs.reset_at_final_state(results.final_state)

        minus_mean_grad_energy = -mean_grad_energy
        return self.replace(
            all_lz_net_sampling_algs=algs,
            train_grad_energy=minus_mean_grad_energy,
        )

    def train(
        self,
        key: KeyArray,
        training_config: RegressionTrainingConfig,
        z_score_output: bool = True,
    ) -> LogZNet:
        def predict_fn(params, theta):
            """
            Given a conditional EBM p(x|Θ; ψ) = exp(-E_ψ(x, Θ)) / Z(Θ, ψ),
            estimate the the log partition function Z(Θ, ψ) w.r.t. Θ by
            minimizing the squared error between the gradient of the log partition
            function and the gradient of the energy function. The validity of this
            operation follows from the fact that
            ∇_Θ(log(Z(Θ,ψ)) = -E_x|Θ(∇_Θ(E_ψ(x, Θ))),
            """

            def loss_fn(theta):
                vals = MLP(width=self.config.width, depth=self.config.depth).apply(
                    {"params": params}, theta
                )[0]
                # pyright type narrowing: in spite of its type signature,
                # `flax.linen.Module.apply` doesn't return a Tuple
                # when mutable=False.
                assert not isinstance(vals, Tuple)
                return vals

            return jax.grad(loss_fn)(theta)

        # Standardize and optionally z-score the energy gradients
        minus_grad_energy = self.train_grad_energy
        thetas = self.train_thetas
        assert minus_grad_energy is not None
        assert thetas is not None

        # first standardize the output
        std = jnp.std(
            minus_grad_energy
        )  # LogZNet cannot easily handle dimension-dependent scaling
        bias = jnp.mean(minus_grad_energy, axis=0)

        if z_score_output:
            minus_grad_energy = (minus_grad_energy - bias) / std
        else:
            std = jnp.ones_like(std)
            bias = jnp.zeros_like(bias)

        state = train_regressor(
            thetas,
            minus_grad_energy,
            key,
            training_config,
            init_params=self.params,
            apply_fn=predict_fn,
            init_fn=MLP(width=self.config.width, depth=self.config.depth).init,
        )
        return self.replace(params=state.params, bias=bias, scale=std)
