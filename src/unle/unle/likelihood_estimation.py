from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from typing import Callable, Literal, Optional, Tuple

    from jax import Array
    from jax.random import KeyArray
    from typing_extensions import Self

    from unle.normalizing_function_estimation import LogZNet
    from unle.typing import PyTreeNode

import copy

import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from numpyro import distributions as np_distributions
from numpyro.distributions.transforms import AffineTransform

from unle.distributions.base import BlockDistribution, ConditionalDistributionBase
from unle.ebm.base import OptimizerConfig, TrainerResults, TrainingConfig
from unle.ebm.train_conditional_ebm import LikelihoodTrainer
from unle.ebm.train_ebm import Trainer
from unle.neural_networks.neural_networks import MLP
from unle.samplers.inference_algorithms.importance_sampling.smc import (
    SMCConfig,
    SMCFactory,
)
from unle.samplers.inference_algorithms.mcmc.base import (
    MCMCAlgorithmFactory,
    MCMCConfig,
)
from unle.samplers.kernels.adaptive_mala import (
    AdaptiveMALAConfig,
    AdaptiveMALAKernelFactory,
)
from unle.samplers.kernels.mala import MALAConfig, MALAKernelFactory
from unle.utils.jax import jittable_method_descriptor
from unle.utils.reparametrization import compose_dense_and_componentwise_transform


def _default_base_measure_log_prob(x: Array) -> Array:
    return -0.5 * jnp.sum(jnp.square((x / 10)))


class EBMLikelihoodConfig(struct.PyTreeNode):
    base_measure_log_prob: Callable[[Array], Array] = struct.field(
        pytree_node=False, default=_default_base_measure_log_prob
    )
    width: int = struct.field(pytree_node=False, default=50)
    depth: int = struct.field(pytree_node=False, default=4)


def _make_sampling_cfg(
    dim_data,
    proposal,
    ebm_model_type,
    num_frozen_steps: int = 50,
    num_mala_steps: int = 50,
    num_smc_steps: int = 5,
    num_particles: int = 1000,
    ess_threshold: float = 0.8,
):
    if ebm_model_type == "likelihood":
        sampling_cfg = MCMCAlgorithmFactory(
            MCMCConfig(
                kernel_factory=MALAKernelFactory(config=MALAConfig(0.1, None)),
                num_samples=1000,
                num_chains=1000,
                thinning_factor=num_frozen_steps,
                num_warmup_steps=num_mala_steps,
                adapt_step_size=True,
                init_using_log_l_mode=False,
                # target_accept_rate=0.2,
            )
        )
        sampling_cfg_first_iter = sampling_cfg.replace(
            config=sampling_cfg.config.replace(num_warmup_steps=100)
        )
        sampling_init_dist = "data"
    else:
        inner_kernel_factory = AdaptiveMALAKernelFactory(
            AdaptiveMALAConfig(0.1, update_cov=False, use_dense_cov=False)
        )
        sampling_cfg = SMCFactory(
            SMCConfig(
                num_samples=num_particles,
                ess_threshold=ess_threshold,
                inner_kernel_factory=inner_kernel_factory,
                num_steps=num_smc_steps,
                inner_kernel_steps=num_mala_steps,
                # num_step_sizes=100
            )
        )
        sampling_cfg_first_iter = sampling_cfg.replace(
            config=sampling_cfg.config.replace(num_steps=10)
        )
        x_dist = np_distributions.MultivariateNormal(
            jnp.zeros((dim_data,)),  # type: ignore
            jnp.eye(dim_data),  # type: ignore
        )
        theta_dist = proposal
        sampling_init_dist = BlockDistribution(distributions=[theta_dist, x_dist])

    return sampling_cfg_first_iter, sampling_cfg, sampling_init_dist


def make_training_config(
    ebm_model_type,
    dim_data,
    proposal,
    max_iter: int = 500,
    num_frozen_steps: int = 50,
    num_mala_steps: int = 50,
    num_particles: int = 1000,
    use_warm_start: bool = True,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-1,
    noise_injection_val: float = 0.001,
    batch_size: Optional[int] = None,
    num_smc_steps: int = 5,
    ess_threshold: float = 0.8,
):
    optimizer_config = OptimizerConfig(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        noise_injection_val=noise_injection_val,
    )
    cfg_first_iter, cfg, init_dist = _make_sampling_cfg(
        dim_data,
        proposal,
        ebm_model_type,
        num_frozen_steps,
        num_mala_steps,
        num_smc_steps,
        num_particles,
        ess_threshold,
    )
    return TrainingConfig(
        max_iter=max_iter,
        sampling_cfg_first_iter=cfg_first_iter,
        sampling_cfg=cfg,
        sampling_init_dist=init_dist,
        num_particles=num_particles,
        use_warm_start=use_warm_start,
        optimizer=optimizer_config,
        batch_size=batch_size,
    )


class EBMLikelihood(ConditionalDistributionBase):
    def __init__(
        self,
        ebm_width: int,
        ebm_depth: int,
        event_shape: Tuple[int],
        conditioned_event_shape: Tuple[int],
        param: Optional[PyTreeNode] = None,
        params: Optional[PyTreeNode] = None,
        key: Optional[KeyArray] = None,
        has_theta_dependent_normalizer: bool = True,
        log_z_net: Optional[LogZNet] = None,
    ):
        assert (log_z_net is None) == has_theta_dependent_normalizer

        super(EBMLikelihood, self).__init__(
            batch_shape=(),
            event_shape=event_shape,
            conditioned_event_shape=conditioned_event_shape,
            condition=param,
            has_theta_dependent_normalizer=has_theta_dependent_normalizer,
        )

        self.config = EBMLikelihoodConfig(
            width=ebm_width,
            depth=ebm_depth,
        )
        self.needs_filtering_correction_if_nans = None
        self.log_z_net = log_z_net
        self._has_theta_dependent_normalizer = has_theta_dependent_normalizer

        if params is None:
            assert key is not None
            self.params = self._initialize_params(key)
        else:
            self.params = params

    def reparametrize(
        self,
        map: AffineTransform,
    ):
        """
        Given a map :math:`f`, returns a new EBM, such that
        the new EBM density is the pushforward density of the
        current EBM under :math:`f`.

        Useful for cross-round adaptation.
        """
        params = self.params
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

        return self.set_params(params=FrozenDict(new_params))

    def _initialize_params(
        self,
        key: KeyArray,
    ) -> PyTreeNode:
        _x = jnp.ones((self.event_shape[0] + self.conditioned_event_shape[0],))
        params = MLP(self.config.width, self.config.depth).init(key, _x)["params"]
        return params

    def set_params(self, params: PyTreeNode) -> Self:
        new_self = copy.copy(self)
        new_self.params = params
        return new_self

    def _set_needs_calibration_correction_if_nans(self, v: bool) -> Self:
        new_self = copy.copy(self)
        new_self.needs_filtering_correction_if_nans = v
        return new_self

    def set_log_z_net(self, log_z_net: Optional[LogZNet]) -> Self:
        if self.log_z_net is None:
            # log-z net setters should not be used on
            # automatically-normalized models
            assert self.has_theta_dependent_normalizer

        if log_z_net is None:
            new_self = self._set_has_theta_dependent_normalizer(True)
        else:
            new_self = self._set_has_theta_dependent_normalizer(False)
        new_self.log_z_net = log_z_net
        return new_self

    def train(
        self,
        parameters: Array,
        observations: Array,
        proposal: Optional[np_distributions.Distribution],
        key: KeyArray,
        ebm_model_type: Literal["joint_tilted", "likelihood"],
        config: TrainingConfig,
    ) -> Tuple[TrainerResults, Self]:
        self = self.set_log_z_net(None)
        if ebm_model_type == "joint_tilted":
            trainer = Trainer(self.conditioned_event_shape[0])
            assert proposal is not None

            def joint_energy_fn(params, x):
                param, obs = (
                    x[: self.conditioned_event_shape[0]],
                    x[self.conditioned_event_shape[0] :],
                )
                return -self.set_params(params).log_prob_override_conditionned(
                    param, obs
                ) - proposal.log_prob(param)

            training_results = trainer.train(
                joint_energy_fn,
                self.params,
                (jnp.concatenate((parameters, observations), axis=1),),
                config,
                key,
            )

        else:

            def conditional_energy_fn(params, theta, x):
                return -self.set_params(params).log_prob_override_conditionned(theta, x)

            trainer = LikelihoodTrainer(self.conditioned_event_shape[0])

            training_results = trainer.train(
                conditional_energy_fn,
                self.params,
                (parameters, observations),
                config,
                key,
            )

        self = self.set_params(training_results.best_state.params)

        if ebm_model_type == "likelihood":
            self = self._set_has_theta_dependent_normalizer(True)
            self = self._set_needs_calibration_correction_if_nans(True)
        else:
            self = self._set_has_theta_dependent_normalizer(False)
            self = self._set_needs_calibration_correction_if_nans(False)

        return training_results, self

    @property
    def param(self):
        return self.condition

    @jittable_method_descriptor
    def log_prob(self, value):
        ret = MLP(self.config.width, self.config.depth).apply(
            {"params": self.params}, jnp.concatenate((self.param, value))
        )
        assert not isinstance(ret, tuple)
        ret = -ret[0]
        ret += self.config.base_measure_log_prob(value)
        ret += self.config.base_measure_log_prob(self.param)
        if self.log_z_net is not None:
            ret -= self.log_z_net(self.param)
        return ret

    def tree_flatten(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        base_params, base_aux_data = super(EBMLikelihood, self).tree_flatten()
        return (
            (
                *base_params,
                self.params,
                self.log_z_net,
                self.config,
            ),
            (*base_aux_data, self.needs_filtering_correction_if_nans),
        )

    def vmap_axes(self, condition):
        # avoid pyright thinking these attributes are optional for the EBMLikelihood
        # because we mark them as None for vmap axes by casting it to Any
        dist_axes = cast(Any, copy.copy(self))
        dist_axes.params = None
        dist_axes.log_z_net = None
        dist_axes.config = None
        dist_axes = dist_axes.set_condition(condition)
        return dist_axes

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        (*base_aux_data, needs_calibration_correction_if_nans) = aux_data
        (*base_params, params, log_z_net, config) = params
        obj = super(EBMLikelihood, cls).tree_unflatten(base_aux_data, base_params)
        obj.params = params
        obj.config = config
        obj.log_z_net = log_z_net
        obj.needs_filtering_correction_if_nans = needs_calibration_correction_if_nans
        return obj
