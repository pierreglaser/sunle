from __future__ import annotations

from typing import TYPE_CHECKING, Generic, NamedTuple, Optional, Tuple, Type

if TYPE_CHECKING:
    from jax.random import KeyArray
    from typing_extensions import Self
    from unle.samplers.kernels.rwmh import RWKernelFactory, RWState
    from unle.samplers.pytypes import Array, LogDensity_T, Numeric

import jax.numpy as jnp
from flax import struct
from jax import random, vmap
from jax.nn import softmax
from unle.samplers.distributions import (
    DoublyIntractableLogDensity,
    ThetaConditionalLogDensity,
)
from unle.samplers.kernels.base import (
    Array_T,
    Config_T,
    Info,
    Info_T,
    State,
    State_T,
    TunableConfig,
    TunableKernel,
    TunableMHKernelFactory,
)


class DiscretizingSampler(struct.PyTreeNode):
    log_prob: LogDensity_T
    bounds: Tuple[Tuple[int, int], Tuple[int, int]] = ((-10, 10), (-10, 10))
    nbins: int = 100

    def sample(self, key: KeyArray) -> Array_T:
        (x_min, x_max), (y_min, y_max) = self.bounds

        num_total_points = self.nbins ** len(self.bounds)

        _X, _Y = jnp.meshgrid(
            jnp.linspace(x_min, x_max, self.nbins),
            jnp.linspace(y_min, y_max, self.nbins),
            indexing="ij",
        )
        _inputs = jnp.stack((_X, _Y), axis=-1).reshape(num_total_points, 2)
        conditioned_log_density_vals = vmap(self.log_prob)(_inputs)

        key, subkey = random.split(key)
        idx = random.choice(
            subkey,
            len(conditioned_log_density_vals),  # type: ignore
            p=softmax(conditioned_log_density_vals),
        )
        return _inputs[idx]


class ExactSAVMConfig(Generic[Config_T, State_T, Info_T], TunableConfig):
    base_var_kernel_factory: RWKernelFactory


class ExactSAVMInfo(Generic[Info_T], Info):
    accept: Numeric
    log_alpha: Numeric


class ExactSAVMState(Generic[Config_T, State_T, Info_T], State):
    base_var_state: RWState = struct.field(pytree_node=True)
    aux_var: Array


class ExactSAVMResult(NamedTuple):
    x: ExactSAVMState
    accept_freq: Numeric


class ExactSAVMKernel(
    TunableKernel[ExactSAVMConfig, ExactSAVMState, ExactSAVMInfo],
    Generic[Config_T, State_T, Info_T],
):
    target_log_prob: DoublyIntractableLogDensity
    config: ExactSAVMConfig[Config_T, State_T, Info_T]

    @property
    def base_var_kernel(self):
        return self.config.base_var_kernel_factory.build_kernel(self.target_log_prob)

    def get_step_size(self) -> Numeric:
        return self.config.base_var_kernel_factory.config.step_size

    def get_inverse_mass_matrix(self) -> Numeric:
        C = self.config.base_var_kernel_factory.config.C
        assert C is not None
        return C

    def set_step_size(self, step_size) -> Self:
        return self.replace(
            config=self.config.replace(
                base_var_kernel_factory=self.config.base_var_kernel_factory.replace(
                    config=self.base_var_kernel.set_step_size(step_size).config
                )
            )
        )

    def set_inverse_mass_matrix(self, inverse_mass_matrix) -> Self:
        return self.replace(
            config=self.config.replace(
                base_var_kernel_factory=self.config.base_var_kernel_factory.replace(
                    config=self.base_var_kernel.set_inverse_mass_matrix(
                        inverse_mass_matrix
                    ).config
                )
            )
        )

    @classmethod
    def create(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls: Type[Self],
        target_log_prob: DoublyIntractableLogDensity,
        config: ExactSAVMConfig[Config_T, State_T, Info_T],
    ) -> Self:
        return cls(target_log_prob, config)

    def init_state(
        self: Self, x: Array_T, aux_var0: Optional[Array_T] = None
    ) -> ExactSAVMState[Config_T, State_T, Info_T]:
        assert len(self.target_log_prob.x_obs.shape) == 1

        aux_var = self.target_log_prob.x_obs

        # x: theta
        base_var_state = self.base_var_kernel.init_state(x)
        return ExactSAVMState(base_var_state.x, base_var_state, aux_var)

    def _build_info(self, accept: Numeric, log_alpha: Numeric) -> ExactSAVMInfo[Info_T]:
        return ExactSAVMInfo(accept, log_alpha)

    def sample_from_proposal(
        self, key: KeyArray, x: ExactSAVMState[Config_T, State_T, Info_T]
    ) -> ExactSAVMState:
        key, key_base_var, key_aux_var = random.split(key, num=3)

        # first, sample base variable
        new_base_var_state = self.base_var_kernel.sample_from_proposal(
            key_base_var, x.base_var_state
        )

        this_iter_log_l = ThetaConditionalLogDensity(
            self.target_log_prob.log_likelihood, new_base_var_state.x
        )

        key, subkey = random.split(key)
        new_x = DiscretizingSampler(this_iter_log_l).sample(subkey)

        return x.replace(
            x=new_base_var_state.x, base_var_state=new_base_var_state, aux_var=new_x
        )

    def _compute_accept_prob(
        self,
        proposal: ExactSAVMState[Config_T, State_T, Info_T],
        x: ExactSAVMState[Config_T, State_T, Info_T],
    ) -> Numeric:
        """Compute α = min(1, (p(xᵢ₊₁)q(xᵢ | xᵢ₊₁)) / (p(xᵢ) q(xᵢ₊₁ | xᵢ)))"""
        # orig_x = theta
        q_theta = self.base_var_kernel.get_proposal()
        log_q_new_given_prev = q_theta.log_prob(
            x=proposal.base_var_state.x, x_cond=x.base_var_state.x
        )
        log_q_prev_given_new = q_theta.log_prob(
            x=x.base_var_state.x, x_cond=proposal.base_var_state.x
        )

        log_alpha = (
            self.target_log_prob(proposal.base_var_state.x)
            + log_q_prev_given_new
            - self.target_log_prob(x.base_var_state.x)
            - log_q_new_given_prev
            + self.target_log_prob.log_likelihood(x.base_var_state.x, proposal.aux_var)
            - self.target_log_prob.log_likelihood(
                proposal.base_var_state.x, proposal.aux_var
            )
        )
        log_alpha = jnp.nan_to_num(log_alpha, nan=-50, neginf=-50, posinf=0)

        return log_alpha


class ExactSAVMKernelFactory(
    TunableMHKernelFactory[
        ExactSAVMConfig[Config_T, State_T, Info_T],
        ExactSAVMState[Config_T, State_T, Info_T],
        ExactSAVMInfo[Info_T],
    ]
):
    kernel_cls: Type[ExactSAVMKernel] = struct.field(
        pytree_node=False, default=ExactSAVMKernel
    )

    def build_kernel(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, log_prob: DoublyIntractableLogDensity
    ) -> ExactSAVMKernel:
        return self.kernel_cls.create(log_prob, self.config)
