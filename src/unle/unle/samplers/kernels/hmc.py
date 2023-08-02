from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Type

if TYPE_CHECKING:
    from jax.random import KeyArray
    from typing_extensions import Self
    from unle.samplers.pytypes import LogDensity_T, Numeric

import jax.numpy as jnp
from blackjax.mcmc.hmc import HMCState as BHMCState
from blackjax.mcmc.hmc import init as hmc_init
from blackjax.mcmc.hmc import kernel as hmc_kernel
from flax import struct
from unle.samplers.kernels.base import (
    Array_T,
    Info,
    Result,
    State,
    TunableConfig,
    TunableKernel,
    TunableMHKernelFactory,
)


class HMCConfig(TunableConfig):
    step_size: Array_T = struct.field(pytree_node=True, default=0.1)
    inverse_mass_matrix: Optional[Array_T] = struct.field(
        pytree_node=True, default=None
    )
    num_integration_steps: int = struct.field(pytree_node=False, default=5)

    @property
    def supports_diagonal_mass(self) -> bool:
        return True

    def get_step_size(self) -> Numeric:
        return self.step_size

    def get_inverse_mass_matrix(self) -> Optional[Array_T]:
        return self.inverse_mass_matrix

    def set_step_size(self, step_size: Array_T) -> Self:
        return self.replace(step_size=step_size)

    def _set_dense_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        return self.replace(inverse_mass_matrix=inverse_mass_matrix)

    def _set_diag_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        return self.replace(inverse_mass_matrix=inverse_mass_matrix)


class HMCInfo(Info):
    accept: Numeric
    log_alpha: Numeric


class HMCState(State):
    blackjax_state: BHMCState


class HMCKernel(TunableKernel[HMCConfig, HMCState, HMCInfo]):
    _kernel_fun = staticmethod(hmc_kernel())

    @classmethod
    def create(
        cls: Type[Self], target_log_prob: LogDensity_T, config: HMCConfig
    ) -> Self:
        return cls(target_log_prob=target_log_prob, config=config)

    def sample_from_proposal(self, key: KeyArray, x: HMCState) -> HMCState:
        raise NotImplementedError

    def _build_info(self, accept: Numeric, log_alpha: Numeric) -> HMCInfo:
        return HMCInfo(accept=accept, log_alpha=log_alpha)

    def _compute_accept_prob(self, proposal: HMCState, x: HMCState) -> Numeric:
        raise NotImplementedError

    def _set_dense_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        return self.replace(
            config=self.config.replace(inverse_mass_matrix=inverse_mass_matrix)
        )

    def _set_diag_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        return self.replace(
            config=self.config.replace(inverse_mass_matrix=inverse_mass_matrix)
        )

    def get_inverse_mass_matrix(self) -> Array_T:
        return self.config.inverse_mass_matrix

    def get_step_size(self) -> Numeric:
        return self.config.step_size

    def set_step_size(self, step_size: Array_T) -> Self:
        return self.replace(config=self.config.replace(step_size=step_size))

    def init_state(self, x: Array_T) -> HMCState:
        blackjax_state = hmc_init(x, self.target_log_prob)

        return HMCState(blackjax_state.position, blackjax_state)

    def one_step(
        self,
        x: HMCState,
        key: KeyArray,
    ) -> Result[HMCState, HMCInfo]:
        if self.config.inverse_mass_matrix is None:
            inverse_mass_matrix = jnp.ones(x.x.shape[-1])
        else:
            inverse_mass_matrix = self.config.inverse_mass_matrix
        _new_state, _new_info = self._kernel_fun(
            rng_key=key,
            state=x.blackjax_state,
            logprob_fn=self.target_log_prob,
            inverse_mass_matrix=inverse_mass_matrix,
            step_size=self.config.step_size,
            num_integration_steps=self.config.num_integration_steps,
        )

        ret = Result(
            HMCState(_new_state.position, _new_state),
            self._build_info(
                _new_info.is_accepted, jnp.log(_new_info.acceptance_probability)
            ),
        )
        return ret


class HMCKernelFactory(TunableMHKernelFactory[HMCConfig, HMCState, HMCInfo]):
    kernel_cls: Type[HMCKernel] = struct.field(pytree_node=False, default=HMCKernel)

    def build_kernel(self, log_prob: LogDensity_T) -> HMCKernel:
        return self.kernel_cls.create(log_prob, self.config)
