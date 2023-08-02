from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Type

if TYPE_CHECKING:
    from blackjax.mcmc.hmc import HMCState as BNUTSState
    from jax.random import KeyArray
    from typing_extensions import Self
    from unle.samplers.pytypes import Array, LogDensity_T, Numeric

import jax.numpy as jnp
from blackjax.mcmc.nuts import init as nuts_init
from blackjax.mcmc.nuts import kernel as nuts_kernel
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


class NUTSConfig(TunableConfig):
    # staying conservative w.r.t
    step_size: Array_T = struct.field(pytree_node=True, default=0.1)
    inverse_mass_matrix: Optional[Array_T] = struct.field(
        pytree_node=True, default=None
    )

    @property
    def supports_diagonal_mass(self) -> bool:
        return True

    def get_step_size(self) -> Array_T:
        return self.step_size

    def get_inverse_mass_matrix(self) -> Array_T:
        return self.inverse_mass_matrix

    def set_step_size(self, step_size: Array_T) -> Self:
        return self.replace(step_size=step_size)

    def _set_dense_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        return self.replace(inverse_mass_matrix=inverse_mass_matrix)

    def _set_diag_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        return self.replace(inverse_mass_matrix=inverse_mass_matrix)


class NUTSInfo(Info):
    accept: Numeric
    log_alpha: Numeric


class NUTSState(State):
    blackjax_state: BNUTSState


class NUTSKernel(TunableKernel[NUTSConfig, NUTSState, NUTSInfo]):
    _kernel_fun = staticmethod(nuts_kernel())

    @classmethod
    def create(
        cls: Type[Self], target_log_prob: LogDensity_T, config: NUTSConfig
    ) -> Self:
        return cls(target_log_prob=target_log_prob, config=config)

    def sample_from_proposal(self, key: KeyArray, x: NUTSState) -> NUTSState:
        raise NotImplementedError

    def _build_info(self, accept: Numeric, log_alpha: Numeric) -> NUTSInfo:
        return NUTSInfo(accept=accept, log_alpha=log_alpha)

    def _compute_accept_prob(self, proposal: NUTSState, x: NUTSState) -> Numeric:
        raise NotImplementedError

    def set_step_size(self, step_size: Array_T) -> Self:
        return self.replace(config=self.config.replace(step_size=step_size))

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

    def init_state(self, x: Array) -> NUTSState:
        blackjax_state = nuts_init(x, self.target_log_prob)
        return NUTSState(blackjax_state.position, blackjax_state)

    def one_step(
        self,
        x: NUTSState,
        key: KeyArray,
    ) -> Result[NUTSState, NUTSInfo]:
        if self.config.inverse_mass_matrix is None:
            inverse_mass_matrix = jnp.ones(x.x.shape[-1])
        else:
            inverse_mass_matrix = self.config.inverse_mass_matrix
        _new_state, _new_info = self._kernel_fun(
            rng_key=key,
            state=x.blackjax_state,
            logprob_fn=self.target_log_prob,
            step_size=self.config.step_size,
            inverse_mass_matrix=inverse_mass_matrix,
        )

        ret = Result(
            NUTSState(_new_state.position, _new_state),
            # accept does not make sense for nuts
            self._build_info(
                jnp.log(_new_info.acceptance_probability),
                jnp.log(_new_info.acceptance_probability),
            ),
        )
        return ret


class NUTSKernelFactory(TunableMHKernelFactory[NUTSConfig, NUTSState, NUTSInfo]):
    kernel_cls: Type[NUTSKernel] = struct.field(pytree_node=False, default=NUTSKernel)

    def build_kernel(self, log_prob: LogDensity_T) -> NUTSKernel:
        return self.kernel_cls.create(log_prob, self.config)
