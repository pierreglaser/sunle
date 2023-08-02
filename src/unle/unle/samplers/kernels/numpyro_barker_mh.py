from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Type, cast

if TYPE_CHECKING:
    from jax.random import KeyArray
    from typing_extensions import Self
    from unle.samplers.pytypes import LogDensity_T, Numeric

import jax.numpy as jnp
from flax import struct
from jax import random
from numpyro.infer import BarkerMH
from numpyro.infer.barker import BarkerMHState as np_BarkerMHState
from numpyro.infer.hmc_util import HMCAdaptState, warmup_adapter
from unle.samplers.kernels.base import (
    Array_T,
    Info,
    Result,
    State,
    TunableConfig,
    TunableKernel,
    TunableMHKernelFactory,
)


class BarkerMHConfig(TunableConfig):
    step_size: Array_T = struct.field(pytree_node=True, default=0.1)
    C: Optional[Array_T] = struct.field(pytree_node=True, default=None)

    @property
    def supports_diagonal_mass(self) -> bool:
        return True

    def get_step_size(self) -> Array_T:
        return self.step_size

    def get_inverse_mass_matrix(self) -> Optional[Array_T]:
        return self.C

    def set_step_size(self, step_size: Array_T) -> Self:
        return self.replace(step_size=step_size)

    def _set_dense_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        return self.replace(C=inverse_mass_matrix)

    def _set_diag_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        return self.replace(C=inverse_mass_matrix)


class BarkerMHInfo(Info):
    accept: Numeric
    log_alpha: Numeric
    accept_prob: Numeric


class BarkerMHState(State):
    numpyro_state: np_BarkerMHState


class BarkerMHKernel(TunableKernel[BarkerMHConfig, BarkerMHState, BarkerMHInfo]):
    @classmethod
    def create(
        cls: Type[Self], target_log_prob: LogDensity_T, config: BarkerMHConfig
    ) -> Self:
        return cls(target_log_prob, config)

    def sample_from_proposal(self, key: KeyArray, x: BarkerMHState) -> BarkerMHState:
        raise NotImplementedError

    def _build_info(self, accept: Numeric, log_alpha: Numeric) -> BarkerMHInfo:
        return BarkerMHInfo(
            accept=accept, log_alpha=log_alpha, accept_prob=jnp.exp(log_alpha)
        )

    def _compute_accept_prob(
        self, proposal: BarkerMHState, x: BarkerMHState
    ) -> Numeric:
        raise NotImplementedError

    def get_step_size(self) -> Array_T:
        return self.config.step_size

    def get_inverse_mass_matrix(self) -> Array_T:
        return self.config.C

    def set_step_size(self, step_size: Array_T) -> Self:
        return self.replace(config=self.config.replace(step_size=step_size))

    def _set_dense_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        return self.replace(config=self.config.replace(C=inverse_mass_matrix))

    def _set_diag_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        return self.replace(config=self.config.replace(C=inverse_mass_matrix))

    def init_state(self, x: Array_T) -> BarkerMHState:
        kernel = BarkerMH(
            potential_fn=lambda x: -self.target_log_prob(x),
            adapt_mass_matrix=False,
            adapt_step_size=False,
            dense_mass=self.config.C is not None and len(self.config.C.shape) == 2,
            step_size=0.0001,
        )
        np_state = cast(
            np_BarkerMHState,
            kernel.init(
                random.PRNGKey(0),
                num_warmup=0,
                init_params=x,
                model_args=(),
                model_kwargs={},
            ),
        )
        np_state = np_state._replace(
            adapt_state=cast(HMCAdaptState, np_state.adapt_state)._replace(
                step_size=self.config.step_size
            )
        )

        wa_init, _ = warmup_adapter(
            0,
            adapt_step_size=False,
            adapt_mass_matrix=False,
            dense_mass=self.config.C is not None and len(self.config.C.shape) == 2,
            target_accept_prob=0.5,
        )
        if self.config.C is not None:
            wa_state = wa_init(
                (x,),
                random.PRNGKey(0),
                self.config.step_size,
                inverse_mass_matrix=self.config.C,
            )
        else:
            wa_state = wa_init(
                (x,), random.PRNGKey(0), self.config.step_size, mass_matrix_size=len(x)
            )

        np_state = np_state._replace(adapt_state=wa_state)
        return BarkerMHState(x=x, numpyro_state=np_state)

    def one_step(
        self, x: BarkerMHState, key: KeyArray
    ) -> Result[BarkerMHState, BarkerMHInfo]:
        key, subkey = random.split(key)
        np_state = self.init_state(x.x).numpyro_state._replace(rng_key=subkey)

        key, subkey = random.split(key)
        np_state = np_state._replace(
            adapt_state=cast(HMCAdaptState, np_state.adapt_state)._replace(
                rng_key=subkey
            )
        )

        kernel = BarkerMH(
            potential_fn=lambda x: -self.target_log_prob(x),
            adapt_mass_matrix=False,
            adapt_step_size=False,
            dense_mass=self.config.C is not None and len(self.config.C.shape) == 2,
            step_size=0.0001,
        )
        _ = cast(
            np_BarkerMHState,
            kernel.init(
                random.PRNGKey(0),
                num_warmup=0,
                init_params=x.x,
                model_args=(),
                model_kwargs={},
            ),
        )
        new_np_state = cast(
            np_BarkerMHState, kernel.sample(np_state, model_args=(), model_kwargs={})
        )

        accept = jnp.sum(jnp.abs(new_np_state.z - x.numpyro_state.z)) > 1e-13
        info = BarkerMHInfo(
            accept=accept,
            log_alpha=jnp.log(
                jnp.clip(new_np_state.accept_prob, a_min=1e-20, a_max=1.0)
            ),
            accept_prob=new_np_state.accept_prob,
        )
        return Result(
            BarkerMHState(x=new_np_state.z, numpyro_state=new_np_state), info=info
        )


class BarkerMHKernelFactory(
    TunableMHKernelFactory[BarkerMHConfig, BarkerMHState, BarkerMHInfo]
):
    kernel_cls: Type[BarkerMHKernel] = struct.field(
        pytree_node=False, default=BarkerMHKernel
    )

    def build_kernel(self, log_prob: LogDensity_T) -> BarkerMHKernel:
        return self.kernel_cls.create(log_prob, self.config)
