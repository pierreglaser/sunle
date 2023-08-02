from __future__ import annotations

from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from jax.random import KeyArray
    from typing_extensions import Self
    from unle.samplers.pytypes import LogDensity_T

import jax.numpy as jnp
from flax import struct
from jax import grad, random

from .base import Array_T, Info, Kernel, KernelConfig, KernelFactory, Result, State


class ULAState(State):
    pass


class ULAInfo(Info):
    pass


class ULAConfig(KernelConfig, struct.PyTreeNode):
    step_size: float = struct.field(pytree_node=True)


class ULAKernel(Kernel[ULAConfig, ULAState, ULAInfo]):
    @classmethod
    def create(
        cls: Type[Self], target_log_prob: LogDensity_T, config: ULAConfig
    ) -> Self:
        return cls(target_log_prob=target_log_prob, config=config)

    def init_state(self, x: Array_T) -> ULAState:
        return ULAState(x=x)

    def one_step(self, x: ULAState, key: KeyArray) -> Result[ULAState, ULAInfo]:
        noise = random.normal(key, x.x.shape)
        new_pos = (
            x.x
            + self.config.step_size * grad(self.target_log_prob)(x.x)
            + jnp.sqrt(2 * self.config.step_size) * noise
        )
        return Result(ULAState(x=new_pos), ULAInfo())


class ULAKernelFactory(KernelFactory[ULAConfig, ULAState, ULAInfo]):
    kernel_cls: Type[ULAKernel] = struct.field(pytree_node=False, default=ULAKernel)
