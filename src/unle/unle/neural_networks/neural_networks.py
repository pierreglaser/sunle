from typing import Callable

from flax import struct
from flax.linen.linear import Dense
from flax.linen.module import Module
from jax.nn import swish


class MLPConfig(struct.PyTreeNode):
    width: int = 150
    depth: int = 4
    activation: Callable = struct.field(pytree_node=False, default=swish)
    use_bias_last_layer: bool = True
    num_outputs: int = 1


class MLP(Module):
    width: int = 150
    depth: int = 4
    activation: Callable = struct.field(pytree_node=False, default=swish)
    use_bias_last_layer: bool = True
    num_outputs: int = 1

    def setup(self):
        self.layers = [Dense(self.width) for _ in range(self.depth)] + [
            Dense(self.num_outputs, use_bias=self.use_bias_last_layer)
        ]

    def __call__(self, inputs):  # pyright: ignore[reportIncompatibleMethodOverride]
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
        return x
