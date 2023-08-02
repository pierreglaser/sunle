from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Union

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

from jax import Array

Scalar: TypeAlias = Union[float, int]

Numeric = Union[Array, Scalar]

PyTreeNode: TypeAlias = Any

Simulator_T = Callable[[Array], Array]
