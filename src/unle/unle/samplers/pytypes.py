from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol, Union

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

from jax import Array

Scalar = Union[float, int]

Numeric = Union[Array, Scalar]

PyTreeNode: TypeAlias = Any

LogLikelihood_T = Callable[[Array, Array], Numeric]

GradLogDensity = Callable[[Array], Numeric]


class LogDensity_T(Protocol):
    def __call__(self, x: Array, /) -> Numeric:
        ...


class LogJoint_T(Protocol):
    def __call__(self, theta: Array, x: Array) -> Numeric:
        ...
