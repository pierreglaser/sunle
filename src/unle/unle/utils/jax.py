from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

from flax import struct
from typing_extensions import ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


class JittableMethod(struct.PyTreeNode, Generic[P, T]):
    method: Callable[P, T] = struct.field(pytree_node=False)
    obj: Any

    def __call__(self, *args: P.args, **kwargs: P.kwargs):
        return self.method(self.obj, *args, **kwargs)  # type: ignore

    def vmap_axes(self, obj: Any):
        return self.replace(obj=obj)


class JittableMethodDescriptor(Generic[P, T]):
    def __init__(self, method: Callable[P, T]):
        self.method = method

    def __get__(self, obj, objtype=None):
        return JittableMethod(self.method, obj)


def jittable_method_descriptor(method):
    return JittableMethodDescriptor(method)
