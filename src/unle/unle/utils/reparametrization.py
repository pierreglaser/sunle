from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import jax.numpy as jnp
from numpyro.distributions.transforms import AffineTransform


def compose_affine_transforms(t1: AffineTransform, t2: AffineTransform):
    return AffineTransform(scale=t1.scale * t2.scale, loc=t1.scale * t2.loc + t1.loc)


def compose_dense_and_componentwise_transform(
    outer_scale, outer_bias, inner_scale, inner_bias
):
    return (
        jnp.diag(inner_scale) @ outer_scale,
        inner_bias @ outer_scale + outer_bias,
    )
