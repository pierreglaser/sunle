from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from jax import Array
    from unle.typing import Numeric

import jax.numpy as jnp
from flax import struct
from numpyro.distributions.transforms import AffineTransform


def find_outliers(
    values: Array,
    loc: Numeric,
    std: Numeric,
    n_std: float,
    norm: Literal["l2", "l_inf"] = "l2",
) -> Array:
    r"""
    Determine elementwise whether:

    .. math::
        \|\text{values} - \text{loc}\|_{k} > n_{\textrm{std}} * \text{std}

    where $k$ is either `2` (`"l2"`) or :math:`\infty` (`"l_inf"`).

    Notes
    -----

    `loc` and `std` need not be the mean and std of `values`:
    for instance, values can be observations/parameters drawn from
    multiple simulation rounds, while `loc` and `std` can be the mean
    and std of the last simulation round of observation/parameters.
    """
    if norm == "l2":
        is_outlier_mask = jnp.linalg.norm(
            values - loc, axis=1
        ) > n_std * jnp.linalg.norm(std)
    elif norm == "l_inf":
        is_outlier_mask = jnp.any(
            jnp.abs(values - loc) > n_std * std,
            axis=1,
        )
    else:
        raise ValueError(f"Unknown norm {norm}")
    return is_outlier_mask


def find_nans(values):
    return jnp.sum(jnp.isnan(values), axis=1) > 0


class Normalizer(struct.PyTreeNode):
    params_mean: Array
    params_std: Array
    observations_mean: Array
    observations_std: Array

    @classmethod
    def create_and_fit(cls, parameters: Array, observations: Array):
        params_mean = jnp.mean(parameters[~find_nans(parameters)], axis=0)
        params_std = jnp.std(parameters[~find_nans(parameters)], axis=0) + 1e-8

        observations_mean = jnp.mean(observations[~find_nans(observations)], axis=0)
        observations_std = (
            jnp.std(observations[~find_nans(observations)], axis=0) + 1e-8
        )
        return cls(params_mean, params_std, observations_mean, observations_std)

    def get_transform(self, who: Literal["params", "observations", "both"]):
        if who == "params":
            mean, std = self.params_mean, self.params_std
        elif who == "observations":
            mean, std = self.observations_mean, self.observations_std
        elif who == "both":
            mean, std = (
                jnp.concatenate([self.params_mean, self.observations_mean]),
                jnp.concatenate([self.params_std, self.observations_std]),
            )
        else:
            raise ValueError
        return AffineTransform(loc=-mean / std, scale=1 / std)

    def get_inverse_transform(self, who: Literal["params", "observations", "both"]):
        t = self.get_transform(who)
        return AffineTransform(scale=1 / t.scale, loc=-t.loc / t.scale)
