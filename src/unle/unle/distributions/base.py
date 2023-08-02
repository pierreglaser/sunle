from __future__ import annotations

import copy
from typing import Generic, List, Optional, Tuple, Type, TypeVar

import jax.numpy as jnp
import numpyro.distributions as np_distributions
from jax import Array
from numpyro.distributions import Distribution, TransformedDistribution
from numpyro.distributions.transforms import IdentityTransform, Transform
from unle.samplers.distributions import (
    DoublyIntractableJointLogDensity,
    DoublyIntractableLogDensity,
    maybe_wrap,
)
from unle.utils.jax import jittable_method_descriptor


class BlockDistribution(np_distributions.Distribution):
    arg_constraints = {"distributions": None}

    def __init__(self, distributions: List[np_distributions.Distribution]):
        self.distributions = distributions
        for dist in distributions:
            assert dist.batch_shape == ()
            assert len(dist.event_shape) == 1

        individual_event_shapes = [dist.event_shape for dist in distributions]
        self._individual_event_shapes = individual_event_shapes
        self._event_shape_bounds = [0] + list(
            jnp.cumsum(jnp.array([es[0] for es in self._individual_event_shapes]))
        )

        super(BlockDistribution, self).__init__(
            batch_shape=(), event_shape=(sum(es[0] for es in individual_event_shapes),)
        )

    def sample(self, key, sample_shape=()):
        samples = []
        for dist in self.distributions:
            samples.append(dist.sample(key, sample_shape))
        return jnp.concatenate(samples, axis=-1)

    def log_prob(self, value):
        log_probs = []
        for i, dist in enumerate(self.distributions):
            log_probs.append(
                dist.log_prob(
                    value[
                        ...,
                        self._event_shape_bounds[i] : self._event_shape_bounds[i + 1],
                    ]
                )
            )
        log_probs = jnp.concatenate(jnp.atleast_1d(*log_probs), axis=-1)
        return jnp.sum(log_probs, axis=-1)


D = TypeVar("D", bound="Distribution")


def np_distribution_unflatten(cls: Type[D], aux, params) -> D:
    event_shape, batch_shape = aux
    obj = cls.__new__(cls)
    obj._event_shape = event_shape  # pyright: ignore[reportPrivateUsage]
    obj._batch_shape = batch_shape  # pyright: ignore[reportPrivateUsage]
    return obj


def _np_distribution_flatten(obj):
    return ((), (obj._event_shape, obj._batch_shape))


class ConditionalDistributionBase(np_distributions.Distribution):
    def __init__(
        self,
        batch_shape,
        event_shape,
        conditioned_event_shape: Optional[Tuple[int]] = None,
        condition: Optional[Array] = None,
        has_theta_dependent_normalizer: bool = True,
    ):
        assert batch_shape == ()
        self._has_theta_dependent_normalizer = has_theta_dependent_normalizer
        if condition is None:
            assert conditioned_event_shape is not None
            condition = jnp.zeros(conditioned_event_shape)
        elif conditioned_event_shape is not None:
            assert conditioned_event_shape == condition.shape
        else:
            conditioned_event_shape = condition.shape

        self._condition = condition
        self._conditioned_event_shape = condition.shape
        Distribution.__init__(self, batch_shape=batch_shape, event_shape=event_shape)

    @property
    def conditioned_event_shape(self):
        return self._conditioned_event_shape

    @property
    def has_theta_dependent_normalizer(self):
        return self._has_theta_dependent_normalizer

    def _set_has_theta_dependent_normalizer(self, v: bool):
        new_self = copy.copy(self)
        new_self._has_theta_dependent_normalizer = v
        return new_self

    @property
    def condition(self):
        return self._condition

    def set_condition(self, condition):
        self = copy.copy(self)
        self._condition = condition
        return self

    @jittable_method_descriptor
    def log_prob(self, value):
        raise NotImplementedError

    @jittable_method_descriptor
    def log_prob_override_conditionned(self, condition, value, /):
        return self.set_condition(condition).log_prob(value)

    def tree_flatten(self):  # pyright: ignore [reportIncompatibleMethodOverride]
        _, base_aux = _np_distribution_flatten(self)
        return (self.condition,), (
            *base_aux,
            self.conditioned_event_shape,
            self.has_theta_dependent_normalizer,
        )

    @classmethod
    def tree_unflatten(
        cls, aux_data, params
    ):  # pyright: ignore [reportIncompatibleMethodOverride]
        (condition,) = params
        (*base_aux, conditioned_event_shape, has_theta_dependent_normalizer) = aux_data
        obj = np_distribution_unflatten(cls, base_aux, ())
        obj._conditioned_event_shape = conditioned_event_shape
        obj._has_theta_dependent_normalizer = has_theta_dependent_normalizer
        obj._condition = condition
        return obj

    def vmap_axes(self, condition):
        raise NotImplementedError


CD = TypeVar("CD", bound=ConditionalDistributionBase, covariant=True)


class TransformedConditionalDistribution(
    ConditionalDistributionBase,
    TransformedDistribution,
    Generic[CD],
):
    dist: CD

    def __init__(
        self,
        dist: CD,
        transform: Transform,
        conditioned_var_transform: Transform = IdentityTransform(),
    ):
        self.base_dist = dist
        self.transform = transform
        self.transforms = [transform]
        self.conditioned_var_transform = conditioned_var_transform
        TransformedDistribution.__init__(self, dist, transform)

    @property
    def condition(self):
        return self.base_dist.condition

    def set_condition(self, condition):
        condition = self.conditioned_var_transform.inv(condition)
        new_self = copy.copy(self)
        new_self.base_dist = self.base_dist.set_condition(condition)
        return new_self

    @property
    def conditioned_event_shape(self):
        return self.base_dist.conditioned_event_shape

    @property
    def has_theta_dependent_normalizer(self):
        return self.base_dist.has_theta_dependent_normalizer

    @jittable_method_descriptor
    def log_prob(self, value):  # pyright: ignore [reportIncompatibleMethodOverride]
        return np_distributions.TransformedDistribution.log_prob(
            self, value  # type: ignore
        )

    def tree_flatten(self):  # pyright: ignore [reportIncompatibleMethodOverride]
        return (self.base_dist,), (
            self.transform,
            self.conditioned_var_transform,
            self.event_shape,
            self.batch_shape,
        )

    @classmethod
    def tree_unflatten(
        cls, aux_data, params
    ):  # pyright: ignore [reportIncompatibleMethodOverride]
        (base_dist,) = params
        (transform, conditioned_var_transform, event_shape, batch_shape) = aux_data
        obj = np_distribution_unflatten(cls, (event_shape, batch_shape), ())
        obj.base_dist = base_dist
        obj.transform = transform
        obj.conditioned_var_transform = conditioned_var_transform
        obj.transforms = [transform]
        return obj


class JointDistribution(np_distributions.Distribution):
    marginal: np_distributions.Distribution
    conditional: ConditionalDistributionBase

    def __init__(self, marginal, conditional):
        assert len(marginal.event_shape) == 1
        assert len(conditional.event_shape) == 1
        self._init(marginal, conditional)

    def _init(self, marginal, conditional):
        event_shape = (marginal.event_shape[0] + conditional.event_shape[0],)
        self.marginal = marginal
        self.conditional = conditional
        super(JointDistribution, self).__init__(batch_shape=(), event_shape=event_shape)

    def tree_flatten(self):  # pyright: ignore [reportIncompatibleMethodOverride]
        if isinstance(self.marginal, Posterior):
            return (self.marginal, self.conditional), ()
        else:
            # hotfix for numpyro distributions not being jittable
            return (self.conditional,), (self.marginal,)

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        if len(params) == 1:
            (conditional,) = params
            (marginal,) = aux_data
        else:
            assert len(params) == 2
            marginal, conditional = params
        obj = cls.__new__(cls)
        obj._init(marginal, conditional)
        return obj

    def condition_in_variable(self, value):
        return self.conditional

    def condition_out_variable(self, value):
        return Posterior(self, value)

    @jittable_method_descriptor
    def _log_prob_singly_intractable(self, value):
        in_var_value = value[..., : self.marginal.event_shape[0]]
        out_var_value = value[..., self.marginal.event_shape[0] :]
        return self.marginal.log_prob(
            in_var_value
        ) + self.conditional.log_prob_override_conditionned(in_var_value, out_var_value)

    @property  # pyright: ignore [reportIncompatibleMethodOverride]
    def log_prob(self):
        if self.conditional.has_theta_dependent_normalizer:
            ret = DoublyIntractableJointLogDensity(
                log_prior=maybe_wrap(self.marginal.log_prob),
                log_likelihood=self.conditional.log_prob_override_conditionned,
                dim_param=self.marginal.event_shape[0],
            )
            return ret
        else:
            return self._log_prob_singly_intractable


class Posterior(ConditionalDistributionBase):
    joint_distribution: JointDistribution

    def __init__(self, joint_distribution, condition: Array):
        self.joint_distribution = joint_distribution
        super(Posterior, self).__init__(
            batch_shape=(),
            event_shape=self.joint_distribution.marginal.event_shape,
            condition=jnp.asarray(condition),
            has_theta_dependent_normalizer=True,
        )

    @property  # pyright: ignore [reportIncompatibleMethodOverride]
    def log_prob(self):
        if self.joint_distribution.conditional.has_theta_dependent_normalizer:
            return DoublyIntractableLogDensity(
                log_prior=maybe_wrap(self.joint_distribution.marginal.log_prob),
                log_likelihood=self.joint_distribution.conditional.log_prob_override_conditionned,  # noqa: E501
                x_obs=self.condition,
            )
        else:
            return self._log_prob_singly_intractable

    @jittable_method_descriptor
    def _log_prob_singly_intractable(self, value):
        return self.joint_distribution.log_prob(
            jnp.concatenate([value, jnp.atleast_1d(self.condition)])
        )

    @property
    def likelihood(self):
        return self.joint_distribution.conditional

    @property
    def prior(self):
        return self.joint_distribution.marginal

    def tree_flatten(self):  # pyright: ignore [reportIncompatibleMethodOverride]
        base_data, base_aux = super(Posterior, self).tree_flatten()
        return (*base_data, self.joint_distribution), base_aux

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        base_aux = aux_data
        (*base_data, joint_distribution) = params
        obj = super(Posterior, cls).tree_unflatten(base_aux, base_data)
        obj.joint_distribution = joint_distribution
        return obj


class TransformedPosterior(TransformedConditionalDistribution[Posterior]):
    @property
    def likelihood(self):
        return TransformedConditionalDistribution(
            self.base_dist.likelihood,
            transform=self.conditioned_var_transform,
            conditioned_var_transform=self.transform,
        )

    @property
    def prior(self):
        return TransformedDistribution(
            self.base_dist.prior,
            self.transform,
        )
