from __future__ import annotations

from typing import Any, Callable

from flax import struct
from jax import Array
from typing_extensions import Self

from unle.samplers.pytypes import (
    LogDensity_T,
    LogJoint_T,
    LogLikelihood_T,
    Numeric,
)


class LogDensityNode(struct.PyTreeNode):
    log_prob: Callable = struct.field(pytree_node=False)

    def __call__(self, x: Array) -> Numeric:
        return self.log_prob(x)


class LogJointNode(struct.PyTreeNode):
    log_prob: Callable = struct.field(pytree_node=False)

    def __call__(self, theta: Array, x: Array) -> Numeric:
        return self.log_prob(theta, x)


def maybe_wrap(log_prob: LogDensity_T) -> LogDensity_T:
    return LogDensityNode(log_prob)


def maybe_wrap_joint(log_prob: LogJoint_T) -> LogJoint_T:
    if isinstance(log_prob, struct.PyTreeNode):
        return log_prob
    return LogJointNode(log_prob)


class LogLikelihoodNode(struct.PyTreeNode):
    log_likelihood: LogLikelihood_T = struct.field(pytree_node=False)

    def __call__(self, theta: Array, x: Array) -> Numeric:
        return self.log_likelihood(theta, x)


def maybe_wrap_log_l(log_prob: LogLikelihood_T) -> LogLikelihood_T:
    return LogLikelihoodNode(log_prob)


class DoublyIntractableJointLogDensity(struct.PyTreeNode):
    log_prior: LogDensity_T
    log_likelihood: LogLikelihood_T
    dim_param: int = struct.field(pytree_node=False)
    expose_tilted_log_joint: bool = struct.field(pytree_node=False, default=True)

    def tilted_log_joint(self, x: Array) -> Numeric:
        # because this function discards the likelihood log-normalizer, the
        # output of this function corresponds to a tilted model where the
        # likelihood corresponds to the true (self.)log-likelihood, but the
        # prior does not match sels.prior.
        theta, x = x[..., : self.dim_param], x[..., self.dim_param :]
        return self.log_prior(theta) + self.log_likelihood(theta, x)

    def __call__(self, x: Array) -> Numeric:
        return self.tilted_log_joint(x)

    def pin(self, x):
        return DoublyIntractableLogDensity(
            log_prior=self.log_prior, log_likelihood=self.log_likelihood, x_obs=x
        )


class DoublyIntractableLogDensity(struct.PyTreeNode):
    log_prior: LogDensity_T
    log_likelihood: LogLikelihood_T
    x_obs: Array

    def __call__(self, x: Array, /) -> Numeric:
        # XXX: this is confusing. Inputs to __call__ log density should have a
        # generic name like "val", and not x.
        theta = x
        return self.log_prior(theta) + self.log_likelihood(theta, self.x_obs)


# wrapper for conditionned likelihood objects
class ThetaConditionalLogDensity(struct.PyTreeNode, LogDensity_T):
    log_prob: LogLikelihood_T
    theta: Array

    def __call__(self, x: Array) -> Numeric:
        return self.log_prob(self.theta, x)

    @classmethod
    def vmap_axes(cls, obj: Self, log_prob: Any, theta: int):
        if obj is None:
            return ThetaConditionalLogDensity(
                log_prob=log_prob,
                theta=theta,  # pyright: ignore [reportGeneralTypeIssues]
            )
        else:
            return obj.replace(log_prob=log_prob, theta=theta)
