from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Optional, Tuple, Type, TypeVar

if TYPE_CHECKING:
    from jax.random import KeyArray
    from numpyro import distributions as np_distributions
    from typing_extensions import Self

import abc

from flax import struct
from unle.samplers.distributions import LogDensity_T

from ..particle_aproximation import ParticleApproximation


class InferenceAlgorithmConfig(struct.PyTreeNode):
    num_samples: int = struct.field(pytree_node=False)


IAC_T = TypeVar("IAC_T", bound=InferenceAlgorithmConfig)


class InferenceAlgorithmInfo(struct.PyTreeNode):
    pass


PA_T = TypeVar("PA_T", bound=ParticleApproximation)


PA_T_co = TypeVar("PA_T_co", bound=ParticleApproximation, covariant=True)


class InferenceAlgorithmResults(struct.PyTreeNode):
    samples: ParticleApproximation
    info: InferenceAlgorithmInfo


LD_T = TypeVar("LD_T", bound=LogDensity_T)


class InferenceAlgorithm(Generic[IAC_T], struct.PyTreeNode, metaclass=abc.ABCMeta):
    config: IAC_T
    log_prob: LogDensity_T
    init_state: Optional[ParticleApproximation] = None

    @property
    @abc.abstractmethod
    def initialized(self):
        raise NotImplementedError

    @abc.abstractmethod
    def init(self, key: KeyArray, dist: np_distributions.Distribution) -> Self:
        raise NotImplementedError

    @abc.abstractmethod
    def run(self, key: KeyArray) -> Tuple[Self, InferenceAlgorithmResults]:
        raise NotImplementedError

    @abc.abstractmethod
    def set_log_prob(self, log_prob: LogDensity_T) -> Self:
        raise NotImplementedError

    # this should be a class attribute but is made a property to avoid conflating with
    # the dataclass __init__ arguments.
    @property
    @abc.abstractmethod
    def can_set_num_samples(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def set_num_samples(self, num_samples: int) -> Self:
        raise NotImplementedError

    def run_and_update_init(
        self, key: KeyArray
    ) -> Tuple[Self, InferenceAlgorithmResults]:
        self, results = self.run(key)
        self = self.replace(init_state=results.samples)
        return self, results


class InferenceAlgorithmFactory(
    Generic[IAC_T], struct.PyTreeNode, metaclass=abc.ABCMeta
):
    config: IAC_T

    @abc.abstractmethod
    def build_algorithm(self, log_prob: LogDensity_T) -> InferenceAlgorithm[IAC_T]:
        raise NotImplementedError

    # this should be a class attribute but is made a property to avoid conflating with
    # the dataclass __init__ arguments.
    @property
    @abc.abstractmethod
    def inference_alg_cls(self) -> Type[InferenceAlgorithm[IAC_T]]:
        raise NotImplementedError
