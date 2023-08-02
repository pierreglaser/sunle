from typing import Type

import jax
import jax.numpy as jnp
import numpy as np
import pyro.distributions as pyro_distributions
import sbibm
import torch
from jax import Array
from numpyro import distributions as np_distributions
from numpyro.distributions import transforms as np_transforms
from sbibm.tasks import Task
from typing_extensions import Self
from unle.typing import Simulator_T

from sbibm_unle_extra.pyro_to_numpyro import convert_dist

from .ldct.task import LDCT
from .multimodal_task.task import MultiModalLikelihoodTask
from .two_moons_with_nans.task import TwoMoonsWithNans
from .base import SimulatorWithPrecomputedDataset

# from sbibm_unle_extra.tasks.ornstein_uhlenbeck.task import OrnsteinUhlenbeck


def get_task(task_name: str) -> Task:
    if task_name == "LDCT":
        return LDCT()
    elif task_name == "two_moons_with_nans":
        return TwoMoonsWithNans()
    elif task_name == "Lorenz96":
        try:
            from sbibm_unle_extra.tasks.lorenz96.task import Lorenz96
        except Exception as e:
            raise ImportError(
                "Trying to load the Lorenz96 task failed. This task requires `abcpy`"
                "which is not installed by default (see the above error for"
                "the full traceback)."
            ) from e
        return Lorenz96()
    elif task_name == "ornstein_uhlenbeck":
        # return OrnsteinUhlenbeck()
        raise NotImplementedError
    elif task_name == "MultiModalLikelihoodTask":
        return MultiModalLikelihoodTask()
    elif task_name == "pyloric":
        from sbibm_unle_extra.tasks.pyloric_stg import Pyloric  # type: ignore

        return Pyloric()
    else:
        return sbibm.get_task(task_name)


class JaxTask:
    def __init__(self, task: Task) -> None:
        self.task = task

    def get_prior_dist(self) -> np_distributions.Distribution:
        prior_dist = self.task.get_prior_dist()
        assert isinstance(prior_dist, pyro_distributions.Distribution), prior_dist

        if self.task.name == "Lorenz96":
            p = getattr(self.task, "_jax_prior_dist", None)
            assert isinstance(p, np_distributions.Distribution)
            return p

        converted_prior_dist = convert_dist(prior_dist, implementation="numpyro")
        assert isinstance(converted_prior_dist, np_distributions.Distribution)

        return converted_prior_dist

    @classmethod
    def from_task_name(cls: Type[Self], task_name: str) -> Self:
        from sbibm_unle_extra.tasks import get_task

        return cls(get_task(task_name))

    def get_simulator(self) -> Simulator_T:
        pyro_simulator = self.task.get_simulator()

        def simulator(thetas: Array) -> Array:
            torch_thetas = torch.from_numpy(np.array(thetas)).float()
            observations = jnp.array(pyro_simulator(torch_thetas))
            if jax.config.jax_enable_x64:  # type: ignore
                observations = jnp.array(observations, dtype=jnp.float64)
            return observations

        if isinstance(pyro_simulator, SimulatorWithPrecomputedDataset):

            def get_precomputed_dataset(num_samples):
                thetas_torch, xs_torch = pyro_simulator.get_large_precomputed_dataset(
                    num_samples
                )
                assert len(thetas_torch) == len(xs_torch) == num_samples
                thetas = jnp.array(thetas_torch.detach().numpy())
                xs = jnp.array(xs_torch.detach().numpy())

                if jax.config.jax_enable_x64:  # type: ignore
                    thetas = jnp.array(thetas, dtype=jnp.float64)
                    xs = jnp.array(xs, dtype=jnp.float64)
                return thetas[:num_samples], xs[:num_samples]

            return SimulatorWithPrecomputedDataset(simulator, get_precomputed_dataset)
        else:
            return simulator

    def get_observation(self, num_observation: int) -> Array:
        return jnp.array(self.task.get_observation(num_observation))

    def _parameter_event_space_bijector(self) -> np_transforms.Transform:
        prior_dist = self.get_prior_dist()
        return np_distributions.biject_to(prior_dist.support)

    def __reduce__(self):
        return JaxTask.from_task_name, (self.task.name,)
