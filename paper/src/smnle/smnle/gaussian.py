from pathlib import Path
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pyro
import pyro.distributions as pdist
import torch
from jax import random
from jax.tree_util import tree_map
from numpyro import distributions as npdist
from sbibm.tasks import Task
from sbibm.tasks.simulator import Simulator
from torch.nn.functional import one_hot

from unle.samplers.distributions import maybe_wrap
from unle.samplers.inference_algorithms.importance_sampling.smc import SMC, SMCConfig
from unle.samplers.kernels.mala import MALAConfig, MALAKernelFactory
from sbibm_unle_extra.pyro_to_numpyro import convert_dist


class GaussianTask(Task):
    def __init__(self, use_deterministic_first_observation=True):
        observation_seeds = [
            1000011,  # observation 1
            1000001,  # observation 2
            1000002,  # observation 3
            1000003,  # observation 4
            1000013,  # observation 5
            1000005,  # observation 6
            1000006,  # observation 7
            1000007,  # observation 8
            1000008,  # observation 9
            1000009,  # observation 10
        ]

        super().__init__(
            dim_parameters=2,
            dim_data=2,
            name="GaussianTask",
            name_display="GaussianTask",
            num_observations=10,
            num_posterior_samples=10000,  # type: ignore
            num_reference_posterior_samples=10000,
            num_simulations=[1000, 10000, 100000, 1000000],
            path=Path(__file__).parent.absolute(),
            observation_seeds=observation_seeds,
        )
        self.prior_params = {
            "mean_bounds": torch.Tensor([-10.0, 10.0]),
            "sigma_bounds": torch.Tensor([1.0, 10.0]),
        }
        self.prior_dist = pdist.Uniform(
            low=torch.Tensor(
                [
                    self.prior_params["mean_bounds"][0],
                    self.prior_params["sigma_bounds"][0],
                ]
            ),
            high=torch.Tensor(
                [
                    self.prior_params["mean_bounds"][1],
                    self.prior_params["sigma_bounds"][1],
                ]
            ),
        ).to_event(1)

        self.prior_dist.set_default_validate_args(False)
        self.jax_prior_dist = convert_dist(self.prior_dist, implementation="numpyro")  # type: ignore

        self.use_deterministic_first_observation = use_deterministic_first_observation

        self.dim_data = 10

    def get_prior(self) -> Callable:
        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior

    def get_prior_dist(self):
        return self.prior_dist

    def get_simulator(self, max_calls=None) -> Callable:
        def simulator(parameters):
            num_samples = parameters.shape[0]

            # loc, scale of shape (num_samples, 1)
            loc, scale = parameters[:, :1], parameters[:, 1:2]

            # reshaped such that:
            # - event_shape = (dim_data,)
            # - batch_shape = (num_samples,)
            dist = (
                pdist.Normal(loc=loc, scale=scale)
                .expand((num_samples, self.dim_data))
                .to_event(1)
            )
            return pyro.sample("data", dist)

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def _log_likelihood(self, theta, x):
        assert len(theta.shape) == len(x.shape)
        if len(theta.shape) == 1:
            loc, scale = theta[0], theta[1]
            conditional = (
                pdist.Normal(loc=loc, scale=scale).expand((self.dim_data,)).to_event(1)
            )
            return conditional.log_prob(x)
        elif len(theta.shape) == 2:
            assert theta.shape[1] == 2
            assert x.shape[1] == self.dim_data

            num_samples = theta.shape[0]

            loc, scale = theta[:, :1], theta[:, 1:2]
            conditional = (
                pdist.Normal(loc=loc, scale=scale)
                .expand((num_samples, self.dim_data))
                .to_event(1)
            )
            return conditional.log_prob(x)
        else:
            return ValueError

    def _jax_log_likelihood(self, theta, x):
        # use numpyro instead of pyro
        assert len(theta.shape) == len(x.shape)
        if len(theta.shape) == 1:
            loc, scale = theta[0], theta[1]
            conditional = (
                npdist.Normal(loc=loc, scale=scale).expand((self.dim_data,)).to_event(1)
            )
            return conditional.log_prob(x)
        elif len(theta.shape) == 2:
            assert theta.shape[1] == 2
            assert x.shape[1] == self.dim_data

            num_samples = theta.shape[0]

            loc, scale = theta[:, :1], theta[:, 1:2]
            conditional = (
                npdist.Normal(loc=loc, scale=scale)
                .expand((num_samples, self.dim_data))
                .to_event(1)
            )
            return conditional.log_prob(x)
        else:
            return ValueError

    def _unnormalized_logpost(self, theta, x):
        # XXX: does not broadcast well.
        return self._log_likelihood(theta, x) + self.prior_dist.log_prob(theta)

    def _jax_unnormalized_logpost(self, theta, x):
        # XXX: does not broadcast well.
        return self._jax_log_likelihood(theta, x) + self.jax_prior_dist.log_prob(theta)

    def _sample_reference_posterior(
        self,
        num_samples: int,
        num_observation: int,
        observation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # TODO(pierreglaser): z-score the posterior before sampling by sampling from the
        # prior-simulator joint distribution to obtain empirical mean and stds.
        key = random.PRNGKey(num_observation)

        if observation is None:
            observation = self.get_observation(num_observation)[0]
        assert observation is not None
        assert len(observation.shape) == 1
        observation = jnp.array(observation)  # type: ignore

        logpost = maybe_wrap(
            lambda x: self._jax_unnormalized_logpost(theta=x, x=observation)
        )

        init_dist = self.jax_prior_dist

        config = SMCConfig(
            num_samples=num_samples,
            num_steps=100,
            ess_threshold=0.8,
            inner_kernel_factory=MALAKernelFactory(MALAConfig(step_size=0.001)),
            inner_kernel_steps=5,
            record_trajectory=False,
        )
        smc = SMC(config=config, log_prob=logpost)

        key, key_init = random.split(key)
        smc = smc.init(key_init, init_dist)

        key, key_smc = random.split(key)
        smc, results = smc.run(key_smc)
        return torch.Tensor(np.array(results.samples.xs))

    def _setup(self, n_jobs: int = -1, create_reference: bool = True, **kwargs: Any):
        """Setup the task: generate observations and reference posterior samples

        In most cases, you don't need to execute this method, since its results are stored to disk.

        Re-executing will overwrite existing files.

        Args:
            n_jobs: Number of to use for Joblib
            create_reference: If False, skips reference creation
        """
        from joblib import Parallel, delayed

        def run(num_observation, observation_seed, **kwargs):
            # fix observation 0 to be at 0 for the paper's illustrative example
            np.random.seed(observation_seed)
            torch.manual_seed(observation_seed)
            self._save_observation_seed(num_observation, observation_seed)

            prior = self.get_prior()
            if self.use_deterministic_first_observation and num_observation == 1:
                observation = torch.zeros((1, self.dim_data))
                true_parameters = -2 * torch.ones((1, self.dim_parameters))
            else:
                true_parameters = prior(num_samples=1)
                simulator = self.get_simulator()
                observation = simulator(true_parameters)

            self._save_true_parameters(num_observation, true_parameters)
            self._save_observation(num_observation, observation)

            if create_reference:
                reference_posterior_samples = self._sample_reference_posterior(
                    num_observation=num_observation,
                    num_samples=self.num_reference_posterior_samples,  # type: ignore
                    **kwargs,
                )
                num_unique = torch.unique(reference_posterior_samples, dim=0).shape[0]
                assert num_unique == self.num_reference_posterior_samples
                self._save_reference_posterior_samples(
                    num_observation,
                    reference_posterior_samples,
                )

        Parallel(n_jobs=n_jobs, verbose=50, backend="sequential")(
            delayed(run)(num_observation, observation_seed, **kwargs)
            for num_observation, observation_seed in enumerate(
                self.observation_seeds, start=1
            )
        )


if __name__ == "__main__":
    task = GaussianTask()
    task._setup()
