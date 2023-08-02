from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Tuple, cast

if TYPE_CHECKING:
    from jax import Array
    from jax.random import KeyArray
    from typing_extensions import Self

    from unle.filtering_correction import FilteringCorrector

import copy

import jax
import jax.numpy as jnp
from flax import struct
from jax import random
from jax.tree_util import tree_map
from numpyro import distributions as np_distributions
from numpyro.distributions import Distribution, TransformedDistribution
from numpyro.distributions.transforms import AffineTransform

from unle.distributions.base import (
    JointDistribution,
    Posterior,
    TransformedConditionalDistribution,
    TransformedPosterior,
)
from unle.filtering_correction import train_filtering_corrector
from unle.neural_networks.classification import ClassificationTrainingConfig
from unle.neural_networks.neural_networks import MLPConfig
from unle.normalizing_function_estimation import LogZNet, RegressionTrainingConfig
from unle.samplers.inference_algorithms.mcmc.base import MCMCAlgorithm
from unle.utils.preprocessing import Normalizer, find_nans, find_outliers
from unle.utils.reparametrization import compose_affine_transforms

from .distributions.auto_tractable import (
    AutoTractableConditionalDistribution,
    make_inference_config,
)
from .likelihood_estimation import EBMLikelihood, TrainerResults

# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------


def _subsample(x, n_max):
    thinning_factor = len(x) / n_max
    if thinning_factor > 1:
        print(
            "reducing the number of points x data from ",
            len(x),
            " to ",
            n_max,
        )
        print("using thinning factor: ", thinning_factor)

        rounded_thinning_factor = int(thinning_factor)
        x = x[: rounded_thinning_factor * n_max : rounded_thinning_factor]
    else:
        print("thinning factor is less than 1, not thinning")
    return x


class UNLE(struct.PyTreeNode):
    _likelihood: Optional[EBMLikelihood] = None
    filtering_corrector: Optional[FilteringCorrector] = None
    lz_net: Optional[LogZNet] = None
    posterior: Optional[AutoTractableConditionalDistribution] = None
    round_no: int = 0
    n_sigma: float = 3
    all_thetas: Tuple[Array, ...] = tuple()
    all_xs: Tuple[Array, ...] = tuple()
    all_priors: Tuple[Optional[Distribution], ...] = tuple()
    all_z_scorers: Tuple[Normalizer, ...] = tuple()
    all_nan_masks: Tuple[Array, ...] = tuple()
    all_outlier_thetas_masks: Tuple[Array, ...] = tuple()
    all_outlier_xs_masks: Tuple[Array, ...] = tuple()
    # stateless for now, pure refactoring

    @classmethod
    def create(
        cls,
        n_sigma: float = 3,
    ):
        return cls(n_sigma=n_sigma)

    @property
    def dim_observations(self):
        if len(self.all_xs) == 0:
            raise ValueError("No observations have been added yet")
        return self.all_xs[0].shape[1]

    @property
    def dim_parameters(self):
        if len(self.all_thetas) == 0:
            raise ValueError("No parameters have been added yet")
        return self.all_thetas[0].shape[1]

    def get_proposal(self, round_no: int) -> Distribution:
        proposal = self.all_priors[round_no]
        assert proposal is not None
        return proposal

    def get_z_scorer(self, round_no: int) -> Normalizer:
        assert len(self.all_z_scorers) > round_no
        return self.all_z_scorers[round_no]

    def get_crossround_pushforward(
        self,
        who: Literal["params", "observations", "both"],
        r_old: int = -2,
        r_new: int = -1,
    ) -> AffineTransform:
        """
        Returns a map transforming the normalized parameters/observations
        from round `r_old` to normalized parameters/observations from round
        `r_new`, e.g. T_{r_old}^{-1} o T_{r_new}., where T_{r_old} and T_{r_new}
        are the normalizers of round `r_old` and `r_new` respectively.
        """
        n_old = self.all_z_scorers[r_old].get_inverse_transform(who)
        n_new = self.all_z_scorers[r_new].get_transform(who)
        return compose_affine_transforms(n_new, n_old)

    def current_dataset_contains_invalid_observations(self, round_no) -> Array:
        return jnp.any(
            self.all_nan_masks[round_no] | self.all_outlier_xs_masks[round_no]
        )

    def get_likelihood_estimation_training_data(
        self, likelihood_training_method: str, round_no: int, z_score: bool
    ) -> Tuple[Array, Array, Optional[Distribution]]:
        proposal_dist = self.all_priors[round_no]
        assert proposal_dist is not None

        if likelihood_training_method == "likelihood" and round_no > 0:
            print(
                "combining unnormalized data with unle-likelihood: merging all datasets"
            )
            # case of an unnormalized likelihood EBM which does not rely on prior
            # probabilities at all: safe to concatenate all datasets
            xs = jnp.concatenate(self.all_xs)
            thetas = jnp.concatenate(self.all_thetas)
            nan_mask = jnp.concatenate(self.all_nan_masks)
            outlier_xs_mask = jnp.concatenate(self.all_outlier_xs_masks)
            outlier_thetas_mask = jnp.concatenate(self.all_outlier_thetas_masks)

            xs = xs[~(nan_mask | outlier_xs_mask | outlier_thetas_mask)]
            thetas = thetas[~(nan_mask | outlier_xs_mask | outlier_thetas_mask)]
            proposal_dist = None
        else:
            if round_no > 0:
                print("not reusing data from previous rounds.")

            xs = self.all_xs[round_no]
            thetas = self.all_thetas[round_no]
            nan_mask = self.all_nan_masks[round_no]

            xs = xs[~nan_mask]
            thetas = thetas[~nan_mask]

            proposal_dist = self.all_priors[round_no]

        if z_score:
            z_scorer = self.get_z_scorer(round_no)
            xs = z_scorer.get_transform("observations")(xs)
            thetas = z_scorer.get_transform("params")(thetas)

            if proposal_dist is not None:
                proposal_dist = cast(
                    TransformedDistribution,
                    TransformedDistribution(
                        proposal_dist, z_scorer.get_transform("params")
                    ),
                )

        return thetas, xs, proposal_dist

    def get_filtering_correction_training_data(
        self,
        round_no: int,  # exclude_based_on_current_round_xs: bool
        z_score: bool = True,
    ) -> Tuple[Array, Array]:
        all_thetas = jnp.concatenate(self.all_thetas)
        if z_score:
            all_thetas = self.get_z_scorer(round_no).get_transform("params")(all_thetas)

        valid_mask = ~(
            jnp.concatenate(self.all_nan_masks)
            | jnp.concatenate(self.all_outlier_xs_masks)
        )
        return all_thetas, valid_mask.astype(jnp.float32)

    def append_simulations(
        self, parameters, observations, proposal, n_sigma: float = 3.0
    ) -> Self:
        all_xs = (*self.all_xs, observations)
        all_thetas = (*self.all_thetas, parameters)
        all_priors = (*self.all_priors, proposal)

        z_scorer = Normalizer.create_and_fit(parameters, observations)
        all_z_scorers = (*self.all_z_scorers, z_scorer)

        # updaate masks
        all_nan_masks = tuple(find_nans(xs) for xs in all_xs)

        all_outlier_xs_masks = tuple(
            find_outliers(
                xs,
                z_scorer.observations_mean,
                z_scorer.observations_std + 1e-8,
                n_sigma,
            )
            for xs in all_xs
        )

        all_outlier_thetas_masks = tuple(
            find_outliers(
                thetas,
                z_scorer.params_mean,
                z_scorer.params_std + 1e-8,
                n_sigma,
                norm="l_inf",
            )
            for thetas in all_thetas
        )

        self = self.replace(
            all_xs=all_xs,
            all_thetas=all_thetas,
            all_priors=all_priors,
            all_z_scorers=all_z_scorers,
            all_nan_masks=all_nan_masks,
            all_outlier_xs_masks=all_outlier_xs_masks,
            all_outlier_thetas_masks=all_outlier_thetas_masks,
        )
        return self

    def _get_zscored_likelihood(self) -> EBMLikelihood:
        assert self._likelihood is not None
        return self._likelihood

    def get_likelihood(self) -> TransformedConditionalDistribution[EBMLikelihood]:
        assert self._likelihood is not None
        return cast(
            TransformedConditionalDistribution,
            TransformedConditionalDistribution(
                self._likelihood,
                self.z_scorer.get_transform("observations").inv,
                self.z_scorer.get_transform("params").inv,
            ),
        )

    def get_filtering_corrector(self) -> FilteringCorrector:
        filtering_corrector = self.filtering_corrector
        assert self.likelihood_requires_filtering_correction()
        assert filtering_corrector is not None
        return filtering_corrector

    @property
    def z_scorer(self):
        return self.all_z_scorers[self.round_no]

    def likelihood_requires_filtering_correction(
        self,
    ) -> bool:
        needs_filtering_correction_if_nans = (
            self._get_zscored_likelihood().needs_filtering_correction_if_nans
        )
        assert needs_filtering_correction_if_nans is not None
        return (
            needs_filtering_correction_if_nans
            and self.current_dataset_contains_invalid_observations(self.round_no)
        )

    def _initialize_likelihood(self, key, width, depth):
        return cast(
            EBMLikelihood,
            EBMLikelihood(
                ebm_width=width,
                ebm_depth=depth,
                event_shape=(self.dim_observations,),
                conditioned_event_shape=(self.dim_parameters,),
                key=key,
            ),
        )

    def train_likelihood(
        self,
        key: KeyArray,
        ebm_model_type: Literal["likelihood", "joint_tilted"],
        normalize_data: bool = True,
        width: int = 50,
        depth: int = 4,
        max_iter: int = 500,
        num_frozen_steps: int = 50,
        num_mala_steps: int = 50,
        num_particles: int = 1000,
        use_warm_start: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-1,
        noise_injection_val: float = 0.001,
        batch_size: Optional[int] = None,
        num_smc_steps: int = 5,
        ess_threshold: float = 0.0,
        correction_net_width: int = 200,
        correction_net_depth: int = 3,
        correction_net_max_iter: int = 200,
    ) -> Tuple[TrainerResults, Self]:
        round_no = self.round_no

        # Prepare Likelihood Training ----------------------------------------------

        (
            parameters,
            observations,
            proposal,
        ) = self.get_likelihood_estimation_training_data(
            ebm_model_type, round_no, z_score=normalize_data
        )

        from .likelihood_estimation import make_training_config

        training_config = make_training_config(
            ebm_model_type,
            self.dim_observations,
            proposal,
            max_iter,
            num_frozen_steps,
            num_mala_steps,
            num_particles,
            use_warm_start,
            learning_rate,
            weight_decay,
            noise_injection_val,
            batch_size,
            num_smc_steps,
            ess_threshold,
        )

        # Build or update the likelihood -------------------------------------------

        if round_no == 0:
            key, subkey = random.split(key)
            likelihood = self._initialize_likelihood(subkey, width, depth)
        else:
            likelihood = self._get_zscored_likelihood()
            if normalize_data:
                likelihood = likelihood.reparametrize(
                    self.get_crossround_pushforward(who="both")
                )

        key, subkey = random.split(key)
        training_results, likelihood = likelihood.train(
            parameters,
            observations,
            proposal,
            subkey,
            ebm_model_type,
            training_config,
        )

        self = self.replace(_likelihood=likelihood)

        # Train Filtering Correction Net -----------------------------------------------
        if self.likelihood_requires_filtering_correction():
            self = self._correct_for_filtering(
                correction_net_width,
                correction_net_depth,
                correction_net_max_iter,
            )

        return training_results, self

    def get_lz_net(self):
        assert self.lz_net is not None
        return self.lz_net

    def _get_new_lznet_points(self, num_samples):
        round_no = self.round_no
        z_scorer = self.get_z_scorer(round_no)
        init_thetas = self.all_thetas[round_no]
        init_xs = self.all_xs[round_no]
        filter = (
            self.all_nan_masks[round_no]
            | self.all_outlier_xs_masks[round_no]
            | self.all_outlier_thetas_masks[round_no]
        )
        init_thetas = z_scorer.get_transform("params")(init_thetas[~filter])
        init_xs = z_scorer.get_transform("observations")(init_xs[~filter])

        init_thetas, init_xs = tree_map(
            lambda x: _subsample(x, num_samples),
            (init_thetas, init_xs),
        )
        return init_thetas, init_xs

    def train_lznet(
        self,
        key: KeyArray,
        width: int = 100,
        depth: int = 4,
        z_score_output: bool = True,
        num_new_training_samples: int = 1000,
    ) -> Self:
        likelihood = self._get_zscored_likelihood()
        key, subkey = random.split(key)

        if self.round_no == 0:
            lz_net = LogZNet.create(
                likelihood=likelihood, config=MLPConfig(width, depth)
            )
        else:
            lz_net = self.get_lz_net()
            lz_net = lz_net.set_likelihood(likelihood)
            lz_net = lz_net.reparametrize(
                self.get_crossround_pushforward(who="params"),
                self.get_crossround_pushforward(who="observations"),
            )

        thetas, init_xs = self._get_new_lznet_points(num_new_training_samples)

        lz_net = lz_net.add_new_points(
            thetas,
            init_xs,
        )

        lz_net = lz_net.create_training_data(subkey)

        key, subkey = random.split(key)
        lz_net = lz_net.train(
            subkey,
            training_config=RegressionTrainingConfig(),
            z_score_output=z_score_output,
        )

        likelihood = likelihood.set_log_z_net(
            lz_net.replace(all_lz_net_sampling_algs=None)
        )

        return self.replace(
            lz_net=lz_net,
            _likelihood=likelihood,
        )

    # Filtering Correction
    # ----------------------------------------------------------------------------------

    def _correct_for_filtering(
        self,
        width: int = 200,
        depth: int = 3,
        max_iter: int = 200,
        normalize_data: bool = True,
    ) -> Self:
        round_no = self.round_no
        (
            all_thetas_zscored,
            valid_mask,
        ) = self.get_filtering_correction_training_data(
            round_no, z_score=normalize_data
        )
        # Filter training data points that are too far away from the mean
        # of the training set for numerical stability purposes.
        # XXX: This call does not follow the outlier-filtering convention adopted
        # in the rest of the codebase, as it uses the std of `all_thetas_zscored`
        # and not the std of the last dataset from the last simulation round.
        # This is not drastically important as this call filters outlier
        # **parameters** from the training set of the correction network for
        # stability purposes. TODO: use the same conventions as elsewhere.
        # TODO: use jnp.mean(all_thetas_zscored, axis=0) instead of 0.0
        # to support non-normalized data
        outlier_theta_mask = find_outliers(
            all_thetas_zscored,
            0.0,
            jnp.std(all_thetas_zscored, axis=0),
            self.n_sigma,
        )
        all_thetas_zscored = all_thetas_zscored[~outlier_theta_mask]
        valid_mask = valid_mask[~outlier_theta_mask]

        print(
            f"using {jnp.sum(~outlier_theta_mask)} non-outlier samples to fit"
            " correction network"
        )
        if sum(valid_mask) >= len(all_thetas_zscored) - 1:
            # avoid sklern issues due 1-sample class
            print("only one invalid sample found, skipping correction step")
            filtering_corrector = None
        else:
            training_config = ClassificationTrainingConfig(max_iter=max_iter)
            mlp_config = MLPConfig(width, depth, activation=jax.nn.relu, num_outputs=2)
            filtering_corrector = train_filtering_corrector(
                all_thetas_zscored, valid_mask, mlp_config, training_config
            )
        return self.replace(filtering_corrector=filtering_corrector)

    # Build Posterior
    # ----------------------------------------------------------------------------------

    def build_posterior(
        self,
        prior: np_distributions.Distribution,
        x_obs: Array,
        sampler: str = "mcmc",
        num_warmup_steps: int = 500,
        exchange_mcmc_inner_sampler_num_steps: int = 100,
    ) -> Tuple[Self, AutoTractableConditionalDistribution]:
        likelihood = self._get_zscored_likelihood()
        if self.likelihood_requires_filtering_correction():
            filtering_corrector = self.get_filtering_corrector()
        else:
            filtering_corrector = None

        z_scored_prior_dist = np_distributions.TransformedDistribution(
            prior, self.z_scorer.get_transform("params")
        )

        init_dist = z_scored_prior_dist
        if filtering_corrector is not None:
            z_scored_prior_dist = cast(
                Posterior,
                JointDistribution(
                    z_scored_prior_dist, filtering_corrector
                ).condition_out_variable(1),
            )

        joint_distribution = cast(
            JointDistribution, JointDistribution(z_scored_prior_dist, likelihood)
        )
        posterior = joint_distribution.condition_out_variable(
            self.z_scorer.get_transform("observations")(x_obs)
        )

        posterior = cast(
            TransformedPosterior,
            TransformedPosterior(
                posterior,
                self.z_scorer.get_transform("params").inv,
                self.z_scorer.get_transform("observations").inv,
            ),
        )

        inference_factory = make_inference_config(
            likelihood,
            sampler,
            num_warmup_steps,
            exchange_mcmc_inner_sampler_num_steps,
        )

        posterior = cast(
            AutoTractableConditionalDistribution,
            AutoTractableConditionalDistribution(
                posterior,
                inference_factory,
                init_dist,
            ),
        )

        if self.posterior is not None:
            posterior = self._warm_start_posterior_sampling_from_previous_round(
                posterior,
                self.posterior,  # previous posterior
            )
        self = self.replace(posterior=posterior)
        return self, posterior

    def _warm_start_posterior_sampling_from_previous_round(
        self,
        posterior: AutoTractableConditionalDistribution,
        prev_posterior: AutoTractableConditionalDistribution,
    ) -> AutoTractableConditionalDistribution:
        new_alg = prev_posterior.sampling_alg
        assert isinstance(new_alg, MCMCAlgorithm)
        new_alg = new_alg.reparametrize(self.get_crossround_pushforward("params"))
        new_alg = new_alg.set_log_prob(posterior.sampling_alg.log_prob)

        new_posterior = copy.copy(posterior)
        new_posterior.sampling_alg = new_alg
        return new_posterior
