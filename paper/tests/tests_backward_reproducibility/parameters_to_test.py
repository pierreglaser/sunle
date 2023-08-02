"""Parameter sets used for testing backward reproducibility of (S)UNLE.

Dictionaries containing the parameters to test against. If adding/changing a
parameter, the test-data generation will need to be re-run to
inlcude test data with the updated parameter set.
"""

# TODO(pierreglaser): parameterize the number of training iterations of the log-z
# net

_default_params = {
    "task": "two_moons",
    "num_observation": 1,
    "learning_rate": 0.01,
    "use_warm_start": True,
    "inference_proposal": "prior",
    "evaluate_posterior": False,
}

# parameter sets with a low compute budget that are tested in CI workflows
# (and compared agains reference results with the same parameter set)
# to ensure that refactoring changes do not affect the numerical results
# of unle/sunle/sunle_vi

_unle_default_params = {
    # design choices that cannot/shouldn't be changed for unle
    "proposal": "prior+noise",
    "sampler": "smc",
    "inference_sampler": "smc",
    "estimate_loss": False,
    "ebm_model_type": "joint_tilted",
    # these parameters affect the training/simulation/inference budget
    "num_smc_steps": 5,
    "num_mala_steps": 3,
    "max_iter": 10,
    "num_samples": 200,
    "num_posterior_samples": 100,
    "ebm_depth": 2,
    "ebm_width": 16,
    "num_particles": 100
    # not used for unle
    # - inference_num_warmup_steps (not used when inference_sampler="smc")
    # - training_num_frozen_steps (not used when sampler="smc")
    # - exchange_mcmc_inner_sampler_num_steps
}

_sunle_default_params = {
    # design choices that cannot/shouldn't be changed for sunle
    "proposal": "data",
    "sampler": "mala",
    "ebm_model_type": "likelihood",
    # these parameters affect the training/simulation/inference budget
    "num_mala_steps": 20,
    "max_iter": 10,
    "num_rounds": 2,
    "num_samples": 200,
    "num_posterior_samples": 100,
    "inference_num_warmup_steps": 10,
    "training_num_frozen_steps": 10,
    "ebm_depth": 2,
    "ebm_width": 16,
    "num_particles": 100
    # not used for sunle
    # num_smc_steps=5,
    # estimate_loss=False,
}

parameters_sets_unle = (
    {
        **_default_params,
        **_unle_default_params,
    },
)

parameters_sets_sunle = (
    {
        **_default_params,
        **_sunle_default_params,
        "estimate_log_normalizer": False,
        "inference_sampler": "exchange_mcmc",
        "exchange_mcmc_inner_sampler_num_steps": 10,
    },
    {
        **_default_params,
        **_sunle_default_params,
        "estimate_log_normalizer": False,
        "inference_sampler": "exchange_mcmc",
        "task": "two_moons_with_nans",  # test nan filtering and posterior correction
        "calibration_net_max_iter": 20,
        "n_sigma": 1,  # test outlier filtering and posterior correction
    },
)

parameters_sets_sunle_vi = (
    {
        **_default_params,
        **_sunle_default_params,
        "estimate_log_normalizer": True,
        "inference_sampler": "mala",
        # - exchange_mcmc_inner_sampler_num_steps not used when inference_sampler="mala"
    },
    {
        **_default_params,
        **_sunle_default_params,
        "estimate_log_normalizer": True,
        "inference_sampler": "mala",
        "task": "two_moons_with_nans",
        "calibration_net_max_iter": 20,
    },
)

# parameter sets with higher compute budget that can be run locally to
# visually inspect the quality of unle/sunle's inference results
_sunle_default_params_larger_budget = {
    # design choices that cannot/shouldn't be changed for sunle
    "proposal": "data",
    "sampler": "mala",
    "ebm_model_type": "likelihood",
    # these parameters affect the training/simulation/inference budget
    "num_mala_steps": 20,
    "max_iter": 300,
    "num_rounds": 2,
    "num_samples": 2000,
    "num_posterior_samples": 1000,
    "inference_num_warmup_steps": 100,
    "training_num_frozen_steps": 20,
    "ebm_depth": 4,
    "ebm_width": 50,
    "num_particles": 1000
    # not used for sunle
    # num_smc_steps=5,
    # estimate_loss=False,
}

_unle_default_params_larger_budget = {
    # design choices that cannot/shouldn't be changed for unle
    "proposal": "prior+noise",
    "sampler": "smc",
    "inference_sampler": "smc",
    "estimate_loss": False,
    "ebm_model_type": "joint_tilted",
    # these parameters affect the training/simulation/inference budget
    "num_smc_steps": 5,
    "num_mala_steps": 3,
    "max_iter": 500,
    "num_samples": 1000,
    "num_posterior_samples": 1000,
    "ebm_depth": 4,
    "ebm_width": 50,
    "num_particles": 1000,
    # not used for unle
    # - inference_num_warmup_steps (not used when inference_sampler="smc")
    # - training_num_frozen_steps (not used when sampler="smc")
    # - exchange_mcmc_inner_sampler_num_steps
}

parameters_sets_unle_larger_budget = (
    {
        **_default_params,
        **_unle_default_params_larger_budget,
    },
)

parameters_sets_sunle_larger_budget = (
    {
        **_default_params,
        **_sunle_default_params_larger_budget,
        "estimate_log_normalizer": False,
        "inference_sampler": "exchange_mcmc",
        "exchange_mcmc_inner_sampler_num_steps": 10,
    },
)

parameters_sets_sunle_vi_larger_budget = (
    {
        **_default_params,
        **_sunle_default_params_larger_budget,
        "estimate_log_normalizer": True,
        "inference_sampler": "mala",
        # - exchange_mcmc_inner_sampler_num_steps not used when inference_sampler="mala"
    },
)
