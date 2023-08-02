import logging
import os
import pickle
import sys
from time import sleep, time

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as npdist
import pymc3 as pm  # type: ignore
import pyro.distributions as pdist
import theano.tensor as tt  # type: ignore
import torch
import torch.distributions as tdist
from abcpy.backends import BackendDummy, BackendMPI
from abcpy.continuousmodels import Uniform
from abcpy.inferences import DrawFromPrior
from abcpy.NN_utilities.utilities import save_net
from abcpy.statistics import Identity
from abcpy.statisticslearning import ExponentialFamilyScoreMatching as ExpFamStatistics
from jax import jit, random
from unle.samplers.distributions import DoublyIntractableLogDensity, maybe_wrap
from unle.samplers.inference_algorithms.mcmc.base import MCMCAlgorithmFactory, MCMCConfig
from unle.samplers.kernels.mala import MALAConfig, MALAKernelFactory
from unle.samplers.kernels.rwmh import RWConfig, RWKernel, RWKernelFactory
from unle.samplers.kernels.savm import SAVMConfig, SAVMKernelFactory
from numpyro import distributions as np_distributions
from numpyro.distributions import transforms as np_transforms
from pyro.distributions import transforms as pyro_transforms
from pyro.distributions.transforms import AffineTransform as pAffineTransform
from pyro.distributions.transforms import identity_transform as pIdentityTransform
from sbi import inference as inference
from sbibm.metrics.c2st import c2st
from sbibm.metrics.mmd import mmd
from sbibm_unle_extra.pyro_to_numpyro import convert_dist, convert_transform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn
from torch.distributions.transformed_distribution import TransformedDistribution

from unle.distributions.base import TransformedConditionalDistribution, JointDistribution

from smnle.jax_torch_interop import make_jax_likelihood
from smnle.src.exchange_mcmc import (
    exchange_MCMC_with_SM_statistics,
    uniform_prior_theta,
)
from smnle.src.functions import (
    DummyScaler,
    LogLike,
    generate_training_samples_ABC_model,
    plot_losses,
    save_dict_to_json,
    scale_samples,
)
from smnle.src.networks import (
    create_PEN_architecture,
    createDefaultNN,
    createDefaultNNWithDerivatives,
)
from smnle.src.parsers import parser_generate_obs, train_net_batch_parser
from smnle.src.training_routines import FP_training_routine
from smnle.src.utils_arma_example import (
    ARMAmodel,
    ar2_log_lik_for_mcmc,
    ma2_log_lik_for_mcmc,
)
from smnle.src.utils_beta_example import generate_beta_training_samples
from smnle.src.utils_gamma_example import generate_gamma_training_samples
from smnle.src.utils_gaussian_example import generate_gaussian_training_samples
from smnle.src.utils_Lorenz95_example import StochLorenz95

sys.path.append(os.getcwd())


os.environ["QT_QPA_PLATFORM"] = "offscreen"


def smnle(
    model,
    cuda=False,
    lam=0,
    seed=1,
    technique="SSM",
    epochs=500,
    no_scheduler=False,
    noise_sliced="radermacher",
    var_red_sliced=True,
    batch_norm_last_layer=True,
    affine_batch_norm=False,
    SM_lr=0.01,
    SM_lr_theta=0.01,
    batch_size=1000,
    early_stopping=True,
    update_batchnorm_running_means_before_eval=True,
    momentum=0.9,
    epochs_test_interval=10,
    epochs_before_early_stopping=200,
    n_samples_training=10**4,
    n_samples_evaluation=10**4,
    num_observation=1,
    nonlinearity=torch.nn.Softplus,
    num_posterior_samples=10000,
    mcmc_num_chains=50,
    mcmc_num_warmup_steps=5000,
    mcmc_num_inner_steps=200,
    return_posterior_samples=False,
    use_orig_mcmc_impl=False,
    return_posterior=False,
    scaling_method="MinMaxScaler"
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    import time

    # from gaussian import GaussianTask
    # from sbibm.tasks import get_task
    from sbibm_unle_extra.tasks import get_task

    t = get_task(model)
    x_obs = t.get_observation(num_observation)[0]
    if model == "lotka_volterra":
        # simulating lotka volterra samples requires julia. use some cached samples to allow users to
        # experiment with amortized inference on this model without having to install julia + compile a sysimage
        p = t.get_prior()
        prior = t.get_prior_dist()
        all_theta_vect = np.load("theta_vect_lv.npy")
        all_samples_matrix = np.load("samples_matrix_lv.npy")

        # samples are positive -> log transform
        assert np.all(np.all(all_samples_matrix > 0))
        assert (x_obs > 0).all()

        x_obs = torch.log(x_obs)
        all_samples_matrix = np.log(all_samples_matrix)

        num_samples = all_theta_vect.shape[0]
        assert num_samples == all_samples_matrix.shape[0]

        assert num_samples > n_samples_training + n_samples_evaluation

        theta_vect = all_theta_vect[:n_samples_training]
        theta_vect_test = all_theta_vect[
            n_samples_training : n_samples_training + n_samples_evaluation
        ]

        samples_matrix = all_samples_matrix[:n_samples_training]
        samples_matrix_test = all_samples_matrix[
            n_samples_training : n_samples_training + n_samples_evaluation
        ]

        lower_bound = upper_bound = None
    else:
        p = t.get_prior()
        prior = t.get_prior_dist()
        s = t.get_simulator()

        # true_theta = p()[0]
        # x_obs = s(true_theta[None, :])[0]
        # print(true_theta, x_obs)

        lower_bound = upper_bound = None
        theta_vect = p(n_samples_training)
        theta_vect_test = p(n_samples_evaluation)

        samples_matrix = s(theta_vect)
        samples_matrix_test = s(theta_vect_test)

    if model == "gaussian":
        net_data_SM_architecture = createDefaultNNWithDerivatives(
            10, 3, [30, 50, 50, 20], nonlinearity=nonlinearity
        )
        # net_data_SM_architecture = createDefaultNN(10, 3, [15, 15, 5], nonlinearity=nonlinearity())
        # net_theta_SM_architecture = createDefaultNN(2, 2, [5, 5], nonlinearity=nonlinearity())
        net_theta_SM_architecture = createDefaultNN(
            2,
            2,
            [15, 30, 30, 15],
            nonlinearity=nonlinearity(),
            batch_norm_last_layer=batch_norm_last_layer,
            affine_batch_norm=affine_batch_norm,
            batch_norm_last_layer_momentum=momentum,
        )
        net_FP_architecture = createDefaultNN(
            10, 2, [30, 50, 50, 20], nonlinearity=nonlinearity()
        )
        # net_FP_architecture = createDefaultNN(10, 2, [30, 50, 50, 20], nonlinearity=torch.nn.ReLU())
        # net_FP_architecture = createDefaultNN(10, 2, [15, 15, 5], nonlinearity=nonlinearity())
    elif model == "two_moons":
        net_data_SM_architecture = createDefaultNNWithDerivatives(
            2, 50, [50, 50, 50, 50], nonlinearity=nonlinearity
        )
        # net_data_SM_architecture = createDefaultNN(10, 3, [15, 15, 5], nonlinearity=nonlinearity())
        # net_theta_SM_architecture = createDefaultNN(2, 2, [5, 5], nonlinearity=nonlinearity())
        net_theta_SM_architecture = createDefaultNN(
            2,
            49,
            [50, 50, 50, 50],
            nonlinearity=nonlinearity(),
            batch_norm_last_layer=batch_norm_last_layer,
            affine_batch_norm=affine_batch_norm,
            batch_norm_last_layer_momentum=momentum,
        )
        # epochs = 500
        # epochs_before_early_stopping = 500

        # net_FP_architecture = createDefaultNN(10, 2, [30, 50, 50, 20], nonlinearity=torch.nn.ReLU())
        # net_FP_architecture = createDefaultNN(10, 2, [15, 15, 5], nonlinearity=nonlinearity())
    elif model == "MultiModalLikelihoodTask":
        net_data_SM_architecture = createDefaultNNWithDerivatives(
            2, 20, [20, 20, 20, 20], nonlinearity=nonlinearity
        )
        # net_data_SM_architecture = createDefaultNN(10, 3, [15, 15, 5], nonlinearity=nonlinearity())
        # net_theta_SM_architecture = createDefaultNN(2, 2, [5, 5], nonlinearity=nonlinearity())
        net_theta_SM_architecture = createDefaultNN(
            2,
            19,
            [20, 20, 20, 20],
            nonlinearity=nonlinearity(),
            batch_norm_last_layer=batch_norm_last_layer,
            affine_batch_norm=affine_batch_norm,
            batch_norm_last_layer_momentum=momentum,
        )
        # epochs = 500
        # epochs_before_early_stopping = 500

        # net_FP_architecture = createDefaultNN(10, 2, [30, 50, 50, 20], nonlinearity=torch.nn.ReLU())
        # net_FP_architecture = createDefaultNN(10, 2, [15, 15, 5], nonlinearity=nonlinearity())
    elif model == "slcp":
        net_data_SM_architecture = createDefaultNNWithDerivatives(
            8, 50, [50, 50, 50, 50], nonlinearity=nonlinearity
        )
        # net_data_SM_architecture = createDefaultNN(10, 3, [15, 15, 5], nonlinearity=nonlinearity())
        # net_theta_SM_architecture = createDefaultNN(2, 2, [5, 5], nonlinearity=nonlinearity())
        net_theta_SM_architecture = createDefaultNN(
            5,
            49,
            [50, 50, 50, 50],
            nonlinearity=nonlinearity(),
            batch_norm_last_layer=batch_norm_last_layer,
            affine_batch_norm=affine_batch_norm,
            batch_norm_last_layer_momentum=momentum,
        )
        # epochs = 2000
        # epochs_before_early_stopping = 2000
        # net_FP_architecture = createDefaultNN(10, 2, [30, 50, 50, 20], nonlinearity=torch.nn.ReLU())
        # net_FP_architecture = createDefaultNN(10, 2, [15, 15, 5], nonlinearity=nonlinearity())
    elif model == "gaussian_linear_uniform":
        net_data_SM_architecture = createDefaultNNWithDerivatives(
            10, 50, [50, 50, 50, 50], nonlinearity=nonlinearity
        )
        # net_data_SM_architecture = createDefaultNN(10, 3, [15, 15, 5], nonlinearity=nonlinearity())
        # net_theta_SM_architecture = createDefaultNN(2, 2, [5, 5], nonlinearity=nonlinearity())
        net_theta_SM_architecture = createDefaultNN(
            10,
            49,
            [50, 50, 50, 50],
            nonlinearity=nonlinearity(),
            batch_norm_last_layer=batch_norm_last_layer,
            affine_batch_norm=affine_batch_norm,
            batch_norm_last_layer_momentum=momentum,
        )
        # epochs = 500
        # epochs_before_early_stopping = 500
        # var_red_sliced = False
        # no_scheduler = True
        # momentum = 0.1

        # net_FP_architecture = createDefaultNN(10, 2, [30, 50, 50, 20], nonlinearity=torch.nn.ReLU())
        # net_FP_architecture = createDefaultNN(10, 2, [15, 15, 5], nonlinearity=nonlinearity())
    elif model == "lotka_volterra":
        net_data_SM_architecture = createDefaultNNWithDerivatives(
            20, 50, [50, 50, 50, 50], nonlinearity=nonlinearity
        )
        # net_data_SM_architecture = createDefaultNN(10, 3, [15, 15, 5], nonlinearity=nonlinearity())
        # net_theta_SM_architecture = createDefaultNN(2, 2, [5, 5], nonlinearity=nonlinearity())
        net_theta_SM_architecture = createDefaultNN(
            4,
            49,
            [50, 50, 50, 50],
            nonlinearity=nonlinearity(),
            batch_norm_last_layer=batch_norm_last_layer,
            affine_batch_norm=affine_batch_norm,
            batch_norm_last_layer_momentum=momentum,
        )
        # epochs = 500
        # epochs_before_early_stopping = 500
        # var_red_sliced = False
        # no_scheduler = True
        # momentum = 0.1

        # net_FP_architecture = createDefaultNN(10, 2, [30, 50, 50, 20], nonlinearity=torch.nn.ReLU())
        # net_FP_architecture = createDefaultNN(10, 2, [15, 15, 5], nonlinearity=nonlinearity())
    else:
        raise ValueError(f"Model {model} not implemented")

    if seed is not None:
        torch.manual_seed(seed)
    # define networks
    net_data_SM = net_data_SM_architecture()
    net_theta_SM = net_theta_SM_architecture()

    # convert the simulations and parameters to numpy if they are torch objects:
    theta_vect = (
        theta_vect.numpy() if isinstance(theta_vect, torch.Tensor) else theta_vect
    )
    theta_vect_test = (
        theta_vect_test.numpy()
        if isinstance(theta_vect_test, torch.Tensor)
        else theta_vect_test
    )
    samples_matrix = (
        samples_matrix.numpy()
        if isinstance(samples_matrix, torch.Tensor)
        else samples_matrix
    )
    samples_matrix_test = (
        samples_matrix_test.numpy()
        if isinstance(samples_matrix_test, torch.Tensor)
        else samples_matrix_test
    )

    # class IdentityTransformer(BaseEstimator, TransformerMixin):
    #     def __init__(self):
    #         pass
    #
    #     def fit(self, input_array, y=None):
    #         return self
    #
    #     def transform(self, input_array, y=None):
    #         return input_array*1
    #

    def get_transform(scaler) -> pAffineTransform:
        # if isinstance(min_max_scaler, IdentityTransformer):
        #     return pIdentityTransform
        # else:
        if isinstance(scaler, MinMaxScaler):
            return pAffineTransform(
                loc=torch.from_numpy(scaler.min_),
                scale=torch.from_numpy(scaler.scale_),
            )
        if isinstance(scaler, StandardScaler):
            return pAffineTransform(
                loc=-torch.from_numpy(scaler.mean_/scaler.scale_),
                scale=1/torch.from_numpy(scaler.scale_),
            )

    if not use_orig_mcmc_impl:
        # perform the scaling step oursevles when using own MCMC implementations, so that we know which are these objects and
        # we can transform them into jax functions.
        if scaling_method == "MinMaxScaler":
            theta_scaler = MinMaxScaler().fit(theta_vect)
        elif scaling_method == "StandardScaler":
            theta_scaler = StandardScaler().fit(theta_vect)
        else:
            raise ValueError(f"Scaling method {scaling_method} not implemented")
        # theta_scaler = IdentityTransformer()

        theta_vect_transformed = theta_scaler.transform(theta_vect)
        theta_vect_test_transformed = theta_scaler.transform(theta_vect_test)

        if scaling_method == "MinMaxScaler":
            samples_scaler = MinMaxScaler().fit(samples_matrix)
        elif scaling_method == "StandardScaler":
            samples_scaler = StandardScaler().fit(samples_matrix)
        else:
            raise ValueError(f"Scaling method {scaling_method} not implemented")

        samples_matrix_transformed = samples_scaler.transform(samples_matrix)
        samples_matrix_test_transformed = samples_scaler.transform(samples_matrix_test)

        scale_parameters = scale_samples = False

    else:
        # scaling done by ExpFamStatistics later on
        assert scaling_method == "MinMaxScaler"
        theta_scaler = samples_scaler = None
        theta_vect_transformed = theta_vect
        theta_vect_test_transformed = theta_vect_test

        samples_matrix_transformed = samples_matrix
        samples_matrix_test_transformed = samples_matrix_test

        scale_parameters = scale_samples = True

    t0_training = time.time()
    statistics_learning = ExpFamStatistics(
        # backend and model are not used
        model=None,
        statistics_calc=Identity(),
        backend=BackendDummy(),
        simulations_net=net_data_SM,
        parameters_net=net_theta_SM,
        parameters=theta_vect_transformed,
        simulations=samples_matrix_transformed,
        parameters_val=theta_vect_test_transformed,
        simulations_val=samples_matrix_test_transformed,
        scale_samples=scale_samples,
        scale_parameters=scale_parameters,
        lower_bound_simulations=lower_bound,
        upper_bound_simulations=upper_bound,
        sliced=technique == "SSM",
        noise_type=noise_sliced,
        variance_reduction=var_red_sliced and not noise_sliced == "sphere",
        n_epochs=epochs,
        batch_size=batch_size,
        lr_simulations=SM_lr,
        lr_parameters=SM_lr_theta,
        seed=seed,
        start_epoch_early_stopping=epochs_before_early_stopping,
        epochs_early_stopping_interval=epochs_test_interval,
        early_stopping=early_stopping,
        scheduler_parameters=False if no_scheduler else None,
        scheduler_simulations=False if no_scheduler else None,
        cuda=cuda,
        lam=lam,
        batch_norm_update_before_test=update_batchnorm_running_means_before_eval,
    )
    time_training = time.time() - t0_training

    loss_list = statistics_learning.train_losses
    test_loss_list = statistics_learning.test_losses

    # scaler_data = statistics_learning.get_simulations_scaler()
    # scaler_theta = statistics_learning.get_parameters_scaler()

    # save_net(nets_folder + "net_theta_SM.pth", net_theta_SM)
    # save_net(nets_folder + "net_data_SM.pth", net_data_SM)
    # pickle.dump(scaler_data, open(nets_folder + "scaler_data_SM.pkl", "wb"))
    # pickle.dump(scaler_theta, open(nets_folder + "scaler_theta_SM.pkl", "wb"))

    # scaler_data_SM = scaler_data
    # scaler_theta_SM = scaler_theta

    np.random.seed(seed)

    t0_inference = time.time()

    if num_posterior_samples > 0 and use_orig_mcmc_impl:
        theta_scaler = statistics_learning.get_parameters_scaler()
        data_scaler = statistics_learning.get_simulations_scaler()

        initial_theta_exchange_MCMC = p()[0].detach().numpy()
        proposal_size_exchange_MCMC = 2 * np.array([0.1])

        print(f"\nPerforming exchange MCMC inference with {technique}.")

        # model = args.model
        n_samples = num_posterior_samples

        # XXX: some parameters are hardcoded in this branch.
        burnin_exchange_MCMC = 1000
        aux_MCMC_inner_steps_exchange_MCMC = 100
        aux_MCMC_proposal_size_exchange_MCMC = 0.1

        debug_level = logging.INFO

        tuning_window_exchange_MCMC = 100
        propose_new_theta_exchange_MCMC = "transformation"
        bridging_exch_MCMC = 0

        assert isinstance(prior, pdist.Independent)
        assert isinstance(prior.base_dist, pdist.Uniform)
        lower_bounds = prior.base_dist.low.detach().numpy()
        upper_bounds = prior.base_dist.high.detach().numpy()

        trace_exchange = exchange_MCMC_with_SM_statistics(
            x_obs,
            initial_theta_exchange_MCMC,
            lambda x: uniform_prior_theta(x, lower_bounds, upper_bounds),
            net_data_SM,
            net_theta_SM,
            data_scaler,
            theta_scaler,
            propose_new_theta_exchange_MCMC,
            T=n_samples,
            burn_in=burnin_exchange_MCMC,
            tuning_window_size=tuning_window_exchange_MCMC,
            aux_MCMC_inner_steps=aux_MCMC_inner_steps_exchange_MCMC,
            aux_MCMC_proposal_size=aux_MCMC_proposal_size_exchange_MCMC,  # type: ignore
            K=bridging_exch_MCMC,
            seed=seed,
            debug_level=debug_level,
            lower_bounds_theta=lower_bounds,
            upper_bounds_theta=upper_bounds,
            sigma=proposal_size_exchange_MCMC,
        )

        posterior_samples_numpy = trace_exchange[burnin_exchange_MCMC:]
    elif num_posterior_samples > 0:
        assert theta_scaler is not None
        assert samples_scaler is not None

        theta_min_max_affine_transform = get_transform(theta_scaler)
        samples_min_max_affine_transform = get_transform(samples_scaler)

        jax_log_likelihood = make_jax_likelihood(net_data_SM, net_theta_SM)
        transformed_prior = TransformedDistribution(
            prior, theta_min_max_affine_transform
        )
        transformed_jax_prior = convert_dist(transformed_prior, "numpyro")

        jax_posterior_log_prob = DoublyIntractableLogDensity(
            log_prior=maybe_wrap(lambda x: transformed_jax_prior.log_prob(x)),
            log_likelihood=jax_log_likelihood,
            x_obs=jnp.array(samples_min_max_affine_transform(x_obs)),  # type: ignore
        )

        config = MCMCAlgorithmFactory(
            config=MCMCConfig(
                kernel_factory=SAVMKernelFactory(
                    config=SAVMConfig(
                        aux_var_kernel_factory=MALAKernelFactory(MALAConfig(0.1)),
                        # aux_var_num_inner_steps=500,
                        aux_var_num_inner_steps=mcmc_num_inner_steps,
                        base_var_kernel_factory=RWKernelFactory(
                            config=RWConfig(
                                0.1, jnp.ones((transformed_jax_prior.event_shape[0],))
                            )
                        ),
                        aux_var_init_strategy="x_obs",
                    )
                ),
                num_samples=num_posterior_samples,
                num_chains=mcmc_num_chains,
                thinning_factor=10,
                target_accept_rate=0.5,
                num_warmup_steps=mcmc_num_warmup_steps,
                adapt_mass_matrix=False,
                adapt_step_size=True,
                progress_bar=True,
                init_using_log_l_mode=False,
            )
        )

        alg = config.build_algorithm(jax_posterior_log_prob)
        key = random.PRNGKey(0)
        key, subkey = random.split(key)

        alg = alg.init(subkey, transformed_jax_prior)

        key, subkey = random.split(key)
        alg, results = jit(type(alg).run)(alg, subkey)

        posterior_samples_torch = torch.from_numpy(np.array(results.samples.xs)).float()
        posterior_samples_torch = theta_min_max_affine_transform.inv(
            posterior_samples_torch
        )

        posterior_samples_numpy = np.array(posterior_samples_torch)

    else:
        pass


    reference_posterior_samples = t.get_reference_posterior_samples(num_observation)
    time_inference = time.time() - t0_inference

    if num_posterior_samples > 0:
        eval_results = (
            # posterior_samples_numpy,
            {
                "c2st": c2st(
                    torch.from_numpy(posterior_samples_numpy), reference_posterior_samples
                ),
                "mmd": mmd(
                    torch.from_numpy(posterior_samples_numpy), reference_posterior_samples
                ),
            },
        )
    else:
        eval_results = (
            {
                "c2st": 0.,
                "mmd": 0.
            },
        )

    if return_posterior_samples:
        return (
            eval_results,
            {"training": time_training, "inference": time_inference},
            posterior_samples_numpy,
        )
    elif return_posterior:
        likelihood = make_jax_likelihood(net_data_SM, net_theta_SM, return_conditional_dist=True)
        theta_min_max_affine_transform = convert_transform(get_transform(theta_scaler), "numpyro")
        samples_min_max_affine_transform = convert_transform(get_transform(samples_scaler), "numpyro")
        likelihood = TransformedConditionalDistribution(
            likelihood,
            transform=samples_min_max_affine_transform.inv,
            conditioned_var_transform=theta_min_max_affine_transform.inv,
        )
        prior = convert_dist(prior, "numpyro")
        posterior = JointDistribution(prior, likelihood).condition_out_variable(
            jnp.array(x_obs.clone().detach().numpy())
        )
        return (
            eval_results,
            {"training": time_training, "inference": time_inference},
            posterior
        )
    else:
        return eval_results, {"training": time_training, "inference": time_inference}


if __name__ == "__main__":
    num_observation = 2
    eval_results, _, posterior_samples = smnle(
        "two_moons",
        # technique="SM",
        technique="SSM",
        num_observation=num_observation,
        return_posterior_samples=True,
        use_orig_mcmc_impl=False,
        epochs=50,
        num_posterior_samples=50,
        batch_size=1000,
        n_samples_training=1000,
        mcmc_num_warmup_steps=10,
        cuda=True,
    )

    # import pickle
    # with open('posterior_samples.pkl', 'wb') as f:
    #     pickle.dump(posterior_samples, f)

    # np.save("./exchange_mcmc_trace{}".format(num_observation), posterior_samples)
