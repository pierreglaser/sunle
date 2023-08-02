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
from abcpy.statisticslearning import \
    ExponentialFamilyScoreMatching as ExpFamStatistics
from jax import jit, random
from numpyro import distributions as np_distributions
from numpyro.distributions import transforms as np_transforms
from pyro.distributions import transforms as pyro_transforms
from pyro.distributions.transforms import AffineTransform as pAffineTransform
from pyro.distributions.transforms import identity_transform as pIdentityTransform
from sbi import inference as inference
from unle.samplers.distributions import DoublyIntractableLogDensity, maybe_wrap
from unle.samplers.inference_algorithms.mcmc.base import (
    MCMCAlgorithmFactory, MCMCConfig)
from unle.samplers.kernels.mala import MALAConfig, MALAKernelFactory
from unle.samplers.kernels.rwmh import RWConfig, RWKernel, RWKernelFactory
from unle.samplers.kernels.savm import SAVMConfig, SAVMKernelFactory
from smnle.jax_torch_interop import make_jax_likelihood
from sbibm_unle_extra.pyro_to_numpyro import convert_dist, convert_transform
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.distributions.transformed_distribution import \
    TransformedDistribution

from src.exchange_mcmc import (exchange_MCMC_with_SM_statistics,
                               uniform_prior_theta)
from src.functions import (DummyScaler, LogLike,
                           generate_training_samples_ABC_model, plot_losses,
                           save_dict_to_json, scale_samples)
from src.networks import (create_PEN_architecture, createDefaultNN,
                          createDefaultNNWithDerivatives)
from src.parsers import parser_generate_obs, train_net_batch_parser
from src.training_routines import FP_training_routine
from src.utils_arma_example import (ARMAmodel, ar2_log_lik_for_mcmc,
                                    ma2_log_lik_for_mcmc)
from src.utils_beta_example import generate_beta_training_samples
from src.utils_gamma_example import generate_gamma_training_samples
from src.utils_gaussian_example import generate_gaussian_training_samples
from src.utils_Lorenz95_example import StochLorenz95

# for some reason it does not see my files if I don't put this
sys.path.append(os.getcwd())


os.environ["QT_QPA_PLATFORM"] = "offscreen"

# start_observation_index = args.start_observation
# n_observations = args.n_observations
# results_folder = args.root_folder

# n_samples_true_MCMC = 20000
# burnin_true_MCMC = 20000
# cores = 1

cuda = False
lam = 0

seed = 1

sys.path.append(os.getcwd())


technique = "SSM"


epochs = 500
no_scheduler = False
# results_folder = args.root_folder

# nets_folder = args.nets_folder
datasets_folder = "observations"

noise_sliced = "radermacher"

var_red_sliced = True
batch_norm_last_layer = True
affine_batch_norm = False
SM_lr = 0.001
FP_lr = 0.001
SM_lr_theta = 0.001

batch_size = 1000
early_stopping = True
update_batchnorm_running_means_before_eval = True
momentum = 0.9

print(early_stopping, update_batchnorm_running_means_before_eval, momentum)

epochs_before_early_stopping = 200
epochs_test_interval = 10
use_MPI = False
generate_data_only = False
save_net_at_each_epoch = False

technique = "SSM"
epochs = 500
SM_lr_theta = 0.001
bn_mom = 0.9
epochs_before_early_stopping = 200
epochs_test_interval = 10

args_dict = {}

n_samples_training = 10**4
n_samples_evaluation = 10**4


# from gaussian import GaussianTask
from gaussian import GaussianTask
from sbibm.tasks import get_task

torch.manual_seed(5)

# t = GaussianTask()
# x_obs = torch.from_numpy(np.array([10.860253,   3.0030434,  3.296743,   1.3824368,  8.1935,    -2.9345102, 11.283547,   2.4779046,  6.2736707,  4.276395 ]))

#
# model = "gaussian"
model = "two_moons"
model = "slcp"
model = "gaussian_linear_uniform"

t = get_task(model)
x_obs = t.get_observation(1)[0]

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


# print(theta_vect.min(0))
# print(theta_vect.max(0))


# mu_bounds = [-10, 10]
# sigma_bounds = [1, 10]
# lower_bounds = np.array([mu_bounds[0], sigma_bounds[0]])
# upper_bounds = np.array([mu_bounds[1], sigma_bounds[1]])

# theta_vect, samples_matrix = generate_gaussian_training_samples(
#     n_theta=n_samples_training,
#     size_iid_samples=10,
#     seed=seed,
#     mu_bounds=mu_bounds,
#     sigma_bounds=sigma_bounds,
# )
# 
# lower_bound = upper_bound = None
# 
# # generate test data for using early stopping in learning the statistics with SM
# theta_vect_test, samples_matrix_test = generate_gaussian_training_samples(
#     n_theta=n_samples_evaluation,
#     size_iid_samples=10,
#     mu_bounds=mu_bounds,
#     sigma_bounds=sigma_bounds,
# )






# update the n samples with the actual ones (if we loaded them from saved datasets).
# args_dict["n_samples_training"] = theta_vect.shape[0]
# args_dict["n_samples_evaluation"] = theta_vect_test.shape[0]
# args_dict["scale_samples"] = str(type(scale_samples_flag))
# args_dict["scale_parameters"] = str(type(scale_parameters_flag))

# if generate_data_only:
#     print("Generating data has finished")
#     exit()

# define network architectures:
nonlinearity = torch.nn.Softplus
# nonlinearity = torch.nn.Tanhshrink
# net_data_SM_architecture = createDefaultNN(10, 3, [30, 50, 50, 20], nonlinearity=nonlinearity())
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
elif  model == "two_moons":
    SM_lr = 0.01
    SM_lr_theta = 0.01
    net_data_SM_architecture = createDefaultNNWithDerivatives(
        2, 50, [30, 50, 50, 20], nonlinearity=nonlinearity
    )
    # net_data_SM_architecture = createDefaultNN(10, 3, [15, 15, 5], nonlinearity=nonlinearity())
    # net_theta_SM_architecture = createDefaultNN(2, 2, [5, 5], nonlinearity=nonlinearity())
    net_theta_SM_architecture = createDefaultNN(
        2,
        49,
        [15, 30, 30, 15],
        nonlinearity=nonlinearity(),
        batch_norm_last_layer=batch_norm_last_layer,
        affine_batch_norm=affine_batch_norm,
        batch_norm_last_layer_momentum=momentum,
    )
    epochs = 500
    epochs_before_early_stopping = 500
    
    # net_FP_architecture = createDefaultNN(10, 2, [30, 50, 50, 20], nonlinearity=torch.nn.ReLU())
    # net_FP_architecture = createDefaultNN(10, 2, [15, 15, 5], nonlinearity=nonlinearity())
elif model == "slcp":
    SM_lr = 0.01
    SM_lr_theta = 0.01
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
    epochs = 2000
    epochs_before_early_stopping = 2000
    # net_FP_architecture = createDefaultNN(10, 2, [30, 50, 50, 20], nonlinearity=torch.nn.ReLU())
    # net_FP_architecture = createDefaultNN(10, 2, [15, 15, 5], nonlinearity=nonlinearity())
elif model == "gaussian_linear_uniform":
    SM_lr = 0.01
    SM_lr_theta = 0.01
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
else:
    raise ValueError


if seed is not None:
    torch.manual_seed(seed)
# define networks
net_data_SM = net_data_SM_architecture()
net_theta_SM = net_theta_SM_architecture()

# convert the simulations and parameters to numpy if they are torch objects:
theta_vect = theta_vect.numpy() if isinstance(theta_vect, torch.Tensor) else theta_vect
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



from sklearn.preprocessing import MinMaxScaler


from sklearn.base import BaseEstimator, TransformerMixin

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
def get_transform(min_max_scaler: MinMaxScaler) -> pAffineTransform:
    # if isinstance(min_max_scaler, IdentityTransformer):
    #     return pIdentityTransform
    # else:
    return pAffineTransform(
        loc=torch.from_numpy(min_max_scaler.min_),
        scale=torch.from_numpy(min_max_scaler.scale_),
    )


theta_scaler = MinMaxScaler().fit(theta_vect)
# theta_scaler = IdentityTransformer()

theta_vect_transformed = theta_scaler.transform(theta_vect)
theta_vect_test_transformed = theta_scaler.transform(theta_vect_test)

samples_scaler = MinMaxScaler().fit(samples_matrix)
# samples_scaler = IdentityTransformer()
samples_matrix_transformed = samples_scaler.transform(samples_matrix)
samples_matrix_test_transformed = samples_scaler.transform(samples_matrix_test)



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
    scale_samples=False,
    scale_parameters=False,
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

loss_list = statistics_learning.train_losses
test_loss_list = statistics_learning.test_losses

scaler_data = statistics_learning.get_simulations_scaler()
scaler_theta = statistics_learning.get_parameters_scaler()

# save_net(nets_folder + "net_theta_SM.pth", net_theta_SM)
# save_net(nets_folder + "net_data_SM.pth", net_data_SM)
# pickle.dump(scaler_data, open(nets_folder + "scaler_data_SM.pkl", "wb"))
# pickle.dump(scaler_theta, open(nets_folder + "scaler_theta_SM.pkl", "wb"))

args_dict["scaler_data"] = str(type(scaler_data))
args_dict["scaler_theta"] = str(type(scaler_theta))


scaler_data_SM = scaler_data
scaler_theta_SM = scaler_theta





seed = 1
np.random.seed(seed)


use_orig_mcmc_impl = False
if use_orig_mcmc_impl:
    initial_theta_exchange_MCMC = np.array([0, 5.5])
    proposal_size_exchange_MCMC = 2 * np.array([1, 0.5])

    theta_dim = 2
    param_names = [r"$\mu$", r"$\sigma$"]

    print(f"\nPerforming exchange MCMC inference with {technique}.")

    start = time()

    obs_index = 1

    # model = args.model
    n_samples = 1000
    burnin_exchange_MCMC = 1000
    aux_MCMC_inner_steps_exchange_MCMC = 100
    aux_MCMC_proposal_size_exchange_MCMC = 0.1

    debug_level = logging.INFO

    tuning_window_exchange_MCMC = 100
    propose_new_theta_exchange_MCMC = "transformation"
    bridging_exch_MCMC = 0

    mu_bounds = [-10, 10]
    sigma_bounds = [1, 10]
    lower_bounds = np.array([mu_bounds[0], sigma_bounds[0]])
    upper_bounds = np.array([mu_bounds[1], sigma_bounds[1]])
    trace_exchange = exchange_MCMC_with_SM_statistics(x_obs, initial_theta_exchange_MCMC,
                                                      lambda x: uniform_prior_theta(x, lower_bounds,
                                                                                    upper_bounds),
                                                      net_data_SM, net_theta_SM, scaler_data_SM,
                                                      scaler_theta_SM, propose_new_theta_exchange_MCMC,
                                                      T=n_samples, burn_in=burnin_exchange_MCMC,
                                                      tuning_window_size=tuning_window_exchange_MCMC,
                                                      aux_MCMC_inner_steps=aux_MCMC_inner_steps_exchange_MCMC,
                                                      aux_MCMC_proposal_size=aux_MCMC_proposal_size_exchange_MCMC,
                                                      K=bridging_exch_MCMC,
                                                      seed=seed,
                                                      debug_level=debug_level,
                                                      lower_bounds_theta=lower_bounds,
                                                      upper_bounds_theta=upper_bounds,
                                                      sigma=proposal_size_exchange_MCMC)

    trace_exchange_burned_in = trace_exchange[burnin_exchange_MCMC:]
    np.save("./exchange_mcmc_trace{}".format(obs_index), trace_exchange_burned_in)


else:
    # smnle_theta_scaler = statistics_learning.get_parameters_scaler()
    # smnle_samples_scaler = statistics_learning.get_simulations_scaler()
    # assert smnle_theta_scaler is not None
    # assert smnle_samples_scaler is not None
    
    theta_min_max_affine_transform = get_transform(theta_scaler)
    samples_min_max_affine_transform = get_transform(samples_scaler)
    
    # theta_min_max_affine_transform = pAffineTransform(
    #     loc=torch.from_numpy(smnle_theta_scaler.min_),
    #     scale=torch.from_numpy(smnle_theta_scaler.scale_),
    # )
    # 
    # samples_min_max_affine_transform = pAffineTransform(
    #     loc=torch.from_numpy(smnle_samples_scaler.min_),
    #     scale=torch.from_numpy(smnle_samples_scaler.scale_),
    # )
    
    
    jax_log_likelihood = make_jax_likelihood(net_data_SM, net_theta_SM)
    transformed_prior = TransformedDistribution(prior, theta_min_max_affine_transform)
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
                    aux_var_num_inner_steps=200,
                    base_var_kernel_factory=RWKernelFactory(
                        config=RWConfig(
                            0.1, jnp.ones((transformed_jax_prior.event_shape[0],))
                        )
                    ),
                    aux_var_init_strategy="x_obs",
                )
            ),
            num_samples=3000,
            num_chains=10,
            thinning_factor=5,
            target_accept_rate=0.5,
            num_warmup_steps=3000,
            adapt_mass_matrix=False,
            adapt_step_size=True,
            progress_bar=True,
            init_using_log_l_mode=False,
        )
    )
    
    alg = config.build_algorithm(jax_posterior_log_prob)
    # alg = alg.init_from_particles(theta0_vals)
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    
    # initial_theta_exchange_MCMC = np.array([[0, 5.5] for _ in range(alg.config.num_chains)])
    alg = alg.init(subkey, transformed_jax_prior)
    
    
    key, subkey = random.split(key)
    alg, results = jit(type(alg).run)(alg, subkey)
    
    posterior_samples_torch = torch.from_numpy(np.array(results.samples.xs)).float()
    posterior_samples_torch = theta_min_max_affine_transform.inv(posterior_samples_torch)
    
    posterior_samples_numpy = np.array(posterior_samples_torch)
    # transform back
    # posterior_samples_numpy = theta_min_max_affine_transform.inv(posterior_samples_numpy)
    
    np.save("./exchange_mcmc_trace{}".format(1), posterior_samples_numpy)
