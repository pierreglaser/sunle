import logging
import os
import pickle
import sys
from time import sleep, time

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as npdist
import pymc3 as pm
import pyro.distributions as pdist
import theano.tensor as tt
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

from smnle.src.exchange_mcmc import (exchange_MCMC_with_SM_statistics,
                               uniform_prior_theta)
from smnle.src.functions import (DummyScaler, LogLike,
                           generate_training_samples_ABC_model, plot_losses,
                           save_dict_to_json, scale_samples)
from smnle.src.networks import (create_PEN_architecture, createDefaultNN,
                          createDefaultNNWithDerivatives)
from smnle.src.parsers import parser_generate_obs, train_net_batch_parser
from smnle.src.training_routines import FP_training_routine
from smnle.src.utils_arma_example import (ARMAmodel, ar2_log_lik_for_mcmc,
                                    ma2_log_lik_for_mcmc)
from smnle.src.utils_beta_example import generate_beta_training_samples
from smnle.src.utils_gamma_example import generate_gamma_training_samples
from smnle.src.utils_gaussian_example import generate_gaussian_training_samples
from smnle.src.utils_Lorenz95_example import StochLorenz95

# for some reason it does not see my files if I don't put this
sys.path.append(os.getcwd())


os.environ["QT_QPA_PLATFORM"] = "offscreen"

# start_observation_index = args.start_observation
# n_observations = args.n_observations
# sleep_time = args.sleep
# results_folder = args.root_folder

default_root_folder = {
    "gaussian": "results/gaussian/",
    "gamma": "results/gamma/",
    "beta": "results/beta/",
    "AR2": "results/AR2/",
    "MA2": "results/MA2/",
    "fullLorenz95": "results/fullLorenz95/",
    "fullLorenz95smaller": "results/fullLorenz95smaller/",
}

model = "gaussia"
results_folder = default_root_folder["gaussian"]

n_samples_true_MCMC = 20000
burnin_true_MCMC = 20000
cores = 1

seed = 1

save_true_MCMC_trace = True
save_observation = True

mu_bounds = [-10, 10]
sigma_bounds = [1, 10]
theta_vect, samples_matrix = generate_gaussian_training_samples(
    n_theta=1,
    size_iid_samples=10,
    seed=seed,
    mu_bounds=mu_bounds,
    sigma_bounds=sigma_bounds,
)

obs_index = 0
print("Observation {}".format(obs_index + 1))
if isinstance(samples_matrix, np.ndarray):
    x_obs = samples_matrix[obs_index]
else:
    x_obs = samples_matrix[obs_index].numpy()
if isinstance(theta_vect, np.ndarray):
    theta_obs = theta_vect[obs_index]
else:
    theta_obs = theta_vect[obs_index].numpy()

# np.save("theta_obs{}".format(obs_index + 1), theta_obs)
# np.save("x_obs{}".format(obs_index + 1), x_obs)


sys.path.append(os.getcwd())


technique = "SSM"
model = "gaussian"

save_train_data = False
load_train_data = False
sleep_time = 0

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

model = "gaussian"
technique = "SSM"
epochs = 50
SM_lr = 0.001
SM_lr_theta = 0.001
bn_mom = 0.9
epochs_before_early_stopping = 10
epochs_test_interval = 10
save_train_data = True


# checks
if model not in (
    "gaussian",
    "beta",
    "gamma",
    "MA2",
    "AR2",
    "fullLorenz95",
    "fullLorenz95smaller",
) or technique not in ("SM", "SSM", "FP"):
    raise NotImplementedError

backend = BackendMPI() if use_MPI else BackendDummy()

if generate_data_only:
    print("Generate data only, no train.")
else:
    print("{} model with {}.".format(model, technique))
# set up the default root folder and other values
default_root_folder = {"gaussian": "results/gaussian/"}

if results_folder is None:
    results_folder = default_root_folder[model]

nets_folder = "net-SM" if technique == "SM" else "net-FP"

results_folder = results_folder + "/"
nets_folder = results_folder + nets_folder + "/"
datasets_folder = results_folder + datasets_folder + "/"

if SM_lr is None:
    SM_lr = 0.001
if SM_lr_theta is None:
    SM_lr_theta = 0.001

if sleep_time > 0:
    print("Wait for {} minutes...".format(sleep_time))
    sleep(60 * sleep_time)
    print("Done waiting!")

seed = 42

cuda = True
torch.set_num_threads(4)

n_samples_training = 10**4
n_samples_evaluation = 10**4

save_net_flag = True
lam = 0

args_dict = {}
# add other arguments to the config dicts
args_dict["seed"] = seed
args_dict["n_samples_training"] = n_samples_training
args_dict["n_samples_evaluation"] = n_samples_evaluation
args_dict["lr_FP_actual"] = FP_lr
args_dict["lr_data_actual"] = SM_lr
args_dict["lr_theta_actual"] = SM_lr_theta
args_dict["batch_size"] = batch_size
args_dict["save_net"] = save_net_flag
args_dict["cuda"] = cuda

mu_bounds = [-10, 10]
sigma_bounds = [1, 10]

args_dict["mu_bounds"] = mu_bounds
args_dict["sigma_bounds"] = sigma_bounds
start = time()
# generate training data
theta_vect, samples_matrix = generate_gaussian_training_samples(
    n_theta=n_samples_training,
    size_iid_samples=10,
    seed=seed,
    mu_bounds=mu_bounds,
    sigma_bounds=sigma_bounds,
)
print("Data generation took {:.4f} seconds".format(time() - start))

scaler_data_FP = MinMaxScaler().fit(
    samples_matrix.reshape(-1, samples_matrix.shape[-1])
)
lower_bound = upper_bound = None
scale_samples_flag = True
scale_parameters_flag = True

# generate test data for using early stopping in learning the statistics with SM
theta_vect_test, samples_matrix_test = generate_gaussian_training_samples(
    n_theta=n_samples_evaluation,
    size_iid_samples=10,
    mu_bounds=mu_bounds,
    sigma_bounds=sigma_bounds,
)


# update the n samples with the actual ones (if we loaded them from saved datasets).
args_dict["n_samples_training"] = theta_vect.shape[0]
args_dict["n_samples_evaluation"] = theta_vect_test.shape[0]
args_dict["scale_samples"] = str(type(scale_samples_flag))
args_dict["scale_parameters"] = str(type(scale_parameters_flag))

if generate_data_only:
    print("Generating data has finished")
    exit()

# define network architectures:
nonlinearity = torch.nn.Softplus
# nonlinearity = torch.nn.Tanhshrink
# net_data_SM_architecture = createDefaultNN(10, 3, [30, 50, 50, 20], nonlinearity=nonlinearity())
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

statistics_learning = ExpFamStatistics(
    # backend and model are not used
    model=None,
    statistics_calc=Identity(),
    backend=BackendDummy(),
    simulations_net=net_data_SM,
    parameters_net=net_theta_SM,
    parameters=theta_vect,
    simulations=samples_matrix,
    parameters_val=theta_vect_test,
    simulations_val=samples_matrix_test,
    scale_samples=scale_samples_flag,
    scale_parameters=scale_parameters_flag,
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


mu_bounds = [-10, 10]
sigma_bounds = [1, 10]
lower_bounds = np.array([mu_bounds[0], sigma_bounds[0]])
upper_bounds = np.array([mu_bounds[1], sigma_bounds[1]])

initial_theta_exchange_MCMC = np.array([0, 5.5])
proposal_size_exchange_MCMC = 2 * np.array([1, 0.5])

theta_dim = 2
param_names = [r"$\mu$", r"$\sigma$"]

print(f"\nPerforming exchange MCMC inference with {technique}.")

start = time()

obs_index = 1


# model = args.model
sleep_time = 0.0
n_samples = 100
burnin_exchange_MCMC = 100
aux_MCMC_inner_steps_exchange_MCMC = 10
aux_MCMC_proposal_size_exchange_MCMC = 0.1

debug_level = logging.INFO

tuning_window_exchange_MCMC = 100
propose_new_theta_exchange_MCMC = "transformation"
bridging_exch_MCMC = 0


seed = 1
np.random.seed(seed)


use_orig_mcmc_impl = True
if use_orig_mcmc_impl:
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
    smnle_theta_scaler = statistics_learning.get_parameters_scaler()
    smnle_samples_scaler = statistics_learning.get_simulations_scaler()
    assert smnle_theta_scaler is not None
    assert smnle_samples_scaler is not None
    
    
    theta_min_max_affine_transform = pAffineTransform(
        loc=torch.from_numpy(smnle_theta_scaler.min_),
        scale=torch.from_numpy(smnle_theta_scaler.scale_),
    )
    
    samples_min_max_affine_transform = pAffineTransform(
        loc=torch.from_numpy(smnle_samples_scaler.min_),
        scale=torch.from_numpy(smnle_samples_scaler.scale_),
    )
    
    
    jax_log_likelihood = make_jax_likelihood(net_data_SM, net_theta_SM)
    
    
    # mu_bounds = [-10, 10]
    # sigma_bounds = [1, 10]
    
    prior = pdist.Uniform(
        low=torch.tensor([-10.0, 1.0]), high=torch.tensor([10.0, 10.0])
    ).to_event(1)
    # transformed_prior = TransformedDistribution(prior, torch_scaler_theta._transform)
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
                    aux_var_num_inner_steps=500,
                    base_var_kernel_factory=RWKernelFactory(
                        config=RWConfig(
                            0.1, jnp.ones((transformed_jax_prior.event_shape[0],))
                        )
                    ),
                    aux_var_init_strategy="x_obs",
                )
            ),
            num_samples=10000,
            num_chains=10,
            thinning_factor=1,
            target_accept_rate=0.5,
            num_warmup_steps=1000,
            adapt_mass_matrix=False,
            adapt_step_size=True,
            progress_bar=True,
            init_using_log_l_mode=False,
        )
    )
    
    # gives similar results
    # config = MCMCAlgorithmFactory(
    #     config=MCMCConfig(
    #         kernel_factory=SAVMKernelFactory(config=SAVMConfig(
    #             aux_var_kernel_factory=MALAKernelFactory(MALAConfig(0.1)),
    #             # aux_var_num_inner_steps=500,
    #             aux_var_num_inner_steps=100,
    #             base_var_kernel_factory=RWKernelFactory(config=RWConfig(0.01, jnp.ones((transformed_jax_prior.event_shape[0],)))),
    #             aux_var_init_strategy="x_obs",
    #         )),
    #         num_samples=1000,
    #         num_chains=10,
    #         thinning_factor=1,
    #         target_accept_rate=0.5,
    #         num_warmup_steps=10000,
    #         adapt_mass_matrix=False,
    #         adapt_step_size=False,
    #         progress_bar=True,
    #         init_using_log_l_mode=False
    #     )
    # )
    
    
    alg = config.build_algorithm(jax_posterior_log_prob)
    # alg = alg.init_from_particles(theta0_vals)
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    
    # alg = alg.init(subkey, transformed_jax_prior)
    initial_theta_exchange_MCMC = np.array([[0, 5.5] for _ in range(alg.config.num_chains)])
    
    alg = alg.init_from_particles(
        jnp.array(theta_min_max_affine_transform(initial_theta_exchange_MCMC))
    )
    
    
    key, subkey = random.split(key)
    alg, results = jit(type(alg).run)(alg, subkey)
    
    posterior_samples_torch = torch.from_numpy(np.array(results.samples.xs)).float()
    posterior_samples_torch = theta_min_max_affine_transform.inv(posterior_samples_torch)
    
    posterior_samples_numpy = np.array(posterior_samples_torch)
    # transform back
    # posterior_samples_numpy = theta_min_max_affine_transform.inv(posterior_samples_numpy)
    
    np.save("./exchange_mcmc_trace{}".format(obs_index), posterior_samples_numpy)
