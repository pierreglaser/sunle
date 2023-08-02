import datetime
import random as python_random
import time
from typing import Literal, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
from flax import struct
from jax import Array, random
from jax.random import KeyArray
from sbibm.tasks.task import Task
from sbibm_unle_extra.tasks import JaxTask, get_task
from torch.types import Number
from unle.distributions.auto_tractable import AutoTractableConditionalDistribution
from unle.ebm.base import TrainerResults
from unle.samplers.inference_algorithms.base import InferenceAlgorithm
from unle.unle import UNLE


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU.
    np.random.seed(seed)
    python_random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore


class Config(struct.PyTreeNode):
    num_samples: Tuple[int]
    ebm_width: int
    ebm_depth: int
    task_name: str
    num_observation: int
    inference_sampler: str
    inference_num_warmup_steps: int
    max_iter: int = 500
    num_frozen_steps: int = 50
    num_mala_steps: int = 50
    num_particles: int = 1000
    use_warm_start: bool = True
    learning_rate: float = 1e-3
    weight_decay: float = 1e-1
    noise_injection_val: float = 1e-3
    batch_size: Optional[int] = None
    num_smc_steps: int = 5
    ess_threshold: float = 0.8
    exchange_mcmc_inner_sampler_num_steps: int = 100
    lznet_width = 100
    lznet_depth = 4
    lznet_z_score_output: bool = True
    frac_test_samples: float = 0.15
    use_data_from_past_rounds: bool = struct.field(pytree_node=False, default=True)
    normalize_data: bool = struct.field(pytree_node=False, default=True)
    n_sigma: float = struct.field(pytree_node=False, default=3.0)
    correction_net_width: int = 200
    correction_net_depth: int = 3
    correction_net_max_iter: int = 200
    estimate_log_normalizer: bool = struct.field(pytree_node=False, default=False)
    ebm_model_type: Literal["likelihood", "joint_tilted"] = struct.field(
        pytree_node=False, default="joint_tilted"
    )
    num_posterior_samples: int = 1000


class MetricResults(NamedTuple):
    mmd: Number
    c2st: Number


class TrainEvalTresults(NamedTuple):
    train_results: "Results"
    eval_results: Optional[MetricResults] = None


def compute_posterior_comparison_metrics(
    posterior_samples: Array, task: Task, num_observation: int
) -> MetricResults:
    from sbibm.metrics import c2st, mmd

    reference_posterior_samples = task.get_reference_posterior_samples(num_observation)
    unle_posterior_samples = torch.from_numpy(np.array(posterior_samples))

    mmd_val = mmd(reference_posterior_samples, unle_posterior_samples).item()
    c2st_val = c2st(reference_posterior_samples, unle_posterior_samples)[0].item()
    return MetricResults(mmd_val, c2st_val)


class SingleRoundResults(NamedTuple):
    round_no: int
    config: Config
    posterior: AutoTractableConditionalDistribution
    train_results: TrainerResults
    x_obs: Array
    inference_state: Optional[InferenceAlgorithm]
    simulation_time: float = 0.0
    inference_time: float = 0.0
    lznet_training_time: float = 0.0


class Results(NamedTuple):
    unle: UNLE
    config: Config
    posterior: AutoTractableConditionalDistribution
    posterior_samples: Array
    single_round_results: Tuple[SingleRoundResults]
    total_time: float = 0.0

    def get_posterior_samples(self, round_no):
        num_rounds = len(self.config.num_samples)
        if round_no == num_rounds - 1:
            return self.posterior_samples
        else:
            return self.unle.all_thetas[round_no + 1]


def run(
    task: Union[Task, str],
    num_samples: Union[int, Tuple[int, ...]],
    num_observation: int,
    max_iter: int = 3000,
    learning_rate: float = 0.001,
    weight_decay: float = 0.1,
    num_smc_steps: int = 30,
    num_mala_steps: int = 3,
    num_particles: int = 1000,
    ess_threshold: float = 0.8,
    use_warm_start=True,
    num_posterior_samples=10000,
    random_seed=41,
    sampler: Literal["mala", "adaptive_mala", "smc", "ula"] = "smc",
    evaluate_posterior: bool = False,
    init_proposal: str = "prior",
    use_nuts=False,
    noise_injection_val: float = 0.001,
    proposal: Literal["noise", "data", "prior+noise"] = "noise",
    inference_sampler: Literal["mala", "smc", "exchange_mcmc"] = "smc",
    batch_size: Optional[int] = 1000,
    use_data_from_past_rounds: bool = True,
    ebm_model_type: Literal["joint_tilted", "likelihood"] = "joint_tilted",
    fit_in_unconstrained_space: bool = False,
    estimate_loss: Union[bool, Literal["auto"]] = "auto",
    inference_proposal: Literal["prior", "noise"] = "noise",
    results: Optional["Results"] = None,
    ebm_depth: int = 4,
    ebm_width: int = 50,
    exchange_mcmc_inner_sampler_num_steps: int = 100,
    inference_num_warmup_steps: int = 2000,
    n_sigma: float = 3.0,
    training_num_frozen_steps: int = 50,
    estimate_log_normalizer=False,
    num_rounds: Optional[int] = None,
    calibration_net_max_iter: int = 200,
):
    # some args validation
    if not fit_in_unconstrained_space:
        assert inference_proposal == "prior"
        assert proposal in ("data", "prior+noise")

    if isinstance(task, str):
        task = get_task(task)
    assert isinstance(task, Task)
    jax_task = JaxTask(task)

    if isinstance(num_samples, int):
        if num_rounds in (None, 1):
            num_samples = (num_samples,)
        elif isinstance(num_rounds, int) and num_rounds > 1:
            num_samples = tuple(num_samples // num_rounds for _ in range(num_rounds))
        else:
            raise ValueError(
                f"num_rounds must be a positive integer or None, got {num_rounds}"
            )
    else:
        assert isinstance(num_samples, tuple)
        num_rounds = len(num_samples)
        assert num_rounds > 0

    seed_everything(random_seed)
    key = random.PRNGKey(random_seed)

    if ebm_model_type == "likelihood":
        if not estimate_log_normalizer:
            assert (
                inference_sampler == "exchange_mcmc"
            ), "only exchange_mcmc is supported for this model type"

    print(f"using a network of width {ebm_width} and depth {ebm_depth}")

    c = Config(
        num_samples=num_samples,
        ebm_width=ebm_width,
        ebm_depth=ebm_depth,
        num_observation=num_observation,
        task_name=task.name,
        inference_sampler="mcmc"
        if inference_sampler in ("exchange_mcmc", "mala")
        else "smc",
        max_iter=max_iter,
        num_frozen_steps=training_num_frozen_steps,
        num_mala_steps=num_mala_steps,
        num_particles=num_particles,
        use_warm_start=use_warm_start,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        noise_injection_val=noise_injection_val,
        batch_size=batch_size,
        num_smc_steps=num_smc_steps,
        ess_threshold=ess_threshold,
        inference_num_warmup_steps=inference_num_warmup_steps,
        exchange_mcmc_inner_sampler_num_steps=exchange_mcmc_inner_sampler_num_steps,
        use_data_from_past_rounds=use_data_from_past_rounds,
        n_sigma=n_sigma,
        correction_net_max_iter=calibration_net_max_iter,
        estimate_log_normalizer=estimate_log_normalizer,
        ebm_model_type=ebm_model_type,
        num_posterior_samples=num_posterior_samples,
    )
    train_results = train_unle(jax_task, c, key=key)

    if evaluate_posterior:
        eval_results = compute_posterior_comparison_metrics(
            train_results.posterior_samples, task, num_observation
        )
    else:
        eval_results = None

    return TrainEvalTresults(train_results, eval_results)


def train_unle(task: JaxTask, config: Config, key: KeyArray) -> Results:
    t0_unle = time.time()
    unle = UNLE.create(n_sigma=config.n_sigma)
    x_obs = task.get_observation(config.num_observation)[0]
    proposal_dist = prior_dist = task.get_prior_dist()
    single_round_results = []
    current_posterior = None
    simulator = task.get_simulator()

    for round_no, num_samples in enumerate(config.num_samples):
        # Simulate -----------------------------------------------------------------
        # Sample from the proposal distribution
        key, subkey = random.split(key)
        if round_no == 0:
            if hasattr(simulator, "get_large_precomputed_dataset"):
                print("loading cached training samples for round 0")
                parameters, observations = simulator.get_large_precomputed_dataset(
                    num_samples
                )
            else:
                parameters = proposal_dist.sample(subkey, (num_samples,))
        else:
            assert isinstance(proposal_dist, AutoTractableConditionalDistribution)
            parameters, proposal_dist = proposal_dist.sample(
                subkey,
                (num_samples,),
                return_updated_dist=True,
            )
            # stateful posterior distribution
            unle = unle.replace(posterior=proposal_dist)

        # Simulate from the simulator
        t0_simulation = time.time()
        observations = simulator(parameters)
        simulation_time = time.time() - t0_simulation
        print(
            "generating data took time: ",
            str(datetime.timedelta(seconds=int(simulation_time))),
        )

        unle = unle.append_simulations(
            parameters, observations, proposal_dist, config.n_sigma
        )

        # Train Likelihood ------------------------------------------------------------

        key, subkey = random.split(key)
        training_results, unle = unle.train_likelihood(
            subkey,
            config.ebm_model_type,
            True,
            config.ebm_width,
            config.ebm_depth,
            config.max_iter,
            config.num_frozen_steps,
            config.num_mala_steps,
            config.num_particles,
            config.use_warm_start,
            config.learning_rate,
            config.weight_decay,
            config.noise_injection_val,
            config.batch_size,
            config.num_smc_steps,
            config.ess_threshold,
            config.correction_net_width,
            config.correction_net_depth,
            config.correction_net_max_iter,
        )

        # Train Log Z Net  ---------------------------------------------------------

        if (
            unle.get_likelihood().has_theta_dependent_normalizer
            and config.estimate_log_normalizer
        ):
            t0_lznet_training = time.time()
            unle = unle.train_lznet(
                subkey,
                config.lznet_width,
                config.lznet_depth,
                config.lznet_z_score_output,
            )
            lznet_training_time = time.time() - t0_lznet_training
            print(
                "training LZNet took time: ",
                str(datetime.timedelta(seconds=int(lznet_training_time))),
            )
        else:
            lznet_training_time = 0.0

        # Build Posterior -----------------------------------------------------------
        unle, current_posterior = unle.build_posterior(
            prior_dist,
            x_obs,
            config.inference_sampler,
            config.inference_num_warmup_steps,
            config.exchange_mcmc_inner_sampler_num_steps,
        )

        # Set New Proposal ---------------------------------------------------------
        proposal_dist = current_posterior

        # Experiment-specific results management ----------------------------------

        this_round_results = SingleRoundResults(
            round_no=round_no,
            config=config,
            posterior=current_posterior,
            train_results=training_results,
            x_obs=x_obs,
            inference_state=current_posterior.sampling_alg,
            simulation_time=simulation_time,
            inference_time=0.0,  # TODO JZ
            lznet_training_time=lznet_training_time,
        )

        single_round_results.append(this_round_results)
        if this_round_results.train_results.best_state.has_nan:
            break

        unle = unle.replace(round_no=round_no + 1)

    assert current_posterior is not None

    key, subkey = random.split(key)
    posterior_samples, posterior = current_posterior.sample(
        subkey,
        (config.num_posterior_samples,),
        return_updated_dist=True,
    )

    results = Results(
        unle=unle,
        config=config,
        posterior=posterior,
        posterior_samples=posterior_samples,
        single_round_results=tuple(single_round_results),
        total_time=time.time() - t0_unle,
    )
    print(f"unle completed in {results.total_time} seconds")

    return results
