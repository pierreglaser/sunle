from __future__ import annotations

from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from typing_extensions import TypeVarTuple, Unpack

if TYPE_CHECKING:
    from jax.random import KeyArray
    from unle.samplers.inference_algorithms.base import (
        InferenceAlgorithm,
        InferenceAlgorithmFactory,
        InferenceAlgorithmInfo,
    )
    from unle.samplers.particle_aproximation import ParticleApproximation

import collections
from time import time

import jax.numpy as jnp
import optax
from flax import struct
from flax.training import train_state
from jax import Array, jit, random
from jax._src.flatten_util import ravel_pytree
from jax.tree_util import tree_leaves, tree_map
from numpyro import distributions as np_distributions
from unle.samplers.inference_algorithms.mcmc.base import MCMCAlgorithm
from unle.typing import Numeric, PyTreeNode


def tree_any(function: Callable[[PyTreeNode], Numeric], tree: PyTreeNode) -> Numeric:
    mapped_tree = tree_map(function, tree)
    return jnp.any(jnp.array(tree_leaves(mapped_tree)))


class OptimizerConfig(NamedTuple):
    learning_rate: float = 0.01
    weight_decay: float = 0.005
    noise_injection_val: float = 0.02


class TrainingConfig(struct.PyTreeNode):
    optimizer: OptimizerConfig
    sampling_cfg_first_iter: InferenceAlgorithmFactory
    sampling_cfg: InferenceAlgorithmFactory
    sampling_init_dist: Union[
        Literal["data"], np_distributions.Distribution
    ] = struct.field(pytree_node=False)
    max_iter: int = struct.field(pytree_node=False)
    num_particles: int = struct.field(pytree_node=False, default=1000)
    use_warm_start: bool = struct.field(pytree_node=False, default=False)
    verbose: bool = struct.field(pytree_node=False, default=True)
    max_num_recordings: int = struct.field(pytree_node=False, default=100)
    batch_size: Optional[int] = struct.field(pytree_node=False, default=None)
    recording_enabled: bool = struct.field(pytree_node=False, default=False)


class TrainState(train_state.TrainState):
    training_alg: InferenceAlgorithm
    has_nan: bool
    opt_is_diverging: bool = False


class MiniTrainState(struct.PyTreeNode):
    params: PyTreeNode


class TrainingStats(struct.PyTreeNode):
    loss: Dict
    sampling: Optional[InferenceAlgorithmInfo]
    mmd: Numeric = 0
    grad_norm: Numeric = 0


class TrainerResults(struct.PyTreeNode):
    init_state: TrainState
    final_state: TrainState
    best_state: TrainState
    trajectory: Optional[TrainState]
    stats: Optional[TrainingStats]
    dataset: Optional[Array]
    config: TrainingConfig
    time: float = 0.0


TT = TypeVarTuple("TT")


class BaseEnergyFnWrapper(Generic[Unpack[TT]], struct.PyTreeNode):
    energy_fn: Callable[[Any, Unpack[TT]], Array] = struct.field(pytree_node=False)
    params: PyTreeNode

    def __call__(self, *args: Unpack[TT]) -> Any:
        return -self.energy_fn(self.params, *args)


T = TypeVar("T")


class Batch(Generic[T], struct.PyTreeNode):
    batch: T
    indices: Array


def maybe_print_info(state: TrainState, config: TrainingConfig, stats: TrainingStats):
    if state.step % max(config.max_iter // 20, 1) == 0 or state.has_nan:
        _iter_str = f"{state.step}/{config.max_iter}"

        if state.has_nan:
            print(f"iteration {_iter_str:<10}: algorithm encountered nans", flush=True)
        else:
            print(
                f"iteration {_iter_str:<10}: {stats.grad_norm:<10.3f}"
                f"unnormalized_train_log_l={stats.loss['unnormalized_train_log_l']:<10.3f} "  # noqa: E501
                f"unnormalized_test_log_l={stats.loss['unnormalized_test_log_l']:<10.3f} "  # noqa: E501
                f"train_log_l={stats.loss['train_log_l']:<10.3f}"
                f"test_log_l={stats.loss['test_log_l']:<10.3f}"
                f"ebm_log_l={stats.loss['ebm_samples_train_log_l']:<10.3f}",
                flush=True,
            )

    return state.step


def has_numerical_instabilities(state: TrainState) -> bool:
    if state.has_nan:
        print(f"iter_no {state.step} encountered NaNs")
        print(state)
        return True

    if state.opt_is_diverging:
        print(f"iter_no {state.step}: opt is diverging")
        print(state)
        return True
    return False


def flag_instabilities(state: TrainState, grads: PyTreeNode) -> TrainState:
    has_nan = tree_any(lambda x: jnp.any(jnp.isnan(x)), state)

    sum_grad_norms = jnp.sum(jnp.square(ravel_pytree(grads)[0]))
    opt_is_diverging = sum_grad_norms > 1e8

    state = state.replace(has_nan=has_nan, opt_is_diverging=opt_is_diverging)
    return state


class BaseTrainer(Generic[Unpack[TT]]):
    dim_z: int
    log_joint: BaseEnergyFnWrapper[Unpack[TT]]

    def __init__(self, dim_z: int):
        self.dim_z = dim_z

    @abstractmethod
    def compute_ebm_approx(
        self,
        alg: InferenceAlgorithm,
        params: PyTreeNode,
        key: KeyArray,
        true_samples: Batch[Tuple[Unpack[TT]]],
        config: TrainingConfig,
    ) -> Tuple[InferenceAlgorithm, ParticleApproximation]:
        raise NotImplementedError

    @abstractmethod
    def estimate_log_likelihood_gradient(
        self,
        params: PyTreeNode,
        true_samples: Batch[Tuple[Unpack[TT]]],
        ebm_samples: ParticleApproximation,
        noise_injection_val: float,
        key: KeyArray,
        log_joint: BaseEnergyFnWrapper[Unpack[TT]],
    ) -> PyTreeNode:
        raise NotImplementedError

    @abstractmethod
    def _initialize_sampling_alg(
        self,
        log_prob: BaseEnergyFnWrapper[Unpack[TT]],
        config: TrainingConfig,
        dataset: Tuple[Unpack[TT]],
        params: PyTreeNode,
        key: KeyArray,
        use_first_iter_cfg: bool = False,
    ) -> InferenceAlgorithm:
        raise NotImplementedError

    @abstractmethod
    def make_log_prob(self, energy_fn, params) -> BaseEnergyFnWrapper[Unpack[TT]]:
        raise NotImplementedError

    def initialize_training_state(
        self,
        energy_fn: Callable,
        params: PyTreeNode,
        dataset: Tuple[Unpack[TT]],
        config: TrainingConfig,
        key: KeyArray,
        use_first_iter_cfg: bool = False,
    ) -> TrainState:
        # 1. OPTIMIZER
        schedule_fn = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.optimizer.learning_rate,
            warmup_steps=max(0, min(50, config.max_iter // 2)),
            decay_steps=config.max_iter,
            end_value=config.optimizer.learning_rate / 50,
        )
        txs = optax.chain(
            optax.clip(5.0),
            optax.adamw(
                learning_rate=schedule_fn, weight_decay=config.optimizer.weight_decay
            ),
        )
        self.log_joint = self.make_log_prob(
            energy_fn=energy_fn,
            params=params,
        )
        opt_state = txs.init(params)

        # 2.a PARTICLE APPROXIMATION (gradient)
        key, key_init_particles = random.split(key)

        training_algs = self._initialize_sampling_alg(
            self.log_joint,
            config,
            dataset,
            params,
            key_init_particles,
            use_first_iter_cfg,
        )

        assert config.num_particles is not None
        state = TrainState(
            apply_fn=None,  # type: ignore
            tx=txs,
            params=params,
            opt_state=opt_state,
            step=0,
            training_alg=training_algs,
            has_nan=False,
            opt_is_diverging=False,
        )
        return state

    def _inject_post_first_iter_config(
        self,
        state: TrainState,
        config: TrainingConfig,
        datasets: Tuple[Unpack[TT]],
        key: KeyArray,
    ) -> TrainState:
        training_algs = self._initialize_sampling_alg(
            self.log_joint,
            config,
            datasets,
            state.params,
            key,
            use_first_iter_cfg=False,
        )
        state = state.replace(
            training_alg=state.training_alg.replace(config=training_algs.config)
        )
        if isinstance(training_algs, MCMCAlgorithm):
            assert isinstance(state.training_alg, MCMCAlgorithm)
            assert state.training_alg.single_chains is not None
            assert training_algs.single_chains is not None
            state = state.replace(
                training_alg=state.training_alg.replace(
                    single_chains=state.training_alg.single_chains.replace(
                        config=training_algs.single_chains.config
                    )
                )
            )
        return state

    def estimate_value_and_grad(
        self,
        params: PyTreeNode,
        noise_injection_val: float,
        ebm_samples: ParticleApproximation,
        key: KeyArray,
        dataset: Batch[Tuple[Unpack[TT]]],
        log_joint: BaseEnergyFnWrapper[Unpack[TT]],
    ) -> Tuple[PyTreeNode, TrainingStats]:
        key, subkey = random.split(key)
        grads = self.estimate_log_likelihood_gradient(
            params,
            dataset,
            ebm_samples,
            noise_injection_val,
            subkey,
            log_joint,
        )

        _keys = (
            "unnormalized_train_log_l",
            "unnormalized_test_log_l",
            "train_log_l",
            "test_log_l",
            "ebm_samples_train_log_l",
        )
        stats = TrainingStats(
            loss={k: 0.0 for k in _keys},
            sampling=None,
            grad_norm=jnp.sum(jnp.square(ravel_pytree(grads)[0])),
        )
        stats.loss["unnormalized_train_log_l"] = 0.0
        stats.loss["unnormalized_test_log_l"] = 0.0
        return grads, stats

    def get_batch(
        self, dataset: Tuple[Unpack[TT]], batch_size: Optional[int], key: KeyArray
    ) -> Batch[Tuple[Unpack[TT]]]:
        assert isinstance(dataset[0], jnp.ndarray)
        num_samples = len(dataset[0])
        if batch_size is not None:
            # XXX: important to randomize even when batch_size >= len(dataset)
            # due to non-randomization of the batch indices for ebm particles
            # in subselection happening later.
            # TODO: fix this
            batch_size = min(batch_size, num_samples)
            key, subkey = random.split(key)
            idxs = random.choice(subkey, num_samples, shape=(batch_size,))
            batched_dataset = tree_map(lambda x: x[idxs], dataset)
            return Batch(batched_dataset, idxs)
        else:
            return Batch(dataset, jnp.arange(num_samples))

    def train_step(
        self,
        state: TrainState,
        batch: Batch[Tuple[Unpack[TT]]],
        config: TrainingConfig,
        key: KeyArray,
        entire_datasets: Tuple[Unpack[TT]],
    ) -> Tuple[TrainState, Tuple[TrainState, TrainingStats]]:
        key, subkey = random.split(key)
        alg, results = self.compute_ebm_approx(
            state.training_alg,
            state.params,
            subkey,
            batch,
            config,
        )

        key, subkey = random.split(key)
        grads, stats = self.estimate_value_and_grad(
            params=state.params,
            noise_injection_val=config.optimizer.noise_injection_val,
            ebm_samples=results,
            key=subkey,
            dataset=batch,
            log_joint=self.log_joint,
        )

        updates, opt_state = state.tx.update(
            grads,
            state.opt_state,
            params=state.params,
        )

        params = optax.apply_updates(state.params, updates)

        if not config.use_warm_start:
            # reinitialize previous sampler state using new particles
            key, key_particles = random.split(key)
            alg = self._initialize_sampling_alg(
                self.log_joint,
                config,
                entire_datasets,
                params,
                key_particles,
            )

        new_state = state.replace(
            step=state.step + 1,
            params=params,
            opt_state=opt_state,
            training_alg=alg,
        )
        new_state = flag_instabilities(new_state, grads)
        return new_state, (new_state, stats)

    def train(
        self,
        energy_fn: Callable,
        params: PyTreeNode,
        dataset: Tuple[Unpack[TT]],
        config: TrainingConfig,
        key: KeyArray,
    ) -> TrainerResults:
        key, subkey = random.split(key)
        init_state = self.initialize_training_state(
            energy_fn,
            params,
            dataset,
            config,
            key=subkey,
            use_first_iter_cfg=True,
        )

        t0_init_training = time()

        best_state = init_state

        jitted_train_step = jit(self.train_step)

        n = 10

        last_n_stable_states = collections.deque(maxlen=n)

        outputs = []
        key, subkey = random.split(key)
        batch = self.get_batch(dataset, config.batch_size, subkey)
        key, subkey = random.split(key)

        print("first step...")
        state, output = jitted_train_step(
            init_state,
            batch,
            config,
            subkey,
            dataset,
        )
        print("....done.")
        state = cast(TrainState, state)

        key, subkey = random.split(key)
        state = self._inject_post_first_iter_config(state, config, dataset, subkey)

        last_n_stable_states.append(state)
        _, stats = output

        keys = random.split(key, num=config.max_iter - 1)

        for key in keys:
            if has_numerical_instabilities(state):
                break

            key, subkey = random.split(key)
            batch = self.get_batch(dataset, config.batch_size, subkey)

            key, subkey = random.split(key)
            state, output = jitted_train_step(state, batch, config, subkey, dataset)

            maybe_print_info(state, config, output[1])

            if state.step % max(config.max_iter // config.max_num_recordings, 1) == 0:
                state, stats = output
                if config.recording_enabled:
                    # whole trainstate is very heavy
                    outputs.append([state, stats])
                else:
                    outputs.append([MiniTrainState(state.params), stats])

            if state.step % 20 == 0:
                last_n_stable_states.append(state)

        trajectory = tree_map(lambda *args: jnp.stack(args), *outputs)

        if state.opt_is_diverging:
            print("optimisation was diverging, using latest stable state")
            best_state = last_n_stable_states.popleft()
            print(best_state.step)
        else:
            best_state = state

        final_state = state

        results = TrainerResults(
            init_state,
            final_state,
            best_state,
            trajectory[0],
            trajectory[1],
            # TODO: don't log the dataset in the results to limit
            #  memory usage and the size of the results, datasets.
            None,
            config,
            time=time() - t0_init_training,
        )
        import datetime

        print(
            "training ebm took time",
            str(datetime.timedelta(seconds=int(time() - t0_init_training))),
        )
        return results
