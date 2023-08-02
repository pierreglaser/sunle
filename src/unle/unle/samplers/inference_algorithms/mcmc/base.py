from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    cast,
)

from numpyro.distributions.transforms import AffineTransform

if TYPE_CHECKING:
    from jax.random import KeyArray
    from unle.samplers.pytypes import LogDensity_T, Numeric, PyTreeNode

import jax
import jax.numpy as jnp
import numpyro.distributions as np_distributions
from flax import struct
from jax import Array, random, vmap
from jax._src.flatten_util import ravel_pytree
from jax.lax import scan  # type: ignore
from jax.tree_util import tree_map
from numpyro.infer.hmc_util import HMCAdaptState
from tqdm.auto import tqdm as tqdm_auto
from typing_extensions import Self, Type
from unle.samplers.distributions import (
    DoublyIntractableLogDensity,
    ThetaConditionalLogDensity,
)
from unle.samplers.inference_algorithms.base import (
    InferenceAlgorithm,
    InferenceAlgorithmConfig,
    InferenceAlgorithmFactory,
    InferenceAlgorithmInfo,
    InferenceAlgorithmResults,
)
from unle.samplers.kernels.adaptive_mala import AdaptiveMALAState
from unle.samplers.kernels.mala import MALAConfig, MALAKernelFactory

from ...kernels.base import (
    Array_T,
    Config_T,
    Info_T,
    KernelFactory,
    Result,
    State_T,
    TunableConfig,
    TunableKernel,
    TunableMHKernelFactory,
)
from ...particle_aproximation import ParticleApproximation
from .util import progress_bar_factory


def tree_any(function: Callable[[PyTreeNode], Numeric], tree: PyTreeNode) -> Numeric:
    mapped_tree = tree_map(function, tree)
    return jnp.any(ravel_pytree(mapped_tree)[0])


def adam_initialize_doubly_intractable(
    theta: Array,
    target_log_prob_fn: DoublyIntractableLogDensity,
    key: KeyArray,
    num_steps=50,
    learning_rate=0.05,
    num_likelihood_sampler_steps: int = 100,
):
    """Use Adam optimizer to get a reasonable initialization for HMC algorithms.

    Args:
      x: Where to initialize Adam.
      target_log_prob_fn: Unnormalized target log-density.
      num_steps: How many steps of Adam to run.
      learning_rate: What learning rate to pass to Adam.

    Returns:
      Optimized version of x.
    """
    import jax
    import optax

    init_mcmc_chain = MCMCChain(
        MCMCChainConfig(
            MALAKernelFactory(MALAConfig(1.0, None)),
            num_steps=num_likelihood_sampler_steps // 2,
            num_warmup_steps=num_likelihood_sampler_steps // 2,
            adapt_mass_matrix=False,
            adapt_step_size=True,
            target_accept_rate=0.5,
            record_trajectory=True,
        ),
        ThetaConditionalLogDensity(target_log_prob_fn.log_likelihood, theta),
    )
    init_mcmc_chain = init_mcmc_chain.init(target_log_prob_fn.x_obs)
    init_mcmc_chain, _ = init_mcmc_chain.run(key=random.fold_in(key, 0))

    def update_step(input_: Tuple[Array, Any, Any, float], i):
        theta, adam_state, mcmc_chain, lr = input_

        _, g_log_prior = tree_map(
            lambda x: -x, jax.value_and_grad(target_log_prob_fn.log_prior)(theta)
        )
        _, g_log_lik_unnormalized = tree_map(
            lambda x: -x,
            jax.value_and_grad(target_log_prob_fn.log_likelihood)(
                theta, target_log_prob_fn.x_obs
            ),
        )

        assert isinstance(mcmc_chain, MCMCChain)
        assert isinstance(mcmc_chain.log_prob, ThetaConditionalLogDensity)
        mcmc_chain = cast(
            MCMCChain,
            mcmc_chain.replace(log_prob=mcmc_chain.log_prob.replace(theta=theta)),
        )
        new_mcmc_chain, results = mcmc_chain.run(key=random.fold_in(key, i))

        g_log_normalizer = jnp.average(
            vmap(jax.grad(target_log_prob_fn.log_likelihood), in_axes=(None, 0))(
                theta, results.chain.x
            ),
            axis=0,
        )
        g = g_log_prior + g_log_lik_unnormalized + g_log_normalizer

        # updates, new_adam_state = optax.adam(0.001).update(g, adam_state)
        updates, new_adam_state = optax.adam(lr).update(g, adam_state)
        new_theta = cast(Array, optax.apply_updates(theta, updates))

        has_nan = tree_any(
            lambda x: jnp.isnan(x), (target_log_prob_fn(new_theta), new_theta)
        )
        has_inf = tree_any(
            lambda x: jnp.isinf(x), (target_log_prob_fn(new_theta), new_theta)
        )

        new_ret = jax.lax.cond(
            has_nan,
            lambda _: (
                theta,
                optax.adam(lr / 1.5).init(theta),
                new_mcmc_chain,
                lr / 1.5,
            ),
            lambda _: (new_theta, new_adam_state, new_mcmc_chain, lr),
            None,
        )
        return new_ret, (new_ret, has_nan, has_inf)

    init_state = optax.adam(learning_rate).init(theta)

    (theta, _, _, final_lr), traj = jax.lax.scan(
        update_step,
        (theta, init_state, init_mcmc_chain, learning_rate),
        jnp.arange(1, num_steps + 1),
    )

    return theta


def adam_initialize(
    x: Array, target_log_prob_fn: LogDensity_T, num_steps=50, learning_rate=0.05
):
    """Use Adam optimizer to get a reasonable initialization for HMC algorithms.

    Args:
      x: Where to initialize Adam.
      target_log_prob_fn: Unnormalized target log-density.
      num_steps: How many steps of Adam to run.
      learning_rate: What learning rate to pass to Adam.

    Returns:
      Optimized version of x.
    """
    import jax
    import optax

    def update_step(input_: Tuple[Array, Any, float], _):
        x, adam_state, lr = input_

        def g_fn(x):
            return tree_map(lambda x: -x, jax.value_and_grad(target_log_prob_fn)(x))

        v, g = g_fn(x)
        updates, adam_state = optax.adam(lr).update(g, adam_state)
        new_x = cast(Array, optax.apply_updates(x, updates))

        has_inf = tree_any(lambda x: jnp.isinf(x), (target_log_prob_fn(new_x), new_x))

        new_ret = jax.lax.cond(
            has_inf,
            lambda _: (x, optax.adam(lr / 1.5).init(x), lr / 1.5),
            lambda _: (new_x, adam_state, lr),
            None,
        )
        return new_ret, (*new_ret, g, v, has_inf, lr)

    init_state = optax.adam(learning_rate).init(x)
    (x, _, lrs), all_vals = jax.lax.scan(
        update_step, (x, init_state, learning_rate), jnp.arange(1, num_steps + 1)
    )
    return x


Carry_T = TypeVar("Carry_T")


def turn_scan_update_fn_into_fori_update_fn(
    update_fn: Callable[[Carry_T, Array_T], Tuple[Carry_T, Any]]
) -> Callable[[Carry_T, Array_T], Carry_T]:
    def fori_loop_update_fn(i: Array_T, carry: Carry_T):
        new_carry, _ = update_fn(carry, i)
        return new_carry

    return fori_loop_update_fn


class MCMCChainConfig(Generic[Config_T, State_T, Info_T], struct.PyTreeNode):
    kernel_factory: KernelFactory[Config_T, State_T, Info_T]
    num_steps: int = struct.field(pytree_node=False)
    record_trajectory: bool = struct.field(pytree_node=False)
    num_warmup_steps: int = struct.field(pytree_node=False)
    adapt_step_size: bool = struct.field(pytree_node=False)
    adapt_mass_matrix: bool = struct.field(pytree_node=False)
    target_accept_rate: float = 0.2
    warmup_method: Literal["numpyro", "jax_samplers"] = struct.field(
        pytree_node=False, default="jax_samplers"
    )
    init_using_log_l_mode: bool = struct.field(pytree_node=False, default=True)
    init_using_log_l_mode_num_opt_steps: int = struct.field(
        pytree_node=False, default=500
    )
    online: bool = struct.field(pytree_node=False, default=False)

    def set_step_size(self, step_size: Array_T):
        assert isinstance(self.kernel_factory, TunableMHKernelFactory)
        return self.replace(
            kernel_factory=self.kernel_factory.set_step_size(step_size=step_size)
        )

    def get_step_size(self):
        assert isinstance(self.kernel_factory, TunableMHKernelFactory)
        return self.kernel_factory.get_step_size()

    def set_inverse_mass_matrix(self, C: Array_T):
        assert isinstance(self.kernel_factory, TunableMHKernelFactory)
        return self.replace(
            kernel_factory=self.kernel_factory.set_inverse_mass_matrix(C=C)
        )


class SingleChainResults(Generic[State_T, Info_T], struct.PyTreeNode):
    final_state: State_T
    chain: State_T
    info: Info_T
    warmup_info: Optional[Info_T] = None
    f_avg: Optional[Array] = None


class MCMCChain(Generic[Config_T, State_T, Info_T], struct.PyTreeNode):
    config: MCMCChainConfig[Config_T, State_T, Info_T]
    log_prob: LogDensity_T
    init_state: Optional[State_T] = None
    _chain_id: int = 0
    p_bar_update_fn: Optional[Callable[[int, int], int]] = struct.field(
        pytree_node=False, default=None
    )

    def init(self, x0: Array) -> Self:
        init_state = self.config.kernel_factory.build_kernel(self.log_prob).init_state(
            x0
        )
        return self.replace(init_state=init_state)

    def get_init_state(self):
        assert self.init_state is not None
        return self.init_state

    def _init_from_log_l_mode(self, key: KeyArray) -> Self:
        assert self.init_state is not None
        init_state = self.init_state
        if not isinstance(self.log_prob, DoublyIntractableLogDensity):
            print("finding good initial position")
            good_first_position = adam_initialize(init_state.x, self.log_prob)
            init_state = init_state.replace(x=good_first_position)
        else:
            print("finding good initial position (doubly intractable)")
            key, subkey = random.split(key)

            good_first_position = adam_initialize_doubly_intractable(
                init_state.x,
                self.log_prob,
                subkey,
                learning_rate=0.05,
                num_steps=self.config.init_using_log_l_mode_num_opt_steps,
            )
            print("good initial position found at: ", good_first_position)
            init_state = init_state.replace(x=good_first_position)

        return self.replace(init_state=init_state)

    def set_log_prob(self, log_prob: LogDensity_T) -> Self:
        self = self.replace(log_prob=log_prob)
        if self.init_state is not None:
            # states like SAVM might depend on the given log_prob and so must
            # thus be updated. this is done by re-initializing the kernel state
            # and keeping the position, but ideally states should either not
            # depend on the log_prob or implement a `set_log_prob` method.
            kernel = self.config.kernel_factory.build_kernel(log_prob)
            new_state = kernel.init_state(self.init_state.x)
            return self.replace(init_state=new_state)
        else:
            return self

    def run(
        self, key: KeyArray, f: Optional[Callable[[Array], Array]] = None
    ) -> Tuple[Self, SingleChainResults[State_T, Info_T]]:
        key, subkey = random.split(key)
        if self.config.init_using_log_l_mode:
            self = self._init_from_log_l_mode(subkey)

        if self.config.num_warmup_steps > 0:
            key, subkey = random.split(key)
            self, warmup_info = self._warmup(subkey)
        else:
            warmup_info = None

        kernel = self.config.kernel_factory.build_kernel(self.log_prob)

        assert self.init_state is not None
        init_state = self.init_state

        if f is not None:
            init_f_val = f(init_state.x)
        else:
            init_f_val = None

        def step_fn(
            carry: Tuple[State_T, Array_T], iter_no: Array
        ) -> Tuple[Tuple[State_T, Array_T], Optional[Result[State_T, Info_T]]]:
            x, f_avg = carry
            mala_result = kernel.one_step(x, random.fold_in(subkey, cast(int, iter_no)))
            self._maybe_update_pbar(iter_no, self._chain_id)

            if f is not None:
                new_f_eval = f(mala_result.state.x)
                new_f_avg = f_avg + (new_f_eval - f_avg) / (
                    iter_no - self.config.num_warmup_steps + 1
                )
                new_carry = (mala_result.state, new_f_avg)
            else:
                new_carry = (mala_result.state, None)

            if not self.config.record_trajectory:
                output = None
            else:
                output = mala_result.replace(state=mala_result.state.trim())

            return new_carry, output

        assert self.init_state is not None
        init_state = self.init_state

        key, subkey = random.split(key)
        if not self.config.online:
            final_carry, outputs = scan(
                step_fn,
                (init_state, init_f_val),
                xs=jnp.arange(
                    self.config.num_warmup_steps,
                    self.config.num_warmup_steps + self.config.num_steps,
                ),
            )  # type: ignore
        else:
            fori_loop_step_fn = turn_scan_update_fn_into_fori_update_fn(step_fn)
            final_carry = jax.lax.fori_loop(
                lower=self.config.num_warmup_steps,
                upper=self.config.num_steps + self.config.num_warmup_steps,
                body_fun=fori_loop_step_fn,
                init_val=(init_state, init_f_val),
            )
            outputs = None

        final_state, avg_f = final_carry

        if self.config.record_trajectory:
            assert not self.config.online
            assert outputs is not None
            stats, chain = outputs.info, outputs.state
        else:
            _smoke_res = kernel.one_step(init_state, key)
            stats = _smoke_res.info
            chain = tree_map(lambda x: x[None, ...], final_state)
        return self, SingleChainResults(final_state, chain, stats, warmup_info, avg_f)

    def _maybe_update_pbar(self, iter_no, _chain_id) -> int:
        if self.p_bar_update_fn is not None:
            return self.p_bar_update_fn(iter_no, _chain_id)
        else:
            return iter_no

    def _warmup(self, key: KeyArray) -> Tuple[Self, Info_T]:
        if self.config.warmup_method == "jax_samplers":
            return self._custom_warmup(key)
        elif self.config.warmup_method == "numpyro":
            return self._warmup_numpyro(key)
        else:
            raise ValueError(f"Unknown warmup method {self.config.warmup_method}")

    def _custom_warmup(self, key: KeyArray) -> Tuple[Self, Info_T]:
        if self.config.adapt_mass_matrix or self.config.adapt_step_size:
            assert isinstance(self.config.kernel_factory, TunableMHKernelFactory)
            assert isinstance(self.config.kernel_factory.config, TunableConfig)
        kernel = self.config.kernel_factory.build_kernel(self.log_prob)

        record_trajectory = False

        def step_fn(
            carry: Tuple[State_T, AdaptiveMALAState], iter_no: Array
        ) -> Tuple[
            Tuple[State_T, AdaptiveMALAState], Optional[Result[State_T, Info_T]]
        ]:
            this_kernel = kernel
            x, adaptation_state = carry
            if self.config.adapt_step_size:
                assert isinstance(this_kernel, TunableKernel)
                this_kernel = this_kernel.set_step_size(adaptation_state.sigma)

            if self.config.adapt_mass_matrix:
                assert isinstance(this_kernel, TunableKernel)
                this_kernel = this_kernel.set_inverse_mass_matrix(
                    adaptation_state.get_C()
                )

            mala_result = this_kernel.one_step(
                x, random.fold_in(subkey, cast(int, iter_no))
            )

            next_adaptation_state = adaptation_state

            if self.config.adapt_step_size:
                next_adaptation_state = next_adaptation_state.update_sigma(
                    log_alpha=getattr(mala_result.info, "log_alpha", 0),
                    gamma_n=1 / (next_adaptation_state.iter_no + 1) ** 0.5,
                )
            if self.config.adapt_mass_matrix:
                next_adaptation_state = next_adaptation_state.update_cov(
                    x=mala_result.state.x,
                    gamma_n=1 / (next_adaptation_state.iter_no + 1) ** 0.5,
                )

            next_adaptation_state = next_adaptation_state.replace(
                iter_no=next_adaptation_state.iter_no + 1, x=mala_result.state.x
            )

            if not record_trajectory:
                output = None
            else:
                output = mala_result

            self._maybe_update_pbar(iter_no, self._chain_id)
            return (mala_result.state, next_adaptation_state), output

        assert self.init_state is not None
        init_state = self.init_state

        target_accept_rate = self.config.target_accept_rate
        if self.config.adapt_mass_matrix:
            assert isinstance(kernel, TunableKernel)
            uses_diagonal_mass_matrix = len(kernel.get_inverse_mass_matrix().shape) == 1
            if uses_diagonal_mass_matrix:
                init_adaptation_state = AdaptiveMALAState(
                    init_state.x,
                    1,
                    init_state.x,
                    jnp.zeros((init_state.x.shape[0],)),
                    1.0,
                    target_accept_rate,
                )
            else:
                init_adaptation_state = AdaptiveMALAState(
                    init_state.x,
                    1,
                    init_state.x,
                    jnp.zeros((init_state.x.shape[0], init_state.x.shape[0])),
                    1.0,
                    target_accept_rate,
                )
        else:
            init_adaptation_state = AdaptiveMALAState(
                init_state.x, 1, None, None, 1.0, target_accept_rate
            )

        key, subkey = random.split(key)
        if not self.config.online:
            final_state, outputs = scan(
                step_fn,
                (init_state, init_adaptation_state),
                xs=jnp.arange(1, self.config.num_warmup_steps + 1),
            )
        else:
            fori_loop_step_fn = turn_scan_update_fn_into_fori_update_fn(step_fn)
            final_state = jax.lax.fori_loop(
                lower=1,
                upper=self.config.num_warmup_steps + 1,
                body_fun=fori_loop_step_fn,
                init_val=(init_state, init_adaptation_state),
            )
            outputs = None

        if record_trajectory:
            assert not self.config.online
            assert outputs is not None
            stats = outputs.info
        else:
            _smoke_res = kernel.one_step(init_state, key)
            stats = _smoke_res.info

        new_init_state = final_state[0]

        final_kernel = kernel

        if self.config.adapt_step_size:
            assert isinstance(final_kernel, TunableKernel)
            final_kernel = final_kernel.set_step_size(step_size=final_state[1].sigma)
        if self.config.adapt_mass_matrix:
            assert isinstance(final_kernel, TunableKernel)
            final_kernel = final_kernel.set_inverse_mass_matrix(final_state[1].get_C())

        return (
            self.replace(
                config=self.config.replace(
                    kernel_factory=self.config.kernel_factory.replace(  # pyright: ignore [reportGeneralTypeIssues]  # noqa
                        config=final_kernel.config
                    )
                ),
                init_state=new_init_state,
            ),
            stats,
        )

    def _warmup_numpyro(self, key: KeyArray) -> Tuple[Self, Info_T]:
        kernel = self.config.kernel_factory.build_kernel(self.log_prob)
        assert isinstance(kernel, TunableKernel)

        if self.config.adapt_mass_matrix:
            init_mass_matrix = kernel.get_inverse_mass_matrix()
        else:
            assert self.init_state is not None
            init_mass_matrix = jnp.ones((self.init_state.x.shape[0],))

        from numpyro.infer.hmc_util import warmup_adapter

        wa_init, _wa_update = warmup_adapter(
            self.config.num_warmup_steps,
            adapt_step_size=self.config.adapt_step_size,
            adapt_mass_matrix=self.config.adapt_mass_matrix,
            dense_mass=init_mass_matrix is not None
            and len(init_mass_matrix.shape) == 2,
            target_accept_prob=self.config.target_accept_rate,
        )

        assert self.init_state is not None
        init_state = self.init_state

        key, subkey = random.split(key)
        init_adaptation_state = wa_init(
            (init_state.x,),
            subkey,
            cast(float, kernel.get_step_size()),
            mass_matrix_size=init_state.x.shape[0],
        )
        init_adaptation_state = init_adaptation_state._replace(rng_key=None)
        record_trajectory = False

        def step_fn(
            carry: Tuple[State_T, HMCAdaptState], iter_no: Array
        ) -> Tuple[Tuple[State_T, HMCAdaptState], Optional[Result[State_T, Info_T]]]:
            this_kernel = kernel
            x, adaptation_state = carry
            if self.config.adapt_step_size:
                assert isinstance(this_kernel, TunableKernel)
                this_kernel = this_kernel.set_step_size(adaptation_state.step_size)

            if self.config.adapt_mass_matrix:
                assert isinstance(this_kernel, TunableKernel)
                this_kernel = this_kernel.set_inverse_mass_matrix(
                    adaptation_state.inverse_mass_matrix
                )

            mala_result = this_kernel.one_step(
                x, random.fold_in(subkey, cast(int, iter_no))
            )

            next_adaptation_state = _wa_update(
                iter_no,
                jnp.exp(
                    jnp.clip(
                        mala_result.info.log_alpha,  # type: ignore
                        a_max=0,
                    )
                ),
                (mala_result.state.x,),
                adaptation_state,
            )

            if not record_trajectory:
                output = None
            else:
                output = mala_result

            self._maybe_update_pbar(iter_no, self._chain_id)
            return (mala_result.state, next_adaptation_state), output

        if not self.config.online:
            final_state, outputs = scan(
                step_fn,
                (init_state, init_adaptation_state),
                xs=jnp.arange(1, self.config.num_warmup_steps + 1),
            )
        else:
            fori_loop_step_fn = turn_scan_update_fn_into_fori_update_fn(step_fn)
            final_state = jax.lax.fori_loop(
                lower=1,
                upper=self.config.num_warmup_steps + 1,
                body_fun=fori_loop_step_fn,
                init_val=(init_state, init_adaptation_state),
            )
            outputs = None

        final_kernel = kernel
        new_init_state = final_state[0]
        if self.config.adapt_step_size:
            assert isinstance(final_kernel, TunableKernel)
            final_kernel = final_kernel.set_step_size(
                step_size=final_state[1].step_size
            )
        if self.config.adapt_mass_matrix:
            assert isinstance(final_kernel, TunableKernel)
            final_kernel = final_kernel.set_inverse_mass_matrix(
                final_state[1].inverse_mass_matrix
            )

        if record_trajectory:
            assert not self.config.online
            assert outputs is not None
            stats = outputs.info
        else:
            _smoke_res = kernel.one_step(init_state, key)
            stats = _smoke_res.info

        return (
            self.replace(
                config=self.config.replace(
                    kernel_factory=self.config.kernel_factory.replace(
                        config=final_kernel.config
                    )
                ),
                init_state=new_init_state,
            ),
            stats,
        )

    def reset_at_final_state(self, final_state: State_T) -> Self:
        return self.replace(init_state=final_state)

    @classmethod
    def vmap_axes(
        cls, obj: Optional[Self], config: int, log_prob: Any, init_state: Optional[int]
    ) -> MCMCChain:
        return MCMCChain(
            config,  # pyright: ignore [reportGeneralTypeIssues]
            log_prob,
            init_state,  # pyright: ignore [reportGeneralTypeIssues]
        )


class MCMCConfig(
    Generic[Config_T, State_T, Info_T], InferenceAlgorithmConfig, struct.PyTreeNode
):
    kernel_factory: KernelFactory[Config_T, State_T, Info_T]
    num_samples: int = struct.field(pytree_node=False)
    num_chains: int = struct.field(pytree_node=False, default=100)
    thinning_factor: int = struct.field(pytree_node=False, default=10)
    record_trajectory: bool = struct.field(pytree_node=False, default=True)
    num_warmup_steps: int = struct.field(pytree_node=False, default=0)
    adapt_step_size: bool = struct.field(pytree_node=False, default=False)
    adapt_mass_matrix: bool = struct.field(pytree_node=False, default=False)
    resample_stuck_chain_at_warmup: bool = struct.field(
        pytree_node=False, default=False
    )
    target_accept_rate: float = struct.field(pytree_node=False, default=0.2)
    progress_bar: bool = struct.field(pytree_node=False, default=False)
    warmup_method: Literal["numpyro", "jax_samplers"] = struct.field(
        pytree_node=False, default="jax_samplers"
    )
    init_using_log_l_mode: bool = struct.field(pytree_node=False, default=True)
    init_using_log_l_mode_num_opt_steps: int = struct.field(
        pytree_node=False, default=50
    )


class MCMCInfo(Generic[State_T, Info_T], InferenceAlgorithmInfo):
    single_chain_results: SingleChainResults[State_T, Info_T]


class MCMCResults(Generic[State_T, Info_T], InferenceAlgorithmResults):
    samples: ParticleApproximation
    info: MCMCInfo[State_T, Info_T]


class MCMCAlgorithm(InferenceAlgorithm[MCMCConfig[Config_T, State_T, Info_T]]):
    single_chains: Optional[MCMCChain[Config_T, State_T, Info_T]] = None

    @property
    def initialized(self):
        return (
            self.single_chains is not None and self.single_chains.init_state is not None
        )

    def get_single_chains(self):
        assert self.single_chains is not None
        return self.single_chains

    @property
    def _uninitialized_chain_vmap_axes(self) -> MCMCChain:
        from jax.tree_util import tree_map

        assert self.single_chains is not None
        return cast(MCMCChain, tree_map(lambda x: 0, self.single_chains)).replace(
            log_prob=None, init_state=None
        )

    @property
    def _initialized_chain_vmap_axes(self) -> MCMCChain:
        from jax.tree_util import tree_map

        assert self.single_chains is not None
        return cast(MCMCChain, tree_map(lambda x: 0, self.single_chains)).replace(
            log_prob=None, init_state=0
        )

    @classmethod
    def get_single_chain_num_steps(
        cls: Type[Self], num_samples: int, thinning_factor: int, num_chains: int
    ) -> int:
        num_total_steps = num_samples * thinning_factor / num_chains
        assert num_total_steps == int(num_total_steps)
        return int(num_total_steps)

    @property
    def can_set_num_samples(self) -> bool:
        return True

    def set_num_samples(self, num_samples: int) -> Self:
        single_chains = self.single_chains
        assert single_chains is not None
        new_num_total_steps = type(self).get_single_chain_num_steps(
            num_samples, self.config.thinning_factor, self.config.num_chains
        )
        new_self = self.replace(
            single_chains=single_chains.replace(
                config=single_chains.config.replace(num_steps=new_num_total_steps)
            ),
            config=self.config.replace(num_samples=num_samples),
        )
        return new_self

    @classmethod
    def create(
        cls, config: MCMCConfig[Config_T, State_T, Info_T], log_prob: LogDensity_T
    ) -> Self:
        # build single chain MCMC configs
        num_total_steps = cls.get_single_chain_num_steps(
            config.num_samples, config.thinning_factor, config.num_chains
        )
        _single_chain_configs = vmap(
            lambda _: MCMCChainConfig(
                config.kernel_factory,
                int(num_total_steps),
                True,
                config.num_warmup_steps,
                config.adapt_step_size,
                config.adapt_mass_matrix,
                config.target_accept_rate,
                warmup_method=config.warmup_method,
                init_using_log_l_mode=config.init_using_log_l_mode,
                init_using_log_l_mode_num_opt_steps=config.init_using_log_l_mode_num_opt_steps,
            )
        )(jnp.arange(config.num_chains))
        single_chains = vmap(MCMCChain, in_axes=(0, None, None, 0), out_axes=MCMCChain(0, None, None, 0))(_single_chain_configs, log_prob, None, jnp.arange(config.num_chains))  # type: ignore # noqa
        return cls(config, log_prob, init_state=None, single_chains=single_chains)

    def init(
        self,
        key: KeyArray,
        dist: np_distributions.Distribution,
        reweight_and_resample: bool = False,
    ) -> Self:
        xs = dist.sample(key, (self.config.num_chains,))
        init_state = ParticleApproximation(xs, jnp.zeros(self.config.num_chains))
        if reweight_and_resample:
            key, subkey = random.split(key)
            log_ratio = vmap(self.log_prob)(init_state.xs) - vmap(dist.log_prob)(
                init_state.xs
            )
            init_state = init_state.replace(
                log_ws=log_ratio
            ).resample_and_reset_weights(subkey)

        single_chains = vmap(
            lambda c, x0: cast(MCMCChain[Config_T, State_T, Info_T], c).init(x0),
            in_axes=(MCMCChain(0, None, None, 0), 0),  #  type: ignore
            out_axes=MCMCChain(0, None, 0, 0),  #  type: ignore
        )(self.single_chains, init_state.particles)

        return self.replace(init_state=init_state, single_chains=single_chains)

    def init_from_particles(self, xs: Array) -> Self:
        assert len(xs.shape) == 2
        assert len(xs) == self.config.num_chains
        init_state = ParticleApproximation(xs, jnp.zeros(self.config.num_samples))

        single_chains = vmap(
            lambda c, x0: cast(MCMCChain[Config_T, State_T, Info_T], c).init(x0),
            in_axes=(MCMCChain(0, None, None, 0), 0),  #  type: ignore
            out_axes=MCMCChain(0, None, 0, 0),  #  type: ignore
        )(self.single_chains, init_state.particles)

        return self.replace(init_state=init_state, single_chains=single_chains)

    def set_log_prob(self, log_prob: LogDensity_T) -> Self:
        self = self.replace(log_prob=log_prob)
        # TODO: handle cases where either single_chains or init_state is None
        if self.single_chains is not None:
            single_chains = vmap(
                lambda c, x0: cast(
                    MCMCChain[Config_T, State_T, Info_T], c
                ).set_log_prob(log_prob),
                in_axes=(MCMCChain(0, None, 0, 0), None),  # type: ignore
                out_axes=MCMCChain(0, None, 0, 0),  #  type: ignore
            )(self.single_chains, log_prob)
            self = self.replace(
                single_chains=single_chains,
            )
        return self

    def set_num_warmup_steps(self, num_warmup_steps) -> Self:
        self = cast(
            Self,
            self.replace(config=self.config.replace(num_warmup_steps=num_warmup_steps)),
        )
        if self.single_chains is not None:
            self = self.replace(
                single_chains=self.single_chains.replace(
                    config=self.single_chains.config.replace(
                        num_warmup_steps=num_warmup_steps
                    )
                )
            )
        return self

    def _maybe_set_progress_bar(self) -> Self:
        assert self.single_chains is not None
        if self.config.progress_bar:
            pbar = tqdm_auto(
                range(
                    (
                        self.single_chains.config.num_steps
                        + self.single_chains.config.num_warmup_steps
                    )
                    * self.config.num_chains
                ),
                miniters=100,
                mininterval=100,
            )
            pbar.set_description("Compiling.. ", refresh=True)

            new_single_chains = self.single_chains.replace(
                p_bar_update_fn=progress_bar_factory(
                    pbar,
                    self.single_chains.config.num_steps
                    + self.single_chains.config.num_warmup_steps,
                )
            )
            return self.replace(single_chains=new_single_chains)
        else:
            return self

    def _maybe_remove_progress_bar(self) -> Self:
        assert self.single_chains is not None
        return self.replace(
            single_chains=self.single_chains.replace(p_bar_update_fn=None)
        )

    def _aggregate_single_chain_results(
        self, single_chain_results: SingleChainResults[State_T, Info_T]
    ) -> Array_T:
        final_samples = single_chain_results.chain.x[
            :, :: -self.config.thinning_factor, :
        ].reshape(-1, single_chain_results.chain.x.shape[-1])
        if len(final_samples) == self.config.num_samples:
            return final_samples
        else:
            final_samples = single_chain_results.chain.x.reshape(
                -1, single_chain_results.chain.x.shape[-1]
            )[:: -self.config.thinning_factor, :]
            assert len(final_samples) == self.config.num_samples
            return final_samples

    def reparametrize(self, map: AffineTransform) -> Self:
        """Given a map :math:`f`, adapt the sampler to sample
        from the pushforward of the target distribution by :math:`f`
        """
        single_chains = self.get_single_chains()

        # step size
        zscoring_ratio = jnp.sqrt(jnp.average(map.scale))
        print("zscoring ratio", zscoring_ratio)
        tuned_chain_config = single_chains.config.set_step_size(
            single_chains.config.get_step_size() * zscoring_ratio
        )
        new_alg = self.replace(
            single_chains=single_chains.replace(config=tuned_chain_config)
        )

        # init state
        init_state = single_chains.get_init_state()
        init_state_pos_rescaled = vmap(map, in_axes=(0,))(init_state.x)
        init_state_rescaled = init_state.replace(x=init_state_pos_rescaled)
        new_alg = new_alg.replace(
            single_chains=new_alg.get_single_chains().replace(
                init_state=init_state_rescaled
            )
        )
        return new_alg

    def run(self, key: KeyArray) -> Tuple[Self, MCMCResults[State_T, Info_T]]:
        self = self._maybe_set_progress_bar()
        assert self.single_chains is not None

        key, subkey = random.split(key)
        new_single_chains, single_chain_results = vmap(
            lambda c, k: cast(MCMCChain[Config_T, State_T, Info_T], c).run(k),
            # in_axes=(0, 0), out_axes=(0, 0)
            in_axes=(
                MCMCChain(
                    0,  # type: ignore
                    None,  # type: ignore
                    0,  # type: ignore
                    0,
                    self.single_chains.p_bar_update_fn,
                ),
                0,
            ),
            out_axes=(
                MCMCChain(
                    0,  # type: ignore
                    None,  # type: ignore
                    0,  # type: ignore
                    0,  # type: ignore
                    self.single_chains.p_bar_update_fn,
                ),
                0,
            ),
        )(self.single_chains, random.split(subkey, self.config.num_chains))

        final_samples = self._aggregate_single_chain_results(single_chain_results)

        self = self.replace(
            single_chains=new_single_chains
        )._maybe_remove_progress_bar()
        return self, MCMCResults(
            ParticleApproximation(final_samples, jnp.zeros((final_samples.shape[0],))),
            info=MCMCInfo(single_chain_results),
        )

    def reset_at_final_state(self, final_state: State_T) -> Self:
        assert self.single_chains is not None
        return self.replace(
            single_chains=self.single_chains.reset_at_final_state(final_state)
        )

    def estimate_expectation_of(
        self, f: Callable[[Array_T], Array_T], key: KeyArray
    ) -> Tuple[Self, Array, SingleChainResults[State_T, Info_T]]:
        self = self._maybe_set_progress_bar()
        single_chains = self.single_chains
        assert single_chains is not None

        init_single_chains = single_chains.replace(
            config=single_chains.config.replace(online=True, record_trajectory=False)
        )

        key, subkey = random.split(key)
        # TODO: consider using `pmap` for increased performance on CPU
        new_single_chains, single_chain_results = vmap(
            lambda c, k: cast(MCMCChain[Config_T, State_T, Info_T], c).run(k, f),
            # in_axes=(0, 0), out_axes=(0, 0)
            in_axes=(
                MCMCChain(
                    0,  # type: ignore
                    None,  # type: ignore
                    0,  # type: ignore
                    0,  # type: ignore
                    single_chains.p_bar_update_fn,
                ),
                0,
            ),
            out_axes=(
                MCMCChain(
                    0, None, 0, 0, self.single_chains.p_bar_update_fn  #  type: ignore
                ),
                0,
            ),
        )(init_single_chains, random.split(subkey, self.config.num_chains))

        single_chain_f_avgs = single_chain_results.f_avg
        assert single_chain_f_avgs is not None

        new_single_chains = new_single_chains.replace(
            config=new_single_chains.config.replace(
                online=False, record_trajectory=single_chains.config.record_trajectory
            )
        )

        self = self.replace(
            single_chains=new_single_chains
        )._maybe_remove_progress_bar()
        return self, jnp.mean(single_chain_f_avgs, axis=0), single_chain_results

    @classmethod
    def vmap_axes(
        cls,
        obj: Optional[Self],
        config: Optional[int],
        log_prob: Any,
        init_state: Optional[int],
        single_chains: Any,
    ) -> Self:
        if obj is None:
            return cls(
                config,  # pyright: ignore [reportGeneralTypeIssues]
                log_prob,
                init_state,  # pyright: ignore [reportGeneralTypeIssues]
                single_chains,
            )
        else:
            return obj.replace(
                config=config,
                log_prob=log_prob,
                init_state=init_state,
                single_chains=single_chains,
            )


class MCMCAlgorithmFactory(
    InferenceAlgorithmFactory[MCMCConfig[Config_T, State_T, Info_T]]
):
    def build_algorithm(
        self, log_prob: LogDensity_T
    ) -> MCMCAlgorithm[Config_T, State_T, Info_T]:
        return self.inference_alg_cls.create(log_prob=log_prob, config=self.config)

    @property
    def inference_alg_cls(self) -> Type[MCMCAlgorithm]:
        return MCMCAlgorithm
