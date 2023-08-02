from typing import Callable, Tuple, cast

from flax import struct
from jax import Array, tree_map, vmap
from jax.random import KeyArray
from jax.tree_util import tree_flatten, tree_unflatten
from numpyro.distributions.transforms import AffineTransform
from typing_extensions import Self, Type, TypeVarTuple, Unpack
from unle.samplers.inference_algorithms.base import InferenceAlgorithmResults
from unle.samplers.inference_algorithms.mcmc.base import (
    MCMCAlgorithm,
    MCMCAlgorithmFactory,
    MCMCChain,
    SingleChainResults,
)
from unle.samplers.kernels.base import Config_T, Info_T, State_T
from unle.samplers.pytypes import LogDensity_T


class VmappedMCMCFactory(MCMCAlgorithmFactory):
    _flattened_log_prob_vmap_axes: Tuple = struct.field(pytree_node=False, default=None)

    @classmethod
    def from_mcmc_factory(
        cls,
        mcmc_factory: MCMCAlgorithmFactory,
        log_prob_vmap_axes: LogDensity_T,
    ):
        return cls(
            mcmc_factory.config,
            tree_flatten(log_prob_vmap_axes),
        )

    @property
    def log_prob_vmap_axes(self):
        return tree_unflatten(
            self._flattened_log_prob_vmap_axes[1], self._flattened_log_prob_vmap_axes[0]
        )

    @property
    def inference_alg_cls(self) -> Type["VmappedMCMC"]:
        return VmappedMCMC

    def build_algorithm(self, log_prob: LogDensity_T) -> "VmappedMCMC":
        out_axes = self.inference_alg_cls.vmap_axes(
            None,
            0,
            self.log_prob_vmap_axes,
            None,
            MCMCChain.vmap_axes(None, 0, self.log_prob_vmap_axes, None),
        )
        vmapped_mcmc_alg = cast(
            VmappedMCMC,
            vmap(
                super(VmappedMCMCFactory, self).build_algorithm,
                in_axes=(self.log_prob_vmap_axes,),
                out_axes=out_axes,
            )(log_prob),
        )
        return vmapped_mcmc_alg.replace(
            _flattened_vmap_axes=self._flattened_log_prob_vmap_axes
        )


T = TypeVarTuple("T")


class VmappedMCMC(MCMCAlgorithm[Config_T, State_T, Info_T]):
    # store flattened `vmap_axes` because auxiliary fields of pytrees must
    # have the same exact hash in order to be manipulated by jax pytree functions
    # like `tree_map`
    _flattened_vmap_axes: Tuple = struct.field(pytree_node=False, default=None)
    passthrough: bool = struct.field(pytree_node=False, default=False)

    @property
    def log_prob_vmap_axes(self):
        return tree_unflatten(
            self._flattened_vmap_axes[1], self._flattened_vmap_axes[0]
        )

    def _inner_vmap_axes(self, initialized: bool = True):
        return VmappedMCMC.vmap_axes(
            self,
            0,
            self.log_prob_vmap_axes,
            0 if initialized else None,
            MCMCChain.vmap_axes(
                None, 0, self.log_prob_vmap_axes, 0 if initialized else None
            ),
        )

    def init_from_particles(self, xs: Array) -> Self:
        if self.passthrough:
            return super().init_from_particles(xs)

        self = vmap(
            MCMCAlgorithm.init_from_particles,
            in_axes=(self._inner_vmap_axes(initialized=False), 0),
            out_axes=self._inner_vmap_axes(initialized=True),
        )(self, xs)
        return cast(VmappedMCMC, self)

    def get_slice(self, indices, excluded_node_types) -> Self:
        sliced_algs = cast(
            type(self),
            tree_map(
                lambda x: x if isinstance(x, excluded_node_types) else x[indices],
                self,
                is_leaf=lambda x: isinstance(x, excluded_node_types),
            ),
        )
        return sliced_algs

    def set_slice(self, indices, slice, excluded_node_types) -> Self:
        sliced_algs = cast(
            type(self),
            tree_map(
                lambda x, y: (
                    x if isinstance(x, excluded_node_types) else x.at[indices].set(y)
                ),
                self,
                slice,
                is_leaf=lambda x: isinstance(x, excluded_node_types),
            ),
        )
        return sliced_algs

    def run_and_update_init(
        self, key: KeyArray
    ) -> Tuple[Self, InferenceAlgorithmResults]:
        if self.passthrough:
            return super().run_and_update_init(key)

        nta, results = vmap(
            MCMCAlgorithm.run_and_update_init, in_axes=(self._inner_vmap_axes(), 0)
        )(self, key)

        return cast(Self, nta), results

    @property
    def num_algs(self):
        if self.single_chains is None or self.single_chains.init_state is None:
            raise ValueError("num_algs is not defined for an uninitialized VmappedMCMC")
        else:
            return self.single_chains.init_state.x.shape[0]

    # pyright cannot seem to detect that this does not violate subtype polymorphism
    def estimate_expectation_of(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        f: Callable[[Array, Unpack[T]], Array],
        key: KeyArray,
        *extra_args: Unpack[T],
    ) -> Tuple[Self, Array, SingleChainResults[State_T, Info_T]]:
        def func(alg, k, *ea: Unpack[T]):
            return MCMCAlgorithm.estimate_expectation_of(alg, lambda x: f(x, *ea), k)

        self, f_avg, single_chain_results = vmap(
            func,
            in_axes=(
                self._inner_vmap_axes(),
                0,
                *(0,) * len(extra_args),
            ),
            out_axes=(self._inner_vmap_axes(), 0, 0),
        )(self, key, *extra_args)
        return cast(Self, self), f_avg, single_chain_results

    def reset_at_final_state(self, final_state: State_T) -> Self:
        if self.passthrough:
            return super().reset_at_final_state(final_state)

        self = vmap(
            MCMCAlgorithm.reset_at_final_state,
            in_axes=(self._inner_vmap_axes(), 0),
            out_axes=(self._inner_vmap_axes()),
        )(self, final_state)
        return cast(Self, self)

    def reparametrize(self, map: AffineTransform) -> Self:
        if self.passthrough:
            super(VmappedMCMC, self).reparametrize(map)

        self = self.replace(passthrough=True)
        self = vmap(
            MCMCAlgorithm.reparametrize,
            in_axes=(self._inner_vmap_axes(), None),
            out_axes=(self._inner_vmap_axes()),
        )(self, map)
        self = self.replace(passthrough=False)
        return cast(Self, self)
