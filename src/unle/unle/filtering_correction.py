from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from unle.distributions.base import ConditionalDistributionBase
from unle.neural_networks.classification import ClassificationTrainingConfig
from unle.neural_networks.neural_networks import MLP, MLPConfig
from unle.utils.jax import jittable_method_descriptor

if TYPE_CHECKING:
    from jax import Array

    from unle.typing import PyTreeNode

import jax
from absl import logging

logging.set_verbosity(logging.INFO)


class FilteringCorrector(ConditionalDistributionBase):
    params: PyTreeNode
    config: MLPConfig

    def __init__(
        self,
        params: PyTreeNode,
        config: MLPConfig,
        conditioned_event_shape: Optional[Tuple[int]] = None,
        param: Optional[Array] = None,
    ):
        super(FilteringCorrector, self).__init__(
            batch_shape=(),
            event_shape=(1,),
            conditioned_event_shape=conditioned_event_shape,
            condition=param,
            has_theta_dependent_normalizer=False,
        )

        self.params = params
        self.config = config

    @jittable_method_descriptor
    def log_prob(self, value):
        assert value.shape == () or value.shape == (1,)
        value = value.reshape(())

        logits = MLP(
            self.config.width,
            self.config.depth,
            activation=self.config.activation,
            num_outputs=self.config.num_outputs,
        ).apply({"params": self.params}, self.condition)
        log_probs = jax.nn.log_softmax(logits)

        return jax.lax.cond(
            value,
            lambda: log_probs[..., 1],
            lambda: log_probs[..., 0],
        )

    def tree_flatten(self):  # pyright: ignore [reportIncompatibleMethodOverride]
        base_params, base_aux = super(FilteringCorrector, self).tree_flatten()
        return (*base_params, self.params, self.config), base_aux

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        (*base_params, params, config) = params
        base_aux = aux_data
        obj = super(FilteringCorrector, cls).tree_unflatten(base_aux, base_params)
        obj.params = params
        obj.config = config
        return obj


def train_filtering_corrector(
    params: Array,
    y: Array,
    mlp_config: MLPConfig,
    training_config: ClassificationTrainingConfig,
) -> FilteringCorrector:
    from unle.neural_networks.classification import train_classifier

    def apply_fn(params, x):
        return MLP(
            mlp_config.width,
            mlp_config.depth,
            activation=mlp_config.activation,
            num_outputs=mlp_config.num_outputs,
        ).apply(params, x)

    def init_fn(key, x):
        return MLP(
            mlp_config.width,
            mlp_config.depth,
            activation=mlp_config.activation,
            num_outputs=mlp_config.num_outputs,
        ).init(key, x)

    state = train_classifier(
        params,
        y,
        training_config,
        apply_fn,
        init_fn,
    )
    return FilteringCorrector(state.params, mlp_config, (params.shape[1],))
