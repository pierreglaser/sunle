from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Tuple

if TYPE_CHECKING:
    from jax import Array
    from jax.random import KeyArray
    from unle.typing import PyTreeNode

from typing import Optional

import jax
import jax.numpy as jnp
import optax
from absl import logging
from flax import struct
from flax.training import train_state

logging.set_verbosity(logging.INFO)


class RegressionTrainingConfig(struct.PyTreeNode):
    max_iter: int = struct.field(pytree_node=False, default=500)
    learning_rate: float = struct.field(pytree_node=True, default=0.001)
    batch_size: int = struct.field(pytree_node=False, default=10000)
    select_based_on_test_loss: bool = struct.field(pytree_node=False, default=False)
    use_l1_loss: bool = struct.field(pytree_node=False, default=False)


def initialize_train_state(
    key: KeyArray,
    X: Array,
    config: RegressionTrainingConfig,
    params: PyTreeNode,
    apply_fn: Callable,
    init_fn: Callable,
) -> train_state.TrainState:
    """Creates initial `TrainState`."""
    assert len(X.shape) == 2

    if params is None:
        key, subkey = jax.random.split(key)
        params = init_fn(subkey, jnp.ones_like(X[0, :]))["params"]
    tx = optax.adamw(config.learning_rate, weight_decay=1e-5)
    return train_state.TrainState.create(apply_fn=apply_fn, params=params, tx=tx)


@jax.jit
def apply_model(
    state: train_state.TrainState,
    X: Array,
    y: Array,
    config: RegressionTrainingConfig,
) -> Tuple[PyTreeNode, float]:
    def loss_fn(params):
        preds = jax.vmap(state.apply_fn, (None, 0))(params, X)
        if config.use_l1_loss:
            print("using l1 loss")
            loss = jnp.average(jnp.sum(jnp.abs(preds - y), axis=1))
        else:
            print("using l2 loss")
            loss = jnp.average(jnp.square(jnp.linalg.norm(preds - y, axis=1)))
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    return grads, loss


def train_epoch(
    state: train_state.TrainState,
    train_dataset: Tuple[Array, Array],
    key: KeyArray,
    config: RegressionTrainingConfig,
    steps_per_epoch: int,
    len_training_data: int,
) -> Tuple[train_state.TrainState, float]:
    """Train for a single epoch."""

    # XXX: doesn't take into account incomplete batchs
    perms = jax.random.permutation(key, len_training_data)[
        : steps_per_epoch * config.batch_size
    ]
    perms = perms.reshape((steps_per_epoch, config.batch_size))

    train_loss = 0.0

    for perm in perms:
        batch_X = train_dataset[0][perm, ...]
        batch_y = train_dataset[1][perm, ...]
        grads, loss = apply_model(state, batch_X, batch_y, config)
        state = state.apply_gradients(grads=grads)
        train_loss += loss / len(perms)

    return state, train_loss


def train_regressor(
    X: Array,
    y: Array,
    key: KeyArray,
    config: RegressionTrainingConfig,
    init_params: Optional[PyTreeNode] = None,
    apply_fn: Optional[Callable] = None,
    init_fn: Optional[Callable] = None,
) -> train_state.TrainState:
    assert apply_fn is not None
    assert init_fn is not None

    from sklearn.model_selection import train_test_split

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(X, y, random_state=43, train_size=0.8)
    assert not isinstance(X_train, list)

    state = initialize_train_state(key, X_train, config, init_params, apply_fn, init_fn)

    config = config.replace(batch_size=min(config.batch_size, X_train.shape[0]))
    steps_per_epoch = max(X_train.shape[0] // config.batch_size, 1)

    jitted_train_epoch = jax.jit(
        train_epoch,
        static_argnums=(
            4,
            5,
        ),
    )
    prev_test_loss = 1e30
    prev_state = None

    for epoch in range(1, config.max_iter):
        key, subkey = jax.random.split(key)
        state, train_loss = jitted_train_epoch(
            state,
            (X_train, y_train),
            subkey,
            config,
            steps_per_epoch,
            X_train.shape[0],
        )
        _, test_loss = apply_model(state, X_test, y_test, config)

        if config.select_based_on_test_loss:
            if epoch > 1:
                if test_loss > prev_test_loss:
                    print(
                        "test loss increased, stopping training at epoch ", str(epoch)
                    )
                    state = prev_state
                    break
            prev_test_loss = test_loss
            prev_state = state

        if (epoch % max(config.max_iter // 20, 1)) == 0:
            logging.info(
                "epoch:% 3d, train_loss: %.4f, test_loss: %.4f"
                % (epoch, train_loss, test_loss)
            )

    assert state is not None
    return state
