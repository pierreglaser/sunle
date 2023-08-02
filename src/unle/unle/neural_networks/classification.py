from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Tuple

from flax import struct

if TYPE_CHECKING:
    from jax import Array
    from jax.random import KeyArray

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging
from flax.training import train_state

logging.set_verbosity(logging.INFO)


class ClassificationTrainingConfig(struct.PyTreeNode):
    max_iter: int = struct.field(pytree_node=False, default=200)
    learning_rate = 1e-3
    weight_decay = 1e-1


@jax.jit
def apply_model(
    state: train_state.TrainState, images: Array, labels: Array, class_weights: Array
):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, images)
        one_hot = jax.nn.one_hot(labels, 2)
        class_weights_arr = class_weights[1] * (labels == 1) + class_weights[0] * (
            labels == 0
        )

        loss = jnp.average(
            optax.softmax_cross_entropy(logits=logits, labels=one_hot),  # type: ignore
            weights=class_weights_arr,
        )
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)

    accuracy_0 = jax.lax.cond(
        (labels == 0).sum() > 0,
        lambda: ((jnp.argmax(logits, -1) == labels) * (labels == 0)).sum()
        / (labels == 0).sum(),
        lambda: 1.0,
    )

    accuracy_1 = jax.lax.cond(
        (labels == 1).sum() > 0,
        lambda: ((jnp.argmax(logits, -1) == labels) * (labels == 1)).sum()
        / (labels == 1).sum(),
        lambda: 1.0,
    )

    return grads, loss, (accuracy, accuracy_0, accuracy_1)


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_epoch(
    state: train_state.TrainState,
    train_dataset: Tuple[Array, Array],
    batch_size: int,
    key: KeyArray,
    class_weights: Array,
) -> Tuple[train_state.TrainState, Array, Tuple[Array, ...]]:
    """Train for a single epoch."""

    train_dataset_size = len(train_dataset[0])
    steps_per_epoch = max(train_dataset_size // batch_size, 1)

    perms = jax.random.permutation(key, len(train_dataset[0]))
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch

    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []
    epoch_a0 = []
    epoch_a1 = []

    for perm in perms:
        batch_images = train_dataset[0][perm, ...]
        batch_labels = train_dataset[1][perm, ...]
        grads, loss, (accuracy, a0, a1) = apply_model(
            state, batch_images, batch_labels, class_weights
        )
        state = update_model(state, grads)
        epoch_loss.append(loss)

        epoch_accuracy.append(accuracy)
        epoch_a0.append(a0)
        epoch_a1.append(a1)

    train_loss = jnp.asarray(np.mean(epoch_loss))

    train_accuracy = jnp.asarray(np.mean(epoch_accuracy))
    a0 = jnp.asarray(np.mean(epoch_a0))
    a1 = jnp.asarray(np.mean(epoch_a1))

    return state, train_loss, (train_accuracy, a0, a1)


def create_train_state(
    key: KeyArray,
    X: Array,
    apply_fn: Callable,
    init_fn: Callable,
    config: ClassificationTrainingConfig,
) -> train_state.TrainState:
    """Creates initial `TrainState`."""
    params = init_fn(key, jnp.ones_like(X))["params"]
    tx = optax.adamw(
        learning_rate=config.learning_rate, weight_decay=config.weight_decay
    )
    return train_state.TrainState.create(apply_fn=apply_fn, params=params, tx=tx)


def train_classifier(
    params: Array,
    y: Array,
    config: ClassificationTrainingConfig,
    apply_fn: Callable,
    init_fn: Callable,
) -> train_state.TrainState:
    from sklearn.model_selection import train_test_split

    theta_train, theta_test, y_train, y_test = train_test_split(
        params, y, random_state=43, stratify=y, train_size=0.8
    )
    assert not isinstance(theta_train, list)
    assert not isinstance(y_train, list)

    key = jax.random.PRNGKey(0)

    key, init_key = jax.random.split(key)

    state = create_train_state(init_key, theta_train, apply_fn, init_fn, config)

    batch_size = 10000
    batch_size = min(batch_size, theta_train.shape[0])

    max_iter = config.max_iter

    class_weights = jnp.array(
        [1 / (y_train == 0).sum(), 1 / (y_train == 1).sum()]  # type: ignore
    )
    for epoch in range(1, max_iter):
        key, input_key = jax.random.split(key)
        state, train_loss, (train_accuracy, train_a0, train_a1) = train_epoch(
            state, (theta_train, y_train), batch_size, input_key, class_weights
        )
        _, test_loss, (test_accuracy, test_a0, test_a1) = apply_model(
            state, theta_test, y_test, class_weights
        )

        if (epoch % max(max_iter // 20, 1)) == 0:
            logging.info(
                "epoch:% 3d, train_a0: %.4f, train_a1: %.4f, test_a0: %.4f, test_a1:"
                " %.4f, test_accuracy: %.4f"
                % (
                    epoch,
                    train_a0 * 100,
                    train_a1 * 100,
                    test_a0 * 100,
                    test_a1 * 100,
                    test_accuracy * 100,
                )
            )

    return state
