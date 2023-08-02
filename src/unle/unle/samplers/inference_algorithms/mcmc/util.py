from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tqdm.std import tqdm as tqdm_tp

import jax
from jax.experimental import host_callback


def progress_bar_factory(tqdm_bar: tqdm_tp, num_iters):
    """Factory that builds a progress bar decorator along
    with the `set_tqdm_description` and `close_tqdm` functions
    """
    if num_iters > 100:
        print_rate = int(num_iters / 100)
    else:
        print_rate = 1

    total_calls = 0

    def _update_tqdm(arg, transform):
        nonlocal total_calls
        total_calls += 1
        thinning, chain_ids = arg
        num_chains = len(chain_ids)

        tqdm_bar.set_description("Running chain", refresh=True)
        tqdm_bar.update(thinning * num_chains)

    def _close_tqdm(arg, transform):
        tqdm_bar.close()

    def _update_progress_bar(iter_num, chain_id) -> int:
        """Updates tqdm progress bar of a JAX loop only if the iteration number is
        a multiple of the print_rate.

        Usage: carry = progress_bar((iter_num, print_rate), carry)
        """
        ret = jax.lax.cond(
            (iter_num + 1) % print_rate == 0,
            lambda: host_callback.id_tap(
                _update_tqdm, (print_rate, chain_id), result=iter_num + 1
            ),
            lambda: iter_num,
        )
        _ = jax.lax.cond(
            iter_num == num_iters - 1,
            lambda: host_callback.id_tap(
                _close_tqdm, (print_rate, chain_id), result=iter_num + 1
            ),
            lambda: iter_num,
        )
        return ret

    return _update_progress_bar
