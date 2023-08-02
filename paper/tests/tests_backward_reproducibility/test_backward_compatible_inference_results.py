from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Tuple

if TYPE_CHECKING:
    from sbibm_unle_extra.unle import TrainEvalTresults

import numpy as np
from jax.config import config

from . import test_utils

config.update("jax_enable_x64", True)


class TestBackwardCompatibleInferenceResults:
    """
    Test UNLE or SUNLE by comparing their inference results to reference ones.
    """

    def test_unle(self, param_set: dict):
        self._run_and_compare(param_set, "unle")

    def test_sunle(self, param_set: dict):
        self._run_and_compare(param_set, "sunle")

    def test_sunle_vi(self, param_set: dict):
        self._run_and_compare(param_set, "sunle_vi")

    def _run_and_compare(
        self, param_set: dict, method: Literal["unle", "sunle", "sunle_vi"]
    ):
        ref_posterior_samples_all_rounds = test_utils.load_test_data(param_set, method)
        from sbibm_unle_extra.unle import run

        ret = run(**param_set)
        num_rounds = len(ret.train_results.config.num_samples)
        posterior_samples_all_rounds = [
            ret.train_results.get_posterior_samples(i) for i in range(num_rounds)
        ]

        for round, (posterior_samples, ref_posterior_samples) in enumerate(
            zip(posterior_samples_all_rounds, ref_posterior_samples_all_rounds)
        ):
            assert np.allclose(
                posterior_samples,
                ref_posterior_samples,
            ), f"round: {round}"
