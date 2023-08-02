"""(Re)generate (S)UNLE inference results using the current state of the codebase

To generate the results, run
`python -m tests.tests_backward_reproducibility.generate_test_data` from the `unle/`
folder, which has the following syntax:
```bash
python -m tests.tests_backward_reproducibility.generate_test_data [--large-budget] [--method <method_list>]
```

Here, `<method_list>` is any combination of `unle`, `sunle` and `sunle_vi`
(example: `unle sunle`). The `--large-budget` flag is used to adapt the (S)UNLE
parameters for a larger compute budget (~3min of total training time on a standard
CPU for each setting). Setting this flag should yield accurate inference results for
(S)UNLE on the set of parameters used to generate the results. However, this flag is
turned off in the CI-based results-generating workflows, as CI runners have limited
compute resources, and the main goal of these tests is to ensure stability of the
results (robustness and accuracy will be tested in separate workflows). The results
will be saved as a pickle file in the
`unle/tests/tests_backward_reproducibility/data/local/<method>` folder, for each
method `<method>`.
"""
import argparse
import os
from pathlib import Path
from typing import Literal

import cloudpickle
from jax.config import config

from . import test_utils

config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--method",
    nargs="+",
    default=["unle", "sunle", "sunle_vi"],
    type=str,
    choices=["unle", "sunle", "sunle_vi"],
    help="sunle and or unle and or sunle_vi",
)
parser.add_argument(
    "--large-budget",
    action="store_true",
    default=False,
    help="generate data in large training/simulation/inference budget settings",
)

parser.add_argument(
    "--save-entire-result",
    action="store_true",
    default=False,
    help=(
        "save the entire result object returned by `run` instead of just the posterior"
        "samples. This is useful to introspect the training process"
    ),
)


def run_and_save(
    param_set: dict,
    method: Literal["unle", "sunle", "sunle_vi"],
    save_entire_result: bool,
):
    """
    Run (S)UNLE for a given set of parameters and save its output.

    The generated data can serve as canonical test data to ensure reproducibility
    is preserved across code changes (when it makes sense that it should be, like
    during refactoring).
    """
    runner = "ci-runner" if os.environ.get("CI") == "true" else "local"
    data_output_dir = Path(__file__).parent.absolute() / "data" / runner / method
    data_output_dir.mkdir(parents=True, exist_ok=True)

    expected_filename = test_utils.get_data_filename(param_set)
    print(f"Saving data in {data_output_dir}/{expected_filename}")

    from sbibm_unle_extra.unle import run

    ret = run(**param_set)

    if save_entire_result:
        with open(data_output_dir / expected_filename, "wb") as f:
            cloudpickle.dump(ret, f)
    else:
        # For the reproducibility tests to be robust to refactoring changes,
        # the pickle-file should not contain `unle`-specific classes/objects
        # as such files could not be read if the classes/objects contained within
        # them were changed. For this reason, we only save the posterior samples
        # and the which are simple arrays.
        num_rounds = len(ret.train_results.config.num_samples)
        posterior_samples_all_rounds = [
            ret.train_results.get_posterior_samples(i) for i in range(num_rounds)
        ]
        with open(data_output_dir / expected_filename, "wb") as f:
            cloudpickle.dump(posterior_samples_all_rounds, f)


def run_in_parallel(methods, large_budget, save_entire_result):
    from concurrent.futures import ProcessPoolExecutor

    e = ProcessPoolExecutor(max_workers=os.cpu_count())
    futures = []
    for method in methods:
        param_sets = test_utils.get_param_sets(method, large_budget)
        for param_set in param_sets:
            f = e.submit(run_and_save, param_set, method, save_entire_result)
            futures.append(f)

    for f in futures:
        f.result()

    e.shutdown()


if __name__ == "__main__":
    args = parser.parse_args()
    run_in_parallel(args.method, args.large_budget, args.save_entire_result)
