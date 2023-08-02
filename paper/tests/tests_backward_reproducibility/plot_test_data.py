"""Plot the saved (S)UNLE inference results.

Plots arg generated using Altair, and will be saved as a .html file.

Example usage (from the `unle/` folder):

```bash
python -m tests.tests_backward_reproducibility.plot_test_data [--large-budget] [--method <method_list>]
```

Running this command will save `html` plots in the
`unle/tests/tests_backward_reproducibility/plots/` folder comparing the pairwise
marginals of (S)UNLE posterior samples and the true posterior samples for the
specified settings. Setting `--large-budget` should yield accurate results.
"""
import argparse
from pathlib import Path

import deneb as den
import numpy as np
import torch
from sbibm.visualisation import fig_posterior

from . import test_utils

PLOTS_DIR = Path(__file__).parent.absolute() / "plots"

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


if __name__ == "__main__":
    args = parser.parse_args()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for method in args.method:
        param_sets = test_utils.get_param_sets(method, args.large_budget)
        for param_set in param_sets:
            posterior_samples_all_rounds = test_utils.load_test_data(param_set, method)
            fig = fig_posterior(
                task_name="two_moons",
                samples_tensor=torch.tensor(np.array(posterior_samples_all_rounds[-1])),
                title=f"{method}",
                num_samples=min(1000, len(posterior_samples_all_rounds[-1])),
            )
            assert fig is not None
            den.set_style(width=200, height=200)
            fig.save(
                Path(PLOTS_DIR)
                / f"{method}_large_budget_{args.large_budget}.posterior.html"
            )
