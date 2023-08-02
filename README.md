This repository contains the code to reproduce the results of the paper "Maximum Likelihood Learning of Unnormalized Models for Simulation-Based Inference"


## Requirements:

- A Linux Machine
- Optionally (recommended) A GPU, along with cuda driver libraries.

To reproduce the experiments, execute the `paper/install.sh` script using `cd paper && source install.sh --gpu` if you have a GPU, or `cd paper && source install.sh` otherwise.
The `install.sh` script will install `Mambaforge` (a conda-based package manager) locally, and create a local `conda` environment named `unle-gpu` (resp. `unle-cpu`) containing all the necessary dependencies. The `python` executable that shall be used to interact with the codebase is `.env/unle-gpu/bin/python` folder.
We strongly recommend that you use a GPU when running the experiments; using a GPU yields considerable speedups for training and inference.
All the experiment submission / visualisation scripts take the form of a `jupyter` notebook. **No `jupyter` notebook engine is not provided** as part of the environment. You can either install `jupyter-notebook`/`jupyterlab` in this environment directly (by running the bash command `conda install -n unle-gpu jupyterlab`), or register the `python` executable of the `unle-gpu` environment to an external `jupyterlab` engine. In the latter case, the aforementioned CUDA environment variables need to be specified in the `share/jupyter/kernels/unle-gpu/kernel.json` (this file being relative to the jupyterlab environment root folder).
Here is an example `kernel.json` file that does so. You will need to change the placeholder paths indicated using </the/following/convention>:

```json
{
 "argv": [
  "</path/to/unle-gpu/env>/bin/python",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "unle-gpu",
 "language": "python",
 "metadata": {
  "debugger": true
 },
 "env": {
   "PATH":"</path/to/unle-gpu/env>/bin:$PATH",
   "XLA_PYTHON_CLIENT_PREALLOCATE":"false",
   "XLA_PYTHON_CLIENT_ALLOCATOR":"platform"
  }
}
```


## Running `unle` on `sbibm`-like `Task`

To do so, you can use the `sbibm_unle_extra.unle.run`.
We recommend using the following hyperparameters:

### Amortized UNLE (UNLE)

```python
ret = run(
    task=<task>
    num_samples=<num_samples>,
    num_observation=<num_observation>,
    num_smc_steps=20,
    num_mala_steps=3,
    use_warm_start=True,
    learning_rate=0.01,  ## better to grid search on [0.01, 0.001, 0.0001]
    max_iter=2000,
    weight_decay=0.1,
    random_seed=42,
    # default AUNLE arguments
    sampler="smc",
    num_particles=1000,
    batch_size=1000,
    num_posterior_samples=10000,
    use_nuts=False,
    init_proposal="prior",
    noise_injection_val=0.0005,
    proposal="prior+noise",
    inference_sampler="smc",
    ebm_model_type="joint_tilted",
    inference_proposal="prior",
    use_data_from_past_rounds=False,
    evaluate_posterior=True
)
```

### Sequential UNLE (SUNLE)

```python
ret = run(
    task=<task>
    num_samples=<num_samples>,
    num_observation=<num_observation>,
    ebm_model_type="likelihood",
    num_smc_steps=5,
    num_mala_steps=50,
    use_warm_start=True,
    learning_rate=0.01,  ## better to grid search on [0.01, 0.001, 0.0001]
    max_iter=500,
    weight_decay=0.1,
    # default SUNLE arguments
    random_seed=42,
    sampler="mala",
    num_particles=1000,
    batch_size=1000,
    use_nuts=False,
    init_proposal="prior",
    noise_injection_val=0.0005,
    proposal="data",
    inference_sampler="mala",
    inference_proposal="prior",
    use_data_from_past_rounds=True,
    inference_num_warmup_steps=500,
    exchange_mcmc_inner_sampler_num_steps=100,
    evaluate_posterior=False,
    estimate_loss=False,
    estimate_log_normalizer=False
)
```


## Developer Instructions


### Type Checking and Formatting

This project is type-checked by [`pyright`](https://microsoft.github.io/pyright/#/), and formatted using [`black`](https://black.readthedocs.io/en/stable/).
Both tools are run regularly on pull requests and on pushes to `main`. If you plan to contribute to this project, make sure that your changes are
black compliant by setting up [pre-commit](https://pre-commit.com/) for this repository by running the following commands from `unle`'s root directory:

```bash
pip install pre-commit
pre-commit install
```

### Testing Backward Reproducibility

#### Testing Backward Reproducibility in automated CI workflows

In order to ensure that refactoring changes do not affect the numerical results produced by (S)UNLE, we expose a github actions workflow that runs (S)UNLE using a fixed set of parameters
and compares the results against a reference set of results that is saved in the repository. This workflow can be triggered from a pull request by adding a commit to the PR branch
with commit message `ci test backward reproducibility` (`git commit --allow-empty -m "ci test backward reproducibility"`).

In the event when the pull request perform changes that are expected to affect the numerical values produced by (S)UNLE, it is necessary to update the reference results using the state
of the codebase present in the pull request. To do so, add a commit to the PR branch with commit message `ci generate results` (`git commit --allow-empty -m "ci generate results"`).
This commit will trigger a github actions workflow that runs (S)UNLE using the current codebase state, and updates the PR with the new reference results. Once the workflow has finished
running, you can update your local branch using `git fetch origin && git rebase origin/<branch_name>`.

#### Testing backward Reproducibility locally

It is also possible to (re)generate the reference results locally by running the command `python -m tests.tests_backward_reproducibility.generate_test_data`
from the root folder, which has the following syntax:

```bash
python -m tests.tests_backward_reproducibility.generate_test_data [--large-budget] [--save-entire-result] [--method <method_list>]
```

Here, `<method_list>` is any combination of `unle`, `sunle` and `sunle_vi` (example: `unle sunle`). The `--large-budget` flag is used to adapt the (S)UNLE parameters for a larger
compute budget (~3min of total training time on a standard CPU for each setting). Setting this flag should yield accurate inference results for (S)UNLE on the set of parameters used to generate
the results. However, this flag is turned off in the CI-based results-generating workflows, as CI runners have limited compute resources, and the main goal of these tests is to
ensure stability of the results (robustness and accuracy will be tested in separate workflows). The results will be saved as a pickle file in
the `./tests/tests_backward_reproducibility/data/local/<method>` folder, for each method `<method>`. Specifying the `--save-entire-result` flag (off by default) will save the entire
result object returned by (S)UNLE. This flag is off by default since the pickle file would not be robust to refactoring changes in the codebase.

Finally, it is possible to inspect the current set of reference results using the `python -m tests.tests_backward_reproducibility.plot_test_data` command from
the root folder, which has the same syntax as the data generation one, e.g:

```bash
python -m tests.tests_backward_reproducibility.plot_test_data [--large-budget] [--method <method_list>]
```

Running this command will save `html` plots in the `./tests/tests_backward_reproducibility/plots/` folder comparing the pairwise marginals of (S)UNLE posterior samples
and the true posterior samples for the specified settings. Setting `--large-budget` should yield accurate results.
