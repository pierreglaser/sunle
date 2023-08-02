import itertools
import shutil
from pathlib import Path

from dask.distributed import CancelledError

from sbibm_smnle import smnle

TECHNIQUES = ("SSM", "SM")[:1]

TASKS = (
    "two_moons",
    "gaussian_linear_uniform",
    "slcp",
    "lotka_volterra",
)

OBSERVATIONS = list(range(1, 11))[:5]

LRS = [0.01, 0.005, 0.001]

NUM_SAMPLES = [1000, 10000, 100000]

if __name__ == "__main__":


    close_after_simulation = True

    Path("dask-simulation-logs").mkdir(exist_ok=True)

    if shutil.which("srun"):
        from dask_jobqueue import SLURMCluster

        cluster = SLURMCluster(  # type: ignore
            n_workers=0,  # create the workers "lazily" (upon cluster.scale)
            memory="24GB",  # amount of RAM per worker
            processes=1,  # number of execution units per worker (threads and processes)
            cores=1,  # among those execution units, number of processes
            job_extra=[
                # '--export=ALL', # default behavior.
                "-p gpu",  # partition
                "--gres=gpu:1",  # number of GPUs per node
                "--output=dask-simulation-logs/R-%x.%j.out",
                "--error=dask-simulation-logs/R-%x.%j.err",
                *(),
                # '--export=OMP_NUM_THREADS=1,MKL_NUM_THREADS=1',
            ],
            # extra = ["--no-nanny"],
            # scheduler_options={'dashboard_address': ':8787', 'port': 45987, 'allowed_failures': 10},
            scheduler_options={
                "dashboard_address": ":8787",
                "allowed_failures": 10,
            },
            job_cpu=8,
            walltime="24:0:0",
        )
        print(cluster.job_script())
        cluster.adapt(minimum_jobs=1, maximum_jobs=40)
        # cluster.scale(20)
    else:
        from dask.distributed import LocalCluster
        cluster = LocalCluster(n_workers=4)

    from distributed import Client

    print('testing cluster')
    client = Client(cluster)
    print(client.submit(lambda x: x + 1, 10).result())
    print('cluster is working')

    import jax.numpy as jnp
    futs = [client.submit(lambda x: str(jnp.ones((2,)).device()), i) for i in range(50)]
    print([f.result() for f in futs])

    futures = []
    args = []

    grid = list(
        itertools.product(
            TECHNIQUES,
            TASKS,
            OBSERVATIONS,
            LRS,
            NUM_SAMPLES,
        )
    )

    for arg in grid:
        technique, task, num_observation, lr, num_samples = arg
        fut = client.submit(
            lambda orig_args, **kwargs: (orig_args, smnle(**kwargs)),
            arg,
            model=task,
            technique=technique,
            num_observation=num_observation,
            SM_lr=lr,
            SM_lr_theta=lr,
            # batch_size=min(num_samples, 10000),
            epochs=500,
            n_samples_training=num_samples,
            # epochs=500,
            # mcmc_num_warmup_steps=2000,
            mcmc_num_chains=100,
            mcmc_num_warmup_steps=5000,
            num_posterior_samples=10000,
            # batch_size=10000,
            cuda=True
        )
        futures.append(fut)
        args.append(arg)

    results ={}

    from dask.distributed import as_completed
    num_finished = 0
    num_remaining = len(futures)
    for fut in as_completed(futures):
        ret = fut.result()  # type: ignore
        arg, result  = ret

        num_finished += 1
        num_remaining -= 1
        print('finished', arg)
        print('num_finished', num_finished, 'num_remaining', num_remaining)


        results[arg] = result

        with open("results.pkl", "wb") as f:
            import pickle
            pickle.dump(results, f)

    print('done, shutting down')
    client.shutdown()
    cluster.close()

    import time
    time.sleep(10)
