#!/usr/bin/env python
# coding: utf-8

# In[1]:


from experiments_utils import run_maybe_remotely
from sbibm_unle_extra.unle import run as run_unle


# In[2]:


### SUNLE with DIVI
for seed in (1, 2, 3):
    for task in ("slcp", "two_moons", "lotka_volterra", "gaussian_linear_uniform",):
        for ns in (
            (100,) * 10,
            (1000,) * 10,
            (10000,) * 10,
        ):
            for no in list(range(1, 10)):
                _ = run_maybe_remotely(
                    run_unle,
                    folder_name="icml",
                    experience_name="sunle",
                    use_slurm=True,
                    slurm_kwargs={},
                    task=task,
                    num_samples=ns,
                    num_observation=no,
                    num_smc_steps=5,
                    num_mala_steps=50,
                    use_warm_start=True,
                    learning_rate=0.001 if task == "lotka_volterra" else 0.01,
                    max_iter=10 if task=="gaussian_linear_uniform" and (ns==(100,)*10 or ns==(1000,)*10) else 500,
                    weight_decay=0.1,
                    random_seed=seed,
                    sampler="mala",
                    num_particles=1000,
                    batch_size=1000,
                    num_posterior_samples=10000,
                    use_nuts=False,
                    init_proposal="prior",
                    noise_injection_val=0.0005,
                    proposal="data",
                    inference_sampler="mala",
                    ebm_model_type="likelihood",
                    inference_proposal="prior",
                    use_data_from_past_rounds=True,
                    inference_num_warmup_steps=500,
                    exchange_mcmc_inner_sampler_num_steps=100,
                    evaluate_posterior=True,
                    estimate_loss=False,
                    estimate_log_normalizer=True
                )


# In[ ]:


### AUNLE                
for seed in (1, 2, 3):
    for ns in (
        (1000,),
        (10000,),
        (100000,),
    ):
        for no in list(range(1, 10)):
            for task in ("slcp", "two_moons", "lotka_volterra", "gaussian_linear_uniform",):
                _ = run_maybe_remotely(
                    run_unle,
                    folder_name="icml",
                    experience_name="aunle",
                    use_slurm=True,
                    slurm_kwargs={},
                    task=task,
                    num_samples=ns,
                    num_observation=no,
                    num_smc_steps=20,
                    num_mala_steps=3,
                    use_warm_start=True,
                    learning_rate=0.001 if task == "lotka_volterra" else 0.01,
                    max_iter=100 if task=="gaussian_linear_uniform" and (ns==(1000,) or ns==(10000,)) else 2000,
                    weight_decay=0.1,
                    random_seed=seed,
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

