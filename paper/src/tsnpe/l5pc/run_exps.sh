# python multiround_pyloric.py cores=32 sims_per_round=30000 num_initial=30000 num_rounds=15 sampling_method=rejection ensemble_size=1
# cores: for simulating samples
OMP_NUM_THREADS=1 MKL_NUM_THREADS=4 python l5pc/multiround_pyloric.py cores=32 sims_per_round=10000 num_initial=50000 num_rounds=8 sampling_method=rejection ensemble_size=1 num_predictives=100 hydra.run.dir=results/p31_4/multiround/rej_8_rounds_2 hydra.sweep.dir=results/p31_4/multiround/rej_8_rounds_2__multirun

# python multiround_pyloric.py  cores=8 sims_per_round=30000 num_initial=30000 num_rounds=10 sampling_method=sir ensemble_size=1 start_round=8 path_to_prev_inference=2022_05_16__08_53_55__multirun/0
OMP_NUM_THREADS=1 MKL_NUM_THREADS=4 python l5pc/multiround_pyloric.py  cores=32 sims_per_round=10000 num_initial=10000 num_rounds=10 sampling_method=sir ensemble_size=1 start_round=8 num_predictives=100 path_to_prev_inference=rej_8_rounds_2 hydra.run.dir=results/p31_4/multiround/sir_10_rounds_2 hydra.sweep.dir=results/p31_4/multiround/sir_10_rounds_2__multirun

