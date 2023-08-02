python run_inference.py --method=prior_torch --task_name=dimension_reduced_multimodal_task --num_rounds=2 --seed=1 --num_seeds=100 --num_simulations=100
python run_coverage.py --method=prior_torch --task_name=dimension_reduced_multimodal_task --num_rounds=2 --seed=1 --num_seeds=100 --num_simulations=100
python plot_coverage.py --method=prior_torch --task_name=dimension_reduced_multimodal_task --num_rounds=2 --seed=1 --num_seeds=100 --num_simulations=100

python run_inference.py --method=prior_jax --task_name=dimension_reduced_multimodal_task --num_rounds=2 --seed=1 --num_seeds=100 --num_simulations=100
python run_coverage.py --method=prior_jax --task_name=dimension_reduced_multimodal_task --num_rounds=2 --seed=1 --num_seeds=100 --num_simulations=100
python plot_coverage.py --method=prior_jax --task_name=dimension_reduced_multimodal_task --num_rounds=2 --seed=1 --num_seeds=100 --num_simulations=100

python run_inference.py --method=prior_torch --task_name=dimension_reduced_multimodal_task --num_rounds=1 --seed=1 --num_seeds=1 --num_simulations=100
python run_coverage.py --method=prior_torch --task_name=dimension_reduced_multimodal_task --num_rounds=1 --seed=1 --num_seeds=1 --num_simulations=100
python plot_coverage.py --method=prior_torch --task_name=dimension_reduced_multimodal_task --num_rounds=1 --seed=1 --num_seeds=1 --num_simulations=100

python run_inference.py --method=prior_jax --task_name=dimension_reduced_multimodal_task --num_rounds=1 --seed=1 --num_seeds=1 --num_simulations=100
python run_coverage.py --method=prior_jax --task_name=dimension_reduced_multimodal_task --num_rounds=1 --seed=1 --num_seeds=1 --num_simulations=100
python plot_coverage.py --method=prior_jax --task_name=dimension_reduced_multimodal_task --num_rounds=1 --seed=1 --num_seeds=1 --num_simulations=100
