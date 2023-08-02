#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

# We put here all commands needed to obtain the results presented in the paper. We remark that running this bash file
# as a single script may not be a good idea, as the different python scripts in here may have very different
# computational demands (some may require parallel computing, and you may choose to use GPU to train the NNs)
# Moreover, all Python scripts below have additional options, which you can list by:
#       python3 <file.py> -h

# CREATE TREE STRUCTURE:

NUM_SAMPLES=( 1000 )

for ((j=0;j<${#NUM_SAMPLES[@]};++j)); do

    num_samples=${NUM_SAMPLES[j]}

    mkdir -p results
    mkdir -p results/two_moons_${num_samples}
    mkdir -p results/two_moons_${num_samples}/ABC-FP
    mkdir -p results/two_moons_${num_samples}/ABC-SM
    mkdir -p results/two_moons_${num_samples}/ABC-SSM
    mkdir -p results/two_moons_${num_samples}/Exc-SM
    mkdir -p results/two_moons_${num_samples}/Exc-SM/10_inn_steps
    mkdir -p results/two_moons_${num_samples}/Exc-SM/30_inn_steps
    mkdir -p results/two_moons_${num_samples}/Exc-SM/100_inn_steps
    mkdir -p results/two_moons_${num_samples}/Exc-SM/200_inn_steps
    mkdir -p results/two_moons_${num_samples}/Exc-SSM
    mkdir -p results/two_moons_${num_samples}/Exc-SSM/10_inn_steps
    mkdir -p results/two_moons_${num_samples}/Exc-SSM/30_inn_steps
    mkdir -p results/two_moons_${num_samples}/Exc-SSM/100_inn_steps
    mkdir -p results/two_moons_${num_samples}/Exc-SSM/200_inn_steps
    mkdir -p results/two_moons_${num_samples}/net-FP
    mkdir -p results/two_moons_${num_samples}/net-SM
    mkdir -p results/two_moons_${num_samples}/net-SSM
    mkdir -p results/two_moons_${num_samples}/observations
    mkdir -p results/two_moons_${num_samples}/PMC-RE
    mkdir -p results/two_moons_${num_samples}/PMC-SL


    # GENERATE OBSERVATIONS AND EXACT POSTERIORS

    n_observations=1
    python3 scripts/generate_obs.py two_moons --n_observations $n_observations --root_folder results/two_moons_${num_samples}/


    # TRAIN THE NNs (amortized over observations)
    python3 scripts/train_net.py SSM two_moons --nets_folder net-SSM --epochs 1000 --lr_data 0.01 --lr_theta 0.01 \
        --update --bn_mom 0.9 --epochs_before_early_stopping 1000 --epochs_test_interval 10 --save_train_data --n_samples_training 1000 --n_samples_evaluation 1000 --root_folder results/two_moons_${num_samples}/

    # python3 plot_scripts/plot_learned_stats.py two_moons --nets_folder net-SSM --no_bn --n_obs 1000
    # python3 plot_scripts/plot_learned_nat_params.py two_moons --nets_folder net-SSM --n_obs 1000


    # python3 scripts/train_net.py SM two_moons --nets_folder net-SM --epochs 500 --lr_data 0.001 --lr_theta 0.001 \
    #     --update --bn_mom 0.9 --epochs_before_early_stopping 150 --epochs_test_interval 10

    # python3 plot_scripts/plot_learned_stats.py two_moons --nets_folder net-SM --no_bn --n_obs 1000
    # python3 plot_scripts/plot_learned_nat_params.py two_moons --nets_folder net-SM --n_obs 1000

    # python3 scripts/train_net.py FP two_moons --nets_folder net-FP --epochs 1000 --lr_data 0.001 --epochs_before_early_stopping 300 --epochs_test_interval 25 --no_scheduler
    # python3 plot_scripts/plot_learned_stats.py two_moons --nets_folder net-FP --no_bn --n_obs 1000 --FP



    # 
    # # INFERENCES WITH Exc-SM AND Exc-SSM WITH DIFFERENT NUMBER OF INNER MCMC STEPS
    # 
    model=two_moons
    # inn_steps_values=( 10 30 100 200 )
    # METHODS=( SM SSM )
    inn_steps_values=( 100 )
    METHODS=( SSM )
    prop_size=0.1
    K=0  # bridging steps
    burnin=1000
    n_samples=1000
    tune_window=100

    for ((j=0;j<${#METHODS[@]};++j)); do
        METHOD=${METHODS[j]}
        net_f=net-${METHOD}
        for ((k=0;k<${#inn_steps_values[@]};++k)); do

            inner_steps=${inn_steps_values[k]}
            inf_f=Exc-${METHOD}/${inner_steps}_inn_steps

            python3 scripts/inferences.py ${METHOD} ${model} --burnin $burnin --n_samples $n_samples \
                --inference_folder $inf_f --nets_f $net_f \
                --root_folder results/two_moons_${num_samples}/ \
                --start 0 --n_obs $n_observations \
                --aux_MCMC_inner_steps_exchange_MCMC $inner_steps --bridging ${K} \
                --aux_MCMC_proposal_size_exchange_MCMC ${prop_size} \
                --tuning ${tune_window} \
                --deb warn \
                --propose_new_theta_exchange_MCMC transformation

        done
    done

    # # INFERENCES WITH ABC-SM, ABC-SSM and ABC-FP; this uses MPI to parallelize, with number of tasks given by NTASKS
    # NTASKS=8  # adapt this to your setup
    # start_obs=0
    # ABC_algorithm=SABC
    # ABC_steps=100
    # n_samples=1000
    # ABC_eps=100000000
    # n_observations=100
    # SABC_cutoff=0  # increase this for faster stop.
    # 
    # technique=FP
    # inference_folder=ABC-FP
    # nets_folder=net-FP
    # 
    # mpirun -n $NTASKS python3 scripts/inferences.py $technique $model \
    #          --use_MPI \
    #          --inference_technique ABC \
    #          --start_observation_index $start_obs \
    #          --n_observations $n_observations \
    #          --ABC_alg $ABC_algorithm \
    #          --ABC_steps $ABC_steps \
    #          --n_samples $n_samples \
    #          --inference_folder $inference_folder \
    #          --nets_folder $nets_folder \
    #          --ABC_full_output \
    #          --ABC_eps $ABC_eps \
    #          --SABC_cutoff $SABC_cutoff \
    #          --load_trace_if_available \
    #          --no_weighted_eucl_dist \
    #          --seed 42
    # 
    # for ((j=0;j<${#METHODS[@]};++j)); do
    #     METHOD=${METHODS[j]}
    #     net_f=net-${METHOD}
    # 
    #     inference_folder=ABC-${METHOD}
    #     nets_folder=net-${METHOD}
    # 
    #     mpirun -n $NTASKS python3 scripts/inferences.py ${METHOD} $model \
    #              --use_MPI \
    #              --inference_technique ABC \
    #              --start_observation_index $start_obs \
    #              --n_observations $n_observations \
    #              --ABC_alg $ABC_algorithm \
    #              --ABC_steps $ABC_steps \
    #              --n_samples $n_samples \
    #              --inference_folder $inference_folder \
    #              --nets_folder $nets_folder \
    #              --ABC_full_output \
    #              --ABC_eps $ABC_eps \
    #              --SABC_cutoff $SABC_cutoff \
    #              --load_trace_if_available \
    #              --seed 42
    # done
    # 
    # # INFERENCES WITH PMC-RE, PMR-SL
    # 
    # start_obs=0
    # steps=10
    # n_samples=1000
    # n_observations=100
    # 
    # n_samples_per_param=100
    # technique=SL
    # inference_folder=PMC-SL
    # mpirun -n $NTASKS python3 scripts/SL_RE_experiments.py $technique $model \
    #          --use_MPI \
    #          --start_obs $start_obs \
    #          --n_observations $n_observations \
    #          --steps $steps \
    #          --n_samples $n_samples \
    #          --n_samples_per_param $n_samples_per_param \
    #          --inference_folder $inference_folder \
    #          --full_output \
    #          --load_trace_if_available \
    #          --seed 2
    # 
    # 
    # n_samples_per_param=1000
    # technique=RE
    # inference_folder=PMC-RE
    # mpirun -n $NTASKS python3 scripts/SL_RE_experiments.py $technique $model \
    #          --use_MPI \
    #          --start_obs $start_obs \
    #          --n_observations $n_observations \
    #          --steps $steps \
    #          --n_samples $n_samples \
    #          --n_samples_per_param $n_samples_per_param \
    #          --inference_folder $inference_folder \
    #          --full_output \
    #          --load_trace_if_available \
    #          --seed 2
    # 
    # 
    # # PRODUCE PLOTS
    # # compare the different number inner MCMC steps:
    # python3 plot_scripts/wass_dist_diff_inner_MCMC_steps.py two_moons --n_samples 10000
    # # single bivariate plot for the observation in the paper:
    # python3 plot_scripts/plot_bivariate.py SM two_moons --inference_folder Exc-SM/30_inn_steps --start 20 --n_obs 21
    # python3 plot_scripts/plot_bivariate.py SSM two_moons --inference_folder Exc-SSM/30_inn_steps --start 20 --n_obs 21
    # python3 plot_scripts/plot_bivariate.py SM two_moons --inference_folder ABC-SM --start 20 --n_obs 21 --inference_technique ABC --n_samples 1000
    # python3 plot_scripts/plot_bivariate.py SSM two_moons --inference_folder ABC-SSM --start 20 --n_obs 21 --inference_technique ABC --n_samples 1000
    # python3 plot_scripts/plot_bivariate.py True two_moons --inference_folder observations --start 20 --n_obs 21
    # # final wass distance plot:
    # n_observations=100
    # python3 plot_scripts/compare_inference_results_diff_techniques.py two_moons final --n_obs $n_observations --subsample_size 10000 --inference_folder_exchange_SM Exc-SM/30_inn_steps --inference_folder_exchange_SSM Exc-SSM/30_inn_steps
    # 
    # # Wass dist plots vs iterations; these may take a while
    # python3 plot_scripts/compare_inference_results_diff_techniques.py two_moons ABC --n_obs $n_observations --subsample_size 1000 --inference_folder_exchange_SM Exc-SM/30_inn_steps --load_exchange_SM_if_available --load_exchange_SSM_if_available
    # python3 plot_scripts/compare_inference_results_diff_techniques.py two_moons SL --n_obs $n_observations --subsample_size 1000 --inference_folder_exchange_SM Exc-SM/30_inn_steps --load_exchange_SM_if_available --load_exchange_SSM_if_available
    # python3 plot_scripts/compare_inference_results_diff_techniques.py two_moons RE --n_obs $n_observations --subsample_size 1000 --inference_folder_exchange_SM Exc-SM/30_inn_steps --load_exchange_SM_if_available --load_exchange_SSM_if_available

done
