#!/bin/bash

# to train informarl (the graph version; aka our method)

# Slurm sbatch options
#SBATCH --job-name fafr_3_GPU
#SBATCH -a 0-1
#SBATCH --gres=gpu:volta:1
##SBATCH --cpus-per-task=8
## SBATCH -n 10 # use with MPI # max cores request limit: -c 48 * 24; -n 48 * 24
##SBATCH -c 48 # cpus per task


##export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# module unload anaconda/2022a
# Loading the required module
source /etc/profile
module load anaconda/2023a
# export LD_LIBRARY_PATH=/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib:$LD_LIBRARY_PATH

logs_folder="out_fair_informarl3"
mkdir -p $logs_folder
# Run the script
seed_max=2
n_agents=3
# graph_feat_types=("global" "global" "relative" "relative")
# cent_obs=("True" "False" "True" "False")
fair_wts=(1)
fair_rews=(1 1)

args_fair_wt=()
args_fair_rew=()

# iterate through all combos and make a list
for i in ${!fair_wts[@]}; do
    for j in ${!fair_rews[@]}; do
        args_fair_wt+=(${fair_wts[$i]})
        args_fair_rew+=(${fair_rews[$j]})
    done
done

seeds=(0 1)

# for seed in `seq ${seed_max}`;
# do
# # seed=`expr ${seed} + 3`
# echo "seed: ${seed}"
# # execute the script with different params
python -u onpolicy/scripts/train_mpe.py --use_valuenorm --use_popart \
--project_name "speedup_efficiency_tests_${n_agents}" \
--env_name "GraphMPE" \
--algorithm_name "rmappo" \
--seed ${seeds[$SLURM_ARRAY_TASK_ID]} \
--experiment_name "tanh5_fafr_nowalls_formation_nocollab_GPU_GoalMatch_1Fair_30goal_5mil" \
--scenario_name "nav_fairassign_fairrew_formation_graph" \
--fair_wt ${args_fair_wt[$SLURM_ARRAY_TASK_ID]} \
--fair_rew ${args_fair_rew[$SLURM_ARRAY_TASK_ID]} \
--num_agents=${n_agents} \
--num_landmarks=${n_agents} \
--collision_rew 30 \
--n_training_threads 1 --n_rollout_threads 128 \
--num_mini_batch 1 \
--episode_length 25 \
--total_actions 9 \
--num_env_steps 5000000 \
--ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
--user_name "marl" \
--use_cent_obs "False" \
--use_dones "False" \
--collaborative "False" \
--goal_rew 30 \
--num_walls 0 \
--zeroshift 5 \
--graph_feat_type "relative" \
--increase_fairness "False" \
--auto_mini_batch_size --target_mini_batch_size 8192 \
&> $logs_folder/tanh5_fairassign_fairrew_nowalls_GoalMatch_GPU_1Fair_30goal_5mil_${seeds[$SLURM_ARRAY_TASK_ID]}

