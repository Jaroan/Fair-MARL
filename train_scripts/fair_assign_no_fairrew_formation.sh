#!/bin/bash

# to train informarl (the graph version; aka our method)

# Slurm sbatch options
#SBATCH --job-name fanfr_3_GPU
#SBATCH -a 0-1
#SBATCH --gres=gpu:volta:1
## SBATCH -n 10 # use with MPI # max cores request limit: -c 48 * 24; -n 48 * 24
##SBATCH -c 48 # cpus per task

# Loading the required module
source /etc/profile
module load anaconda/2023a
# export LD_LIBRARY_PATH=/state/partition1/llgrid/pkg/anaconda/anaconda3-2022b/lib:$LD_LIBRARY_PATH

logs_folder="out_fair_informarl3"
mkdir -p $logs_folder
# Run the script
seed_max=2
n_agents=3

seeds=(0 1)

# for seed in `seq ${seed_max}`;
# do
# # seed=`expr ${seed} + 3`
# echo "seed: ${seed}"
# execute the script with different params
python -u onpolicy/scripts/train_mpe.py --use_valuenorm --use_popart \
--project_name "speedup_efficiency_tests_${n_agents}" \
--env_name "GraphMPE" \
--algorithm_name "rmappo" \
--seed ${seeds[$SLURM_ARRAY_TASK_ID]} \
--experiment_name "fairassign_nofairrew_fa_nfr_nocollab_goalMatch_noFair_30_5mil" \
--scenario_name "nav_fairassign_nofairrew_formation_graph" \
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
--auto_mini_batch_size --target_mini_batch_size 8192 \
&> $logs_folder/fairassign_nofairrew_fa_nfr_nocollab_goalMatch_noFair_30_5mil_${seeds[$SLURM_ARRAY_TASK_ID]}