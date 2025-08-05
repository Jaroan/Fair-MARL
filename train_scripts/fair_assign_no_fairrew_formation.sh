#!/bin/bash

# to train informarl (the graph version; aka our method)

# Slurm sbatch options
#SBATCH --job-name unicycle
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

# "double_integrator" or "unicycle_vehicle"
dynamics_type="unicycle_vehicle"

seeds=(0 1)
datetime_str=$(date '+%y%m%d_%H%M%S')

if [ "$dynamics_type" == "unicycle_vehicle" ]; then
    str_dynamics_type="uv"
    world_size=4
elif [ "$dynamics_type" == "double_integrator" ]; then
    str_dynamics_type="di"
    world_size=4
else
    echo "Error: Unsupported dynamics type '$dynamics_type'"
    exit 1  # Exit with a non-zero status to indicate an error
fi

echo "datetime_str: ${datetime_str}"
echo "dynamics_type: ${dynamics_type}"
# for seed in `seq ${seed_max}`;
# do
# # seed=`expr ${seed} + 3`
# echo "seed: ${seed}"
# execute the script with different params
python -u onpolicy/scripts/train_mpe.py --use_valuenorm --use_popart \
--project_name "unicycle_dynamics_${n_agents}" \
--env_name "GraphMPE" \
--algorithm_name "rmappo" \
--seed ${seeds[$SLURM_ARRAY_TASK_ID]} \
--experiment_name "${str_dynamics_type}_${datetime_str}_fairassign_nofairrew_fa_nfr_nocollab_goalMatch_noFair_30_5mil" \
--scenario_name "nav_fairassign_nofairrew_formation_graph" \
--dynamics_type ${dynamics_type} \
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
--world_size=${world_size} \
--graph_feat_type "relative" \
--auto_mini_batch_size --target_mini_batch_size 8192 \
&> $logs_folder/${str_dynamics_type}_${datetime_str}_fairassign_nofairrew_fa_nfr_nocollab_goalMatch_noFair_30_5mil_${seeds[$SLURM_ARRAY_TASK_ID]}