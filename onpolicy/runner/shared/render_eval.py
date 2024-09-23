import numpy as np
import torch
import time
import imageio
import csv

@torch.no_grad()
	def render(get_metrics:bool=False):
		"""
			Visualize the env.
			get_metrics: bool (default=False)
				if True, just return the metrics of the env and don't render.
		"""
		envs = envs
		reset_number = 0
		all_frames = []
		total_dists_traveled, total_time_taken = [], []
		rewards_arr, success_rates_arr, num_collisions_arr, frac_episode_arr, fairness_param = [], [], [], [],[]
		dist_mean_arr, time_mean_arr = [],[]
		stddev_param = []
		dists_trav_list = np.zeros((num_agents))
		time_taken_list = np.zeros((num_agents))
		time_fairness, time_stddev_param, time_mean = [], [], []

		###### store the fairness values in a CSV file ##########
		with open('/Users/jasmine/Jasmine/MIT/MARL/Codes/Team-Fair-MARL/'+str(all_args.model_name)+'_firstgoaldone_nogoal_fair_vs_success.csv', 'a', newline="") as f1:
			# create the csv writer
			writer = csv.writer(f1)
		#########################################################






        ######## fairmarl code evaluation block ########################
        # run for 100 episodes
		for episode in range(all_args.render_episodes):
			# print("episode", episode)
			obs, agent_id, node_obs, adj = envs.reset()
			if not get_metrics:
				if all_args.save_gifs:
					image = envs.render('rgb_array')[0][0]
					all_frames.append(image)
				else:
					envs.render('human')

			rnn_states = np.zeros((n_rollout_threads, 
									num_agents, 
									recurrent_N, 
									hidden_size), 
									dtype=np.float32)
			masks = np.ones((n_rollout_threads, 
							num_agents, 1), 
							dtype=np.float32)
			available_actions = np.ones((num_agents, 5), 
										dtype=np.float32)
			episode_rewards = []
			
			for step in range(episode_length):

				calc_start = time.time()
				zero_masks = masks[0] == 0

				if 	not zero_masks.all():
					available_actions = np.ones((num_agents, 5), 
										dtype=np.float32)
				# Broadcast the boolean mask to match the shape of available_actions
				broadcasted_zero_masks = np.broadcast_to(zero_masks, available_actions.shape)

				available_actions[zero_masks[:,0]] = np.array([1, 0, 0, 0, 0])
				self.trainer.prep_rollout()
				action, rnn_states = self.trainer.policy.act(
													np.concatenate(obs),
													np.concatenate(node_obs),
													np.concatenate(adj),
													np.concatenate(agent_id),
													np.concatenate(rnn_states),
													np.concatenate(masks),
													available_actions = available_actions,
													deterministic=True)
				actions = np.array(np.split(_t2n(action), n_rollout_threads))
				rnn_states = np.array(np.split(_t2n(rnn_states), 
									n_rollout_threads))

				if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
					for i in range(envs.action_space[0].shape):
						uc_actions_env = np.eye(
								envs.action_space[0].high[i]+1)[actions[:, :, i]]
						if i == 0:
							actions_env = uc_actions_env
						else:
							actions_env = np.concatenate((actions_env, 
														uc_actions_env), 
														axis=2)
				elif envs.action_space[0].__class__.__name__ == 'Discrete':
					actions_env = np.squeeze(np.eye(
											envs.action_space[0].n)[actions], 2)
				else:
					raise NotImplementedError

				# Obser reward and next obs
				obs, agent_id, node_obs, adj, \
					rewards,dones, infos, reset_count = envs.step(actions_env)

				episode_rewards.append(rewards)
				dones_env = np.all(dones)
				rnn_states[dones == True] = np.zeros(((dones == True).sum(), 
													recurrent_N, 
													hidden_size), 
													dtype=np.float32)
				masks = np.ones((n_rollout_threads, 
								num_agents, 1), 
								dtype=np.float32)
				masks[dones == True] = np.zeros(((dones == True).sum(), 1), 
												dtype=np.float32)
				dones_env = np.all(dones, axis=1)
				masks[dones_env == True] = np.ones(((dones_env == True).sum(), num_agents, 1), dtype=np.float32)
				if not get_metrics:
					if all_args.save_gifs:

						image = envs.render('rgb_array')[0][0]
						all_frames.append(image)
						calc_end = time.time()
						elapsed = calc_end - calc_start
						if elapsed < all_args.ifi:
							time.sleep(all_args.ifi - elapsed)
					else:
						envs.render('human')


				reset_number += reset_count
				if reset_count > 0:
					break

				if reset_number == all_args.render_episodes :
					break
            

            ########## end of fairmarl code evaluation block ####################






            ##########  evaluation data collection ############################
			env_infos = process_infos(infos)

			num_collisions = get_collisions(env_infos)
			frac, success,time_taken = get_fraction_episodes(env_infos)
			if np.any(frac==1):
				frac_max = 1.0
			else:
				frac_max = np.max(frac)
			rewards_arr.append(np.mean(np.sum(np.array(episode_rewards), axis=0)))
			frac_episode_arr.append(frac_max)
			success_rates_arr.append(success)
			num_collisions_arr.append(num_collisions)
			fairness_metric = get_fairness_metric(env_infos)

			fairness_param.append(fairness_metric[-1])

			dist_mean = get_dist_mean(env_infos)
			dist_mean_arr.append(dist_mean[-1])
			time_mean = get_time_mean(env_infos)
			time_mean_arr.append(time_mean[-1])
	
			dists_traveled = get_dists_traveled(env_infos)
			dists_trav_list +=dists_traveled
			time_taken_list +=time_taken

			total_dists_traveled.append(np.sum(dists_traveled))
			total_time_taken.append(np.sum(time_taken))

			csv_data1 = [num_obstacles, 
						num_agents,
						all_args.world_size,
						episode_length,
						all_args.render_episodes,
						reset_number, 
						step, 
						fairness_metric[-1],
						np.mean(success),
						frac_max,
						total_dists_traveled[-1],
						total_time_taken[-1],
						]
			###### store the fairness values in a CSV file ##########
			with open('/Users/jasmine/Jasmine/MIT/MARL/Codes/Team-Fair-MARL/'+str(all_args.model_name)+'_firstgoaldone_nogoal_fair_vs_success_new.csv', 'a', newline="") as f1:
				# create the csv writer
				writer = csv.writer(f1)
				writer.writerow(csv_data1)
			#########################################################



		# Calculate the statistics for the box whisker plot
		fair_minimum = np.min(fairness_param)
		fair_0_1_quantile = np.percentile(fairness_param, 10)
		fair_median = np.median(fairness_param)
		fair_0_9_quantile = np.percentile(fairness_param, 90)
		fair_maximum = np.max(fairness_param)
		fair_mean = np.mean(fairness_param)


		# calculate statistics for dist_mean_arr
		dist_mean_minimum = np.min(dist_mean_arr)
		dist_mean_0_1_quantile = np.percentile(dist_mean_arr, 10)
		dist_mean_median = np.median(dist_mean_arr)
		dist_mean_0_9_quantile = np.percentile(dist_mean_arr, 90)
		dist_mean_maximum = np.max(dist_mean_arr)
		dist_mean_mean = np.mean(dist_mean_arr)

		# calculate statistics for time_mean_arr
		time_mean_minimum = np.min(time_mean_arr)
		time_mean_0_1_quantile = np.percentile(time_mean_arr, 10)
		time_mean_median = np.median(time_mean_arr)
		time_mean_0_9_quantile = np.percentile(time_mean_arr, 90)
		time_mean_maximum = np.max(time_mean_arr)
		time_mean_mean = np.mean(time_mean_arr)

		# calculate statistics for success rates
		success_rates_minimum = np.min(success_rates_arr)
		success_rates_0_1_quantile = np.percentile(success_rates_arr, 10)
		success_rates_median = np.median(success_rates_arr)
		success_rates_0_9_quantile = np.percentile(success_rates_arr, 90)
		success_rates_maximum = np.max(success_rates_arr)
		success_rates_mean = np.mean(success_rates_arr)

		total_dists_traveled_median = np.median(total_dists_traveled)
		total_dists_traveled_mean = np.mean(total_dists_traveled)
		total_dists_traveled_0_1_quantile = np.percentile(total_dists_traveled, 10)
		total_dists_traveled_0_9_quantile = np.percentile(total_dists_traveled, 90)
		total_dists_traveled_min = np.min(total_dists_traveled)
		total_dists_traveled_max = np.max(total_dists_traveled)

		total_time_taken_median = np.median(total_time_taken)
		total_time_taken_mean = np.mean(total_time_taken)
		total_time_taken_0_1_quantile = np.percentile(total_time_taken, 10)
		total_time_taken_0_9_quantile = np.percentile(total_time_taken, 90)
		total_time_taken_min = np.min(total_time_taken)
		total_time_taken_max = np.max(total_time_taken)		


		np.set_printoptions(linewidth=400)
		print("Rewards", np.mean(rewards_arr))
		print("Frac of episode", np.mean(frac_episode_arr))

		# report the success rates statistics
		print("Success rates mean", success_rates_mean)
		print("Success rates median",success_rates_median)

		print("Num collisions", np.mean(num_collisions_arr))
		print("Fairness Median", fair_median)
		print("Fairness Mean", fair_mean)
		# Print the values
		print("Fair Median:", fair_median)
		print("Dists traveled", dists_trav_list)
		print("Time taken", time_taken_list)

		# print dist mean and time mean
		print("Dist Mean Median:", dist_mean_median)
		print("Time Mean Median:", time_mean_median)

		print("Total Dists Traveled Median:", total_dists_traveled_median)
		print("Total Time Taken Median:", total_time_taken_median)
		rewards_mean = np.mean(rewards_arr)

		csv_data = [
			num_obstacles, 
			num_agents,
			all_args.world_size,
			episode_length,
			all_args.render_episodes,
			fair_mean, 
			fair_minimum,
			fair_0_1_quantile, 
			fair_median, 
			fair_0_9_quantile,
			fair_maximum,
			np.mean(frac_episode_arr),
			success_rates_minimum,
			success_rates_0_1_quantile,
			success_rates_median ,
			success_rates_0_9_quantile ,
			success_rates_maximum,
			success_rates_mean,
			np.mean(num_collisions_arr),
			rewards_mean,
			rewards_mean / num_agents,
			rewards_mean / (num_agents * episode_length),
			dists_trav_list,
			time_taken_list,
			dist_mean_mean,
			dist_mean_minimum,
			dist_mean_0_1_quantile,
			dist_mean_median,
			dist_mean_0_9_quantile,
			dist_mean_maximum,
			time_mean_mean,
			time_mean_minimum,
			time_mean_0_1_quantile,
			time_mean_median,
			time_mean_0_9_quantile,
			time_mean_maximum,
			total_dists_traveled_median,
			total_dists_traveled_mean,
			total_dists_traveled_0_1_quantile,
			total_dists_traveled_0_9_quantile,
			total_dists_traveled_min,
			total_dists_traveled_max,
			total_time_taken_median,  # Add median value for total_time_taken
			total_time_taken_mean,
			total_time_taken_0_1_quantile,
			total_time_taken_0_9_quantile,
			total_time_taken_min,
			total_time_taken_max			
		]

		# csv_2 = [fairness_param]
		# open the file in the write mode
		with open('/Users/jasmine/Jasmine/MIT/MARL/Codes/Team-Fair-MARL/'+str(self.all_args.model_name)+'_firstgoaldone_nogoal_results_collect_new.csv', 'a', newline="") as f:
			# create the csv writer
			writer = csv.writer(f)
			# write a row to the csv file
			writer.writerow(csv_data)

		if not get_metrics:
			if all_args.save_gifs:
				# print("Hello")
				# print("gif dir", self.gif_dir)
				imageio.mimsave(str(gif_dir) + '/'+str(all_args.model_name)+'_check'+str(all_args.num_agents)+'.gif', 
								all_frames, duration=all_args.ifi, loop=0)
				





####### Helper functions taken from base runner ###########




def get_collisions( env_infos:Dict):
    """
        Get the collisions from the env_infos
    """
    collisions = 0
    for k, v in env_infos.items():
        if 'agent_collision' in k:
                collisions += v[0]/2.0
        if 'obstacle_collisions' in k:
            if len(v)>0:
                collisions += v[0]
    return collisions



def get_fraction_episodes(env_infos:Dict):
    """
        Get the fraction of episode required to get to the goals
        from env_infos
    """
    fracs = []
    success = []
    time_taken = []
    for k, v in env_infos.items():
        if 'time_to_goal' in k and 'min_time_to_goal' not in k:
            fracs.append(v[0] / (all_args.episode_length * dt))
            time_taken.append(v[0])
        ## add success based on dist_to_goal
        if 'dist_to_goal' in k:
            if v[0] < all_args.min_dist_thresh:
                success.append(1)
            else:
                success.append(0)
    assert len(success) == all_args.num_agents

    return fracs, success, time_taken



def process_infos(infos):
    """Process infos returned by environment."""
    env_infos = {}
    for agent_id in range(num_agents):
        idv_rews = []
        dist_goals, time_to_goals, min_times_to_goal,distance_mean, distance_variance, mean_by_cov = [], [], [], [], [], []
        idv_collisions, obst_collisions = [], []
        dists_traveled_list = []
        time_taken_list = []
        formation_distances = []
        time_mean_list, time_variance_list, time_mean_by_stddev_list = [], [], []
        for info in infos:
            if 'individual_reward' in info[agent_id].keys():
                idv_rews.append(info[agent_id]['individual_reward'])
            if 'Dist_to_goal' in info[agent_id].keys():
                dist_goals.append(info[agent_id]['Dist_to_goal'])
            if 'Time_req_to_goal' in info[agent_id].keys():
                times = info[agent_id]['Time_req_to_goal']
                if times == -1:
                    times = all_args.episode_length * dt    # NOTE: Hardcoding `dt`
                time_to_goals.append(times)
            if 'Num_agent_collisions' in info[agent_id].keys():
                idv_collisions.append(info[agent_id]['Num_agent_collisions'])
            if 'Num_obst_collisions' in info[agent_id].keys():
                obst_collisions.append(info[agent_id]['Num_obst_collisions'])
            if 'Min_time_to_goal' in info[agent_id].keys():
                min_times_to_goal.append(info[agent_id]['Min_time_to_goal'])
            if 'Distance_mean' in info[agent_id].keys():
                distance_mean.append(info[agent_id]['Distance_mean'])
            if 'Distance_variance' in info[agent_id].keys():
                distance_variance.append(info[agent_id]['Distance_variance'])
            if 'Mean_by_variance' in info[agent_id].keys():
                mean_by_cov.append(info[agent_id]['Mean_by_variance'])
            if 'Dists_traveled' in info[agent_id].keys():
                dists_traveled_list.append(info[agent_id]['Dists_traveled'])
            if 'Time_taken' in info[agent_id].keys():
                time_taken_list.append(info[agent_id]['Time_taken'])
            if 'Time_mean' in info[agent_id].keys():
                time_mean_list.append(info[agent_id]['Time_mean'])
            if 'Time_stddev' in info[agent_id].keys():
                time_variance_list.append(info[agent_id]['Time_stddev'])
            if 'Time_mean_by_stddev' in info[agent_id].keys():
                # print("mean by variance is", info[agent_id]['Mean_by_variance'])
                time_mean_by_stddev_list.append(info[agent_id]['Time_mean_by_stddev'])
        
        agent_rew = f'agent{agent_id}/individual_rewards'
        times     = f'agent{agent_id}/time_to_goal'
        dists     = f'agent{agent_id}/dist_to_goal'
        agent_col = f'agent{agent_id}/num_agent_collisions'
        obst_col  = f'agent{agent_id}/num_obstacle_collisions'
        min_times = f'agent{agent_id}/min_time_to_goal'
        dist_mean = f'agent{agent_id}/distance_mean'
        dist_variance  = f'agent{agent_id}/distance_variance'
        mean_variance  = f'agent{agent_id}/mean_variance'
        dists_traveled  = f'agent{agent_id}/dists_traveled'
        time_taken  = f'agent{agent_id}/time_taken'
        time_mean  = f'agent{agent_id}/time_mean'
        time_variance  = f'agent{agent_id}/time_variance'
        time_mn_by_stddev  = f'agent{agent_id}/time_mn_by_stddev'

        env_infos[agent_rew] = idv_rews
        env_infos[times]     = time_to_goals
        env_infos[min_times] = min_times_to_goal
        env_infos[dists]     = dist_goals
        env_infos[agent_col] = idv_collisions
        env_infos[obst_col]  = obst_collisions
        env_infos[dist_mean] = distance_mean
        env_infos[dist_variance]  = distance_variance
        env_infos[mean_variance]  = mean_by_cov
        env_infos[dists_traveled] = dists_traveled_list
        env_infos[time_taken] = time_taken_list
        env_infos[time_mean] = time_mean_list
        env_infos[time_variance]  = time_variance_list
        env_infos[time_mn_by_stddev]  = time_mean_by_stddev_list
    return env_infos


def get_fairness_metric(env_infos:Dict):
    """
        Get the collisions from the env_infos
    """
    mean_variance = []
    for k, v in env_infos.items():
        if 'mean_variance' in k:
            mean_variance.append(v[0])
    return mean_variance


def get_dist_mean( env_infos:Dict):
    """
        Get the collisions from the env_infos
    """
    distance_mean = []
    for k, v in env_infos.items():
        if 'distance_mean' in k:
            distance_mean.append(v[0])
    return distance_mean

def get_time_mean( env_infos:Dict):
    """
        Get the time_mean from the env_infos
    """
    time_mean = []
    for k, v in env_infos.items():
        if 'time_mean' in k:
            time_mean.append(v[0])
    return time_mean


def get_dists_traveled(env_infos:Dict):
    """
        Get the collisions from the env_infos
    """
    dists_traveled = []
    for k, v in env_infos.items():
        if 'dists_traveled' in k:
            dists_traveled.append(v[0])
    return dists_traveled