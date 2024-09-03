from common.run_training import run_training

def grid_search(task, num_envs_values, nproc_per_node_values, mini_epoch_values, minibatch_size_values, seed_values,
                default_num_envs, default_nproc_per_node, default_mini_epoch, default_minibatch_size, 
                max_time=1e6, max_iterations=1e5):
    
    # Adjust num_envs and loop over seeds
    for num_envs in num_envs_values:
        tuning_param = 'ne'
        for seed in seed_values:
            run_training(
                task=task,
                num_envs=num_envs,
                nproc_per_node=default_nproc_per_node,
                mini_epoch=default_mini_epoch,
                minibatch_size=default_minibatch_size,
                seed=seed,
                tuning_param=tuning_param,
                tuning_value=num_envs,
                max_time=max_time,
                max_iterations=max_iterations
            )
    
    # Adjust nproc_per_node and loop over seeds
    for nproc_per_node in nproc_per_node_values:
        tuning_param = 'np'
        for seed in seed_values:
            run_training(
                task=task,
                num_envs=default_num_envs,
                nproc_per_node=nproc_per_node,
                mini_epoch=default_mini_epoch,
                minibatch_size=default_minibatch_size,
                seed=seed,
                tuning_param=tuning_param,
                tuning_value=nproc_per_node,
                max_time=max_time,
                max_iterations=max_iterations
            )
    
    # Adjust mini_epoch and loop over seeds
    for mini_epoch in mini_epoch_values:
        tuning_param = 'me'
        for seed in seed_values:
            run_training(
                task=task,
                num_envs=default_num_envs,
                nproc_per_node=default_nproc_per_node,
                mini_epoch=mini_epoch,
                minibatch_size=default_minibatch_size,
                seed=seed,
                tuning_param=tuning_param,
                tuning_value=mini_epoch,
                max_time=max_time,
                max_iterations=max_iterations
            )
    
    # Adjust minibatch_size and loop over seeds
    for minibatch_size in minibatch_size_values:
        tuning_param = 'mb'
        for seed in seed_values:
            run_training(
                task=task,
                num_envs=default_num_envs,
                nproc_per_node=default_nproc_per_node,
                mini_epoch=default_mini_epoch,
                minibatch_size=minibatch_size,
                seed=seed,
                tuning_param=tuning_param,
                tuning_value=minibatch_size,
                max_time=max_time,
                max_iterations=max_iterations
            )
