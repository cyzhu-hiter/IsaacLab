from common.generate_grid import grid_search

# Define your parameter ranges
num_envs_values = [2048, 4096, 8192, 16384]
nproc_per_node_values = [1, 2, 4, 8]
mini_epoch_values = [2, 4, 6, 8, 10, 12]
minibatch_size_values = [8192, 16384, 32768, 65536]
seed_values = [101, 102, 103]

# Default values specific to Isaac-Ant-v0 task
default_num_envs = 4096
default_nproc_per_node = 2
default_mini_epoch = 8
default_minibatch_size = 32768
default_max_time = 360  # Set the maximum time for each run

# Call the grid search function for Isaac-Ant-v0 task
grid_search(
    task="Isaac-Ant-v0",
    num_envs_values=num_envs_values,
    nproc_per_node_values=nproc_per_node_values,
    mini_epoch_values=mini_epoch_values,
    minibatch_size_values=minibatch_size_values,
    seed_values=seed_values,
    default_num_envs=default_num_envs,
    default_nproc_per_node=default_nproc_per_node,
    default_mini_epoch=default_mini_epoch,
    default_minibatch_size=default_minibatch_size,
    max_time=default_max_time
)
