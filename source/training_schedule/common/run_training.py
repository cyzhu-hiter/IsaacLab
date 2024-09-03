import os
import time
import traceback

def log_run_status(log_file, task, num_envs, nproc_per_node, mini_epoch, minibatch_size, seed, tuning_param, tuning_value, status, wall_time=None):
    """
    Logs the status of a training run to a log file.

    Parameters:
    - log_file (str): Path to the log file.
    - task (str): The RL task name.
    - num_envs (int): Number of environments to simulate.
    - nproc_per_node (int): Number of processes per node.
    - mini_epoch (int): Number of mini epochs.
    - minibatch_size (int): Minibatch size.
    - seed (int): Random seed.
    - tuning_param (str): The parameter being tuned.
    - tuning_value (int/str): The value of the tuning parameter.
    - status (str): Status of the run ("SUCCESS" or "FAILED").
    - wall_time (float, optional): The wall time of the run in seconds.
    """
    with open(log_file, "a") as log:
        log.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Task: {task}\n")
        log.write(f"Tuning {tuning_param}={tuning_value}, Seed={seed}\n")
        log.write(f"Parameters: num_envs={num_envs}, nproc_per_node={nproc_per_node}, mini_epoch={mini_epoch}, minibatch_size={minibatch_size}\n")
        log.write(f"Status: {status}\n")
        if wall_time is not None:
            log.write(f"Wall Time: {wall_time:.2f} seconds\n")
        log.write("-" * 50 + "\n")

def run_training(task, num_envs, nproc_per_node, mini_epoch, minibatch_size, seed, tuning_param, tuning_value, max_time=1e6, max_iterations=1e5):
    """
    Execute the training command with the specified parameters and log the results.

    Parameters:
    - task (str): The RL task name.
    - num_envs (int): Number of environments to simulate.
    - nproc_per_node (int): Number of processes per node (GPUs per node).
    - mini_epoch (int): Number of mini epochs for training.
    - minibatch_size (int): Size of the minibatch.
    - seed (int): Random seed for reproducibility.
    - tuning_param (str): The parameter being tuned (e.g., 'mb' for minibatch size).
    - tuning_value (int/str): The value of the parameter being tuned.
    - max_time (float): Maximum allowed time for the training in seconds.
    - max_iterations (int): Maximum number of iterations for the training.
    """
    log_file = "/workspace/isaaclab/source/training_schedule/training_log.txt"  # Absolute path to the log file

    command = (
        f"/workspace/isaaclab/_isaac_sim/kit/python/bin/python3 -m torch.distributed.run --nnodes=1 --nproc_per_node={nproc_per_node} "
        f"source/standalone/workflows/rl_games/train.py --task={task} "
        f"--headless --distributed --wandb --pre-ex --num_envs {num_envs} "
        f"--mini_epoch {mini_epoch} --minibatch_size {minibatch_size} "
        f"--seed {seed} --tuning_param {tuning_param} --max_time {max_time} --max_iterations {int(max_iterations)}"
    )
    
    print(f"Running task: {task}, Tuning {tuning_param}={tuning_value}, Seed={seed}")
    
    start_time = time.time()
    try:
        result = os.system(command)
        end_time = time.time()
        wall_time = end_time - start_time
        
        if result == 0:
            status = "SUCCESS"
        else:
            status = "FAILED"
            
        log_run_status(log_file, task, num_envs, nproc_per_node, mini_epoch, minibatch_size, seed, tuning_param, tuning_value, status, wall_time)
    
    except Exception as e:
        end_time = time.time()
        wall_time = end_time - start_time
        status = "FAILED"
        log_run_status(log_file, task, num_envs, nproc_per_node, mini_epoch, minibatch_size, seed, tuning_param, tuning_value, status, wall_time)
        print(f"An error occurred: {e}")
        traceback.print_exc()
