from datetime import datetime
import wandb
import json
import os
class WandbManager:
    def __init__(self, config_path="api_key.json"):
        """
        Initialize the WandbManager with the API key from a JSON file.
        
        Args:
            config_path (str): The path to the JSON file containing the API key.
            
        Raises:
            FileNotFoundError: If the configuration file is not found.
            ValueError: If the API key is not found or is invalid.
        """
        # Get the absolute path of the current script file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Join the script directory with the config_path to form an absolute path
        self.config_path = os.path.join(script_dir, config_path)
        
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file {self.config_path} not found.")
        
        with open(self.config_path, 'r') as file:
            self.config = json.load(file)
        
        self.api_key = self.config.get("API_KEY", 0)
        if not self.api_key or not isinstance(self.api_key, str) or self.api_key == "add your wandb api key":
            raise ValueError("API key is not found or invalid in the configuration file.")
        
    def initialize(self, args_cli):
        """
        Initialize the wandb run with the provided configuration.
        
        Args:
            args_cli: Parsed command-line arguments.
        """
        wandb.login(key=self.api_key)
        wandb_project = "isaaclab-prl"
        wandb_group = "dorl-prl"
        
        # Extract the middle part of the task name by splitting and joining appropriately
        task_name = '-'.join(args_cli.task.split('-')[1:-1])
        
        prefix = 'Prexp' if args_cli.pre_ex else 'Opt' # pre-experiment or optimization
        
        # Construct the run name using the extracted task name
        run_name = f"{prefix}_{task_name}_{args_cli.tuning_param}_{args_cli.minibatch_size}mb_" \
                   f"{args_cli.mini_epochs}me_{args_cli.num_envs}ne_" \
                   f"{args_cli.num_nodes}nn_{args_cli.nproc_per_node}np_" \
                   f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            project=wandb_project,
            group=wandb_group,
            # NOTE: Uncommenting the line below may cause issues if multiple users are accessing the same wandb project.
            # entity=cfg.wandb_entity, 
            config=self.config,
            sync_tensorboard=True,
            name=run_name, 
            resume="allow",
        )
        
        print("Wandb initialized successfully.")
    
    def finish(self):
        """
        Finish the wandb run and ensure proper cleanup.
        """
        wandb.finish()
        print("Wandb session finished.")
    
def adjust_gpu_parameters(env_cfg, curr_num_envs):
    # Default number of environments
    default_num_envs = env_cfg.scene.num_envs
    
    # Calculate the scaling factor based on the current number of environments
    scale_factor = curr_num_envs / default_num_envs
    
    # Update GPU-related parameters based on the scaling factor
    env_cfg.sim.physx.gpu_max_rigid_contact_count = int(8388608 * scale_factor)
    env_cfg.sim.physx.gpu_max_rigid_patch_count = int(163840 * scale_factor)
    env_cfg.sim.physx.gpu_found_lost_pairs_capacity = int(2097152 * scale_factor)
    env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = int(33554432 * scale_factor)
    env_cfg.sim.physx.gpu_total_aggregate_pairs_capacity = int(2097152 * scale_factor)
    env_cfg.sim.physx.gpu_collision_stack_size = int(67108864 * scale_factor)
    env_cfg.sim.physx.gpu_heap_capacity = int(67108864 * scale_factor)
    env_cfg.sim.physx.gpu_temp_buffer_capacity = int(16777216 * scale_factor)
    env_cfg.sim.physx.gpu_max_soft_body_contacts = int(1048576 * scale_factor)
    env_cfg.sim.physx.gpu_max_particle_contacts = int(1048576 * scale_factor)
    
    print("GPU parameters have been adjusted based on the current number of environments.")
    return env_cfg
