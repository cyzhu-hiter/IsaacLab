import wandb
import json
import os

def initialize_wandb(config, config_path="api_key.json"):
    """
    Initialize wandb with the API key from a JSON file.
    
    Args:
        config_path (str): The path to the JSON file containing the API key.
        
    Raises:
        ValueError: If the API key is not found or is invalid.
    """
    # Get the absolute path of the current script file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Join the script directory with the config_path to form an absolute path
    config_path = os.path.join(script_dir, config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")

    with open(config_path, 'r') as file:
        config = json.load(file)
    
    api_key = config.get("API_KEY", 0)
    if not api_key or not isinstance(api_key, str) or api_key == "add your wandb api key":
        raise ValueError("API key is not found or invalid in the configuration file.")
    
    wandb.login(key=api_key)
    wandb_project = "isaaclab-prl"
    wandb_group = "dorl-prl"
    
    run_name = f"test"

    wandb.init(
        project=wandb_project,
        group=wandb_group,
        # NOTE: Uncommenting the line below may cause issues if multiple users are accessing the same wandb project.
        # entity=cfg.wandb_entity, 
        config=config,
        sync_tensorboard=True,
        name=run_name, 
        resume="allow",
    )
    
    print("Wandb initialized successfully.")
