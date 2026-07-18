import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config: dict, save_path: str):
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
