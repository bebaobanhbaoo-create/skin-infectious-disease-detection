"""Run all experiments and collect results."""
import argparse
import yaml
import torch
from pathlib import Path
from src.utils import set_seed, get_logger

def run_experiment(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger = get_logger(config['experiment_name'])
    set_seed(config['seed'])

    logger.info(f"Running experiment: {config['experiment_name']}")
    logger.info(f"Description: {config['description']}")

    # Run training based on config
    # ...

    return {'config': config_path, 'status': 'completed'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run_experiment(args.config)
