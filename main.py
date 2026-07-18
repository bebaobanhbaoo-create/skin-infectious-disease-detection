"""Main entry point for training and evaluation."""
import argparse
from src.utils import load_config, set_seed, get_logger

def main():
    parser = argparse.ArgumentParser(description='Skin Disease Classification')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'gradcam'], default='train')
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get('seed', 42))
    logger = get_logger(__name__)

    logger.info(f"Running in {args.mode} mode")
    logger.info(f"Config: {args.config}")

    if args.mode == 'train':
        from scripts.train import train
        train(config)
    elif args.mode == 'eval':
        from scripts.evaluate import evaluate
        evaluate(config)
    elif args.mode == 'gradcam':
        from scripts.generate_gradcam import generate
        generate(config)

if __name__ == '__main__':
    main()
