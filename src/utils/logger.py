import logging
import sys

def get_logger(name: str, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

# Support file logging

def log_metrics(metrics, step, prefix=''):
    """Log metrics with step number and optional prefix."""
    for key, value in metrics.items():
        metric_name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, (int, float)):
            print(f"Step {step} | {metric_name}: {value:.4f}")
        else:
            print(f"Step {step} | {metric_name}: {value}")

def create_experiment_logger(experiment_name, log_dir='logs'):
    """Create logger for experiment tracking."""
    import os
    from datetime import datetime
    
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{experiment_name}_{timestamp}.log')
    
    logger = get_logger(experiment_name)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(file_handler)
    
    return logger, log_file

class MetricsTracker:
    """Track and aggregate metrics during training."""
    
    def __init__(self):
        self.history = {}
    
    def update(self, metrics, epoch):
        """Add metrics for current epoch."""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append({'epoch': epoch, 'value': value})
    
    def get_best(self, metric_name, mode='max'):
        """Get best value for a metric."""
        if metric_name not in self.history:
            return None
        values = [entry['value'] for entry in self.history[metric_name]]
        if mode == 'max':
            best_idx = values.index(max(values))
        else:
            best_idx = values.index(min(values))
        return self.history[metric_name][best_idx]
    
    def save(self, path):
        """Save metrics history to JSON."""
        import json
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
