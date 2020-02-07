import numpy as np
import torch
from .model import get_model, train, validate
from .visualize import plot

__all__ = ['run_exp']

def run_exp(exp_config, train_model=False):
    train_file_path = f"data/output/{exp_config['id']}_train_loss.npy"
    val_file_path = f"data/output/{exp_config['id']}_val_loss.npy"
    m = get_model()
    if train_model:
        m, training_loss = train(m, exp_config)
        np.save(train_file_path, training_loss)
    else:
        pt = torch.load(exp_config['model_path'])
        m = m.load_state_dict(pt['model_state_dict'])
        print(f"Loaded model from epoch {pt['epoch']}.")
    val_loss = validate(m, exp_config)
    np.save(val_file_path, val_loss)
    plot(training_loss, exp_config, train=True)
    plot(val_loss, exp_config, train=False)
