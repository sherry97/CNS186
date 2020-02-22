import numpy as np
from os import path
import torch
from .model import get_model, train, validate
from .visualize import plot

__all__ = ['run_exp']

def run_exp(exp_config, train_model=False):
    train_file_path = f"data/output/{exp_config['id']}_train_loss.npy"
    val_file_path = f"data/output/{exp_config['id']}_val_loss.npy"
    starting_epoch = 0
    training_loss = np.zeros([1])
    m = get_model()
    # load model
    if path.exists(exp_config['model_path']):
        pt = torch.load(exp_config['model_path'])
        m.load_state_dict(pt['model_state_dict'])
        try:
            training_loss = np.load(train_file_path)
        except FileNotFoundError:
            pass
        starting_epoch = pt['epoch']+1
        print(f"Loaded model from epoch {pt['epoch']+1}.")
    # train, or else load training data
    if train_model and starting_epoch < exp_config['n_epochs']:
        m, training_loss = train(m, exp_config, starting_epoch, training_loss)
        np.save(train_file_path, training_loss)
    else:
        training_loss = np.load(train_file_path)
    # validate
    val_loss = validate(m, exp_config)
    np.save(val_file_path, val_loss)
    # val_loss = np.load(val_file_path)
    print('Completed validation. Plotting...')
    plot(training_loss, exp_config, train=True)
    plot(val_loss, exp_config, train=False)
    print(f"Finished experiment {exp_config['id']}.")
