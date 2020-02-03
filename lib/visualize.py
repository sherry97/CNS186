import matplotlib.pyplot as plt

__all__ = ['plot']

def plot(data, exp_config, train=True):
    fig_path = f"data/output/{exp_config['id']}_"
    if train:
        fig_path += 'train.png'
    else:
        fig_path += 'val.png'

    plt.figure(figsize=(16,10))
    plt.plot(data)
    plt.xlabel('iteration')
    plt.ylabel('cross entropy loss')
    plt.yscale('log')
    plt.savefig(fig_path)
