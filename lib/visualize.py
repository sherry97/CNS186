import matplotlib.pyplot as plt

__all__ = ['plot']

def plot(data, exp_config, train=True):
    fig_path = f"data/output/{exp_config['id']}_"

    if train:
        fig_path += 'train.png'
        plt.figure(figsize=(16,10))
        plt.plot(data)
        plt.xlabel('iteration')
        plt.ylabel('cross entropy loss')
        plt.yscale('log')
    else:
        fig_path += 'val.png'
        fig, axs = plt.subplots(1,2,figsize=(16,10))
        axs[0].scatter(0,data[0])
        axs[0].set_ylabel('cross entropy loss')
        axs[0].set_yscale('log')
        axs[1].scatter(0,data[1])
        axs[1].set_ylabel('accuracy')

    plt.savefig(fig_path)
