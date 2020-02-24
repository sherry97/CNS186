import numpy as np
import matplotlib.pyplot as plt
import json

exps = [i for i in range(6,9)]+[i for i in range(10,18)]
n_plots = len(exps)
datadir = 'data/output'
expdir = 'experiments'

# dataset stats
n_walk = 549
n_turn = 241
n_run = 233

fig, axs = plt.subplots(1,2,figsize=(16,10))
baseline_color='blue'
test_color='red'
# data
for i in exps:
    data = np.load(f'{datadir}/{i}_val_loss.npy')
    exp_config_filename = f'{expdir}/{i}.json'
    with open(exp_config_filename,'r') as f:
        exp_config = json.load(f)
    color = baseline_color
    if exp_config['transform_mode']:
        color = test_color
    axs[0].scatter(i,data[0],label=f'exp {i}',color=color)
    axs[1].scatter(i,data[1],label=f'exp {i}',color=color)
    axs[0].annotate(f"exp {i}\nlr={exp_config['lr']}\
        \nflow={exp_config['flow_index']};int={exp_config['integration_index']} \
        \nval={data[0]:0.4f}",xy=(i,data[0]))
    axs[1].annotate(f"exp {i}\nlr={exp_config['lr']}\
        \nflow={exp_config['flow_index']};int={exp_config['integration_index']} \
        \nval={data[1]:0.4f}",xy=(i,data[1]))

# subplot configs
axs[1].plot([min(exps),max(exps)],[1/3,1/3],'--',label='chance',color='gray')
axs[0].set_title('Validation Cross Entropy Loss')
axs[0].set_yscale('log')
axs[0].set_xlabel('experiment id')
axs[1].set_title('Top-1 Accuracy')
axs[1].set_xlabel('experiment id')

# plot configs
handles, labels = axs[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=n_plots+2)
plt.savefig(f'{datadir}/combo_val_plot.png')
plt.clf()
