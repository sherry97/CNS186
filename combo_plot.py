import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

sns.set()

exps = [2,5,6,10,14,17,18,22,26]
# exps = [i for i in range(27)]
n_plots = len(exps)
datadir = 'data/output'
expdir = 'experiments'

# dataset stats
n_walk = 549
n_turn = 241
n_run = 233
# n_samples = n_walk + n_turn + n_run
n_samples = 1284
assert n_samples == 1284, f'Expected 1284 samples, got {n_samples}'

# fig, axs = plt.subplots(1,2,figsize=(16,10))
fig = plt.figure()

colors = ['green','blue','red']
flow_titles = ['optical flow\n(magnitude)','optical flow\n(direction)', 'luminance']
integration_titles = ['uniform average','Gaussian average','end-heavy average']

handles = []
labels = []
# data
n_pts = len(exps)
Y = []
c = []
labels = []
ann_txt = []
ann_xy = []
for n,i in enumerate(exps):
    data = np.load(f'{datadir}/{i}_val_loss.npy')
    exp_config_filename = f'{expdir}/{i}.json'
    with open(exp_config_filename,'r') as f:
        exp_config = json.load(f)
    ft_ind = 2 if not exp_config['transform_mode'] else exp_config['flow_index']
    it_ind = exp_config['integration_index']
    acc = data[1]
    err_range = 1.96 * np.sqrt((acc * (1-acc)) / n_samples)
    if n == 0 or n == 5:
        xy = (flow_titles[ft_ind],acc-0.006)
    else:
        xy = (flow_titles[ft_ind],acc+0.002)
    Y.append([flow_titles[ft_ind], acc, err_range])
    c.append(colors[ft_ind])
    labels.append(integration_titles[it_ind])
    ann_txt.append(f'{np.round(acc,3)} +- {np.round(err_range,3)}')
    ann_xy.append(xy)

# plot datafs=18
fig, axs = plt.subplots(1,3, sharey=True, figsize=(10,3))
for offset in range(3):
    x = [Y[p*3 + offset][0] for p in range(3)]
    y = [Y[p*3 + offset][1] for p in range(3)]
    err_range = [Y[p*3 + offset][2] for p in range(3)]
    m_color = c[3*offset]
    axs[offset].errorbar(x,y,yerr=err_range,mfc=m_color,ms=5,ecolor='gray')
    axs[offset].set_title(labels[offset])
    # for p in range(3):
    #     axs[offset].annotate(ann_txt[p*3 + offset], xy=ann_xy[p*3+offset])

# plot configs
axs[0].set_ylabel('Top-1 Accuracy')
plt.tight_layout()
plt.savefig(f'{datadir}/combo_val_plot.png')
plt.clf()


for n,i in enumerate(exps):
    data = np.load(f'{datadir}/{i}_train_loss.npy')
    exp_config_filename = f'{expdir}/{i}.json'
    with open(exp_config_filename,'r') as f:
        exp_config = json.load(f)
    ft_ind = 2 if not exp_config['transform_mode'] else exp_config['flow_index']
    it_ind = exp_config['integration_index']
    x = np.arange(data.shape[0])
    plt.plot(x,data,label=f'exp {i}, {flow_titles[ft_ind]}, {integration_titles[it_ind]}')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('cross entropy loss')
plt.yscale('log')
plt.tight_layout()
plt.savefig(f'{datadir}/combo_loss_plot.png')
plt.clf()
