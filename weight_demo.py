import matplotlib.pyplot as plt
import torch
from scipy.stats import norm, beta
import seaborn as sns

sns.set()

ii = [0,1,2]
x = torch.zeros([10,1,240,240])
fig,axs = plt.subplots(3,1,figsize=(4,8))
integration_titles = ['uniform average','Gaussian average','end-heavy average']

for i,integration_index in enumerate(ii):
  if integration_index == 0:
      weights = torch.ones(x.shape[0])
  elif integration_index == 1:
      max_x = x.shape[0]
      weights = torch.tensor([norm.pdf(dx/max_x-0.5) for dx in range(max_x)])
  else:
      max_x = x.shape[0]
      a = 2
      b = 1.5
      weights = torch.tensor([beta.pdf(dx/max_x-0.5,a,b) for dx in range(max_x)])
  weights = weights / torch.sum(weights)
  axs[i].plot(weights)
  axs[i].set_title(integration_titles[i])
  axs[i].set_xlabel('frame index')

axs[0].set_ylabel('weight')
plt.tight_layout()
plt.savefig('data/output/weight_demo.png')
plt.clf()
