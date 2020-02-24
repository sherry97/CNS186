import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from lib import model

pairs = [(0,0),(0,1),(1,0),(1,1)]
n_rows = 2
n_cols = 2
assert n_rows * n_cols >= len(pairs)
fig, axs = plt.subplots(n_rows+1, n_cols, figsize=(16,10))

for i,t in enumerate(pairs):
    flow, integration = t
    transform = model.get_optical_flow_transform(flow, integration)
    dataloader = model.load_data(transform, train=False)
    for _, data in enumerate(dataloader):
        img,_,labels = data
        pic = transforms.ToPILImage()(img[1])
        axs[i//n_rows, i%n_rows].imshow(np.asarray(pic))
        axs[i//n_rows, i%n_rows].set_title(f'OF {t}')
        break

for i,t in enumerate(pairs):
    flow, integration = t
    if flow > 0:
        continue
    transform = model.get_generic_transform(integration)
    dataloader = model.load_data(transform, train=False)
    for _,data in enumerate(dataloader):
        img,_,labels = data
        pic = transforms.ToPILImage()(img[1])
        axs[(i+len(pairs))//n_rows, i%n_rows].imshow(np.asarray(pic))
        axs[(i+len(pairs))//n_rows, i%n_rows].set_title(f'Generic {t}')
        break

plt.savefig('data/output/transformation_demo.png')
plt.clf()
