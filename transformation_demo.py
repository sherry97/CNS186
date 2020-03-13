import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from lib import model
import cv2

pairs = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
n_rows = 2
n_cols = 3
assert n_rows * n_cols >= len(pairs)
fig, axs = plt.subplots(n_rows+1, n_cols, sharex=True, sharey=True, figsize=(14,10))

flow_titles = ['optical flow\n(magnitude)','optical flow\n(direction)', 'luminance']
integration_titles = ['uniform average','Gaussian average','end-heavy average']
fs = 18
for i in range(3):
    axs[0,i].set_title(f'{integration_titles[i]}',fontsize=fs)
    axs[i,0].set_ylabel(flow_titles[i],fontsize=fs)

for i,t in enumerate(pairs):
    flow, integration = t
    transform = model.get_optical_flow_transform(flow, integration)
    dataloader = model.load_data(transform, train=False)
    for _, data in enumerate(dataloader):
        img,_,labels = data
        pic = transforms.ToPILImage()(img[1])
        axs[i//n_cols, i%n_cols].imshow(np.asarray(pic))
        # axs[i//n_cols, i%n_cols].set_title(f'Optical Flow ({flow_titles[flow]} integrated with {integration_titles[integration]})')
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
        axs[(i+len(pairs))//n_cols, i%n_cols].imshow(np.asarray(pic))
        # axs[(i+len(pairs))//n_cols, i%n_cols].set_title(f'RGB (integrated with {integration_titles[integration]})')
        break

plt.tight_layout()
plt.savefig('data/output/transformation_demo.png')
plt.clf()

frame_size = 240

def clip_transform(x):
    # get luminance
    x = x.to(dtype=torch.float).mean(dim=-1)/255
    # pad if necessary
    if x.shape[2] <= frame_size:
        size = ((frame_size - x.shape[2]) // 2)+1
        pad = (size,size)
        x = nn.functional.pad(x, pad)
    # crop to square
    i = torch.randint(x.shape[2]-frame_size,size=(1,1)).item()
    x = x[:,:,i:i+frame_size]
    return x

t = transforms.Lambda(lambda x : clip_transform(x))
dataloader = model.load_data(t, train=False)
for _,data in enumerate(dataloader):
    img,_,labels = data
    clip_size = img.shape[1]
    fig, axs = plt.subplots(1, clip_size, figsize=(16,5))
    for ind, frame in enumerate(img[1]):
        pic = transforms.ToPILImage()(frame)
        axs[ind].imshow(pic)
        axs[ind].axis('off')
    break
plt.tight_layout()
plt.savefig('data/output/lum_clip_demo.png')
plt.clf()

def optical_flow(x, flow_index):
    # get luminance
    x = x.to(dtype=torch.float).mean(dim=-1)
    # pad if necessary
    if x.shape[2] <= frame_size:
        size = ((frame_size - x.shape[2]) // 2)+1
        pad = (size,size)
        x = nn.functional.pad(x, pad)
    # crop to square
    i = torch.randint(x.shape[2]-frame_size,size=(1,1)).item()
    x = x[:,:,i:i+frame_size]
    # use OpenCV to calculate Farneback dense optical flow
    flow = x.numpy().astype(np.uint8)
    for i in range(x.shape[0]-1):
        prvs = flow[i]
        next = flow[i+1]
        out = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # flow_index=0 => flow value; flow_index=1 => flow direction
        flow[i] = out[:,:,flow_index]
    x = torch.tensor(flow)
    return x

t = transforms.Lambda(lambda x : optical_flow(x,0))
dataloader = model.load_data(t, train=False)
for _,data in enumerate(dataloader):
    img,_,labels = data
    clip_size = img.shape[1]
    fig, axs = plt.subplots(1, clip_size, figsize=(16,5))
    for ind, frame in enumerate(img[1]):
        pic = transforms.ToPILImage()(frame)
        axs[ind].imshow(pic)
        axs[ind].axis('off')
    break
plt.tight_layout()
plt.savefig('data/output/flow0_clip_demo.png')
plt.clf()

t = transforms.Lambda(lambda x : optical_flow(x,1))
dataloader = model.load_data(t, train=False)
for _,data in enumerate(dataloader):
    img,_,labels = data
    clip_size = img.shape[1]
    fig, axs = plt.subplots(1, clip_size, figsize=(16,5))
    for ind, frame in enumerate(img[1]):
        pic = transforms.ToPILImage()(frame)
        axs[ind].imshow(pic)
        axs[ind].axis('off')
    break
plt.tight_layout()
plt.savefig('data/output/flow1_clip_demo.png')
plt.clf()
