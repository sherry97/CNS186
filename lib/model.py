import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from scipy.stats import norm

__all__ = ['get_model', 'train', 'validate']

#############################
#   Data
#############################

data_dir = r'data/input'
data_dir_ext = r'data/annotations'
frame_size = 240

def optical_flow(x, flow_index, integration_index):
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
    # integrate over time via weighted sum
    if integration_index == 0:
        weights = torch.ones(x.shape[0]) / x.shape[0]
    else:
        max_x = x.shape[0]
        weights = torch.tensor([norm.pdf(dx/max_x-0.5) for dx in range(max_x)])
    weights = weights / torch.sum(weights)
    weights = weights.repeat_interleave(frame_size*frame_size)
    weights = weights.reshape(-1,frame_size,frame_size)
    assert weights.shape == x.shape, \
        f'Invalid weights shape. Expected {x.shape}, got {weights.shape}'
    x = torch.sum(weights * x, dim=0, keepdim=True) / 255
    assert x.shape == (1,frame_size,frame_size), \
        f'Invalid input tensor size. Expected (1,{frame_size},{frame_size}), got {x.shape}'
    return x

def generic_transform(x, integration_index):
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
    # integrate over time via weighted sum
    if integration_index == 0:
        weights = torch.ones(x.shape[0]) / x.shape[0]
    else:
        max_x = x.shape[0]
        weights = torch.tensor([norm.pdf(dx/max_x-0.5) for dx in range(max_x)])
    weights = weights / torch.sum(weights)
    weights = weights.repeat_interleave(frame_size*frame_size)
    weights = weights.reshape(-1,frame_size,frame_size)
    assert weights.shape == x.shape, \
        f'Invalid weights shape. Expected {x.shape}, got {weights.shape}'
    x = torch.sum(weights * x, dim=0, keepdim=True)
    assert x.shape == (1,frame_size,frame_size), \
        f'Invalid input tensor size. Expected (1,{frame_size},{frame_size}), got {x.shape}'
    return x

def get_optical_flow_transform(flow_index, integration_index):
    return transforms.Lambda(lambda x : optical_flow(x,flow_index,integration_index))

def get_generic_transform(integration_index):
    return transforms.Lambda(lambda x : generic_transform(x,integration_index))

def load_data(transform, train=True):
    print('Start loading data...')
    dataset = torchvision.datasets.HMDB51(data_dir,
                                        data_dir_ext,
                                        frames_per_clip=10,
                                        step_between_clips=5,
                                        train=train,
                                        transform=transform)
    print(f'Loaded {len(dataset)} entries from {data_dir}. Convert to DataLoader...')
    data_loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=64,
                                        shuffle=train)
    print('Done.')
    return data_loader

#############################
#   Model
#############################

def get_model():
    # using All-CNN for simple model
    n_channel = 16
    model = nn.Sequential(
            nn.Conv2d(1,n_channel,3,padding=1),
            nn.BatchNorm2d(n_channel),
            nn.ReLU(),
            nn.Conv2d(n_channel,n_channel,3,padding=1),
            nn.BatchNorm2d(n_channel),
            nn.ReLU(),
            nn.Conv2d(n_channel,n_channel,3, stride=2),
            nn.ReLU(),
            nn.Conv2d(n_channel,2 * n_channel,3,padding=1),
            nn.BatchNorm2d(2 * n_channel),
            nn.ReLU(),
            nn.Conv2d(2 * n_channel,2 * n_channel,3,padding=1),
            nn.BatchNorm2d(2 * n_channel),
            nn.ReLU(),
            nn.Conv2d(2 * n_channel,2 * n_channel,3, stride=2),
            nn.ReLU(),
            nn.Conv2d(2 * n_channel,2 * n_channel,3),
            nn.BatchNorm2d(2 * n_channel),
            nn.ReLU(),
            nn.Conv2d(2 * n_channel,2 * n_channel,1),
            nn.BatchNorm2d(2 * n_channel),
            nn.ReLU(),
            nn.Conv2d(2 * n_channel,3,1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device)
    return model

def train(model, exp_config, starting_epoch, training_loss):
    # training setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Training on {device}.')
    if exp_config['transform_mode']:
        transform = get_optical_flow_transform(exp_config['flow_index'],
                                               exp_config['integration_index'])
    else:
        transform = get_generic_transform(exp_config['integration_index'])
    data_loader = load_data(transform, train=True)
    dataset_size = len(data_loader)

    # init auxiliary training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                        lr=exp_config['lr'],
                        momentum=0.9,
                        weight_decay=0.001)
    if starting_epoch > 0:
        pt = torch.load(exp_config['model_path'])
        optimizer.load_state_dict(pt['optimizer_state_dict'])
    n_epochs = exp_config['n_epochs']

    # store results
    performance = np.zeros([dataset_size*n_epochs])
    starting_iteration = dataset_size*starting_epoch
    performance[:starting_iteration] = training_loss[:starting_iteration]

    # iterate over dataset
    model.train()
    global_start = time.clock()
    for epoch in range(starting_epoch, n_epochs):
        start_time = time.clock()
        for i, data in enumerate(data_loader):
            # print(data)
            vid, _, label = data
            # continue
            vid, label = vid.to(device=device), label.to(device=device)
            optimizer.zero_grad()
            out = model(vid).squeeze(3).squeeze(2)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            performance[epoch*dataset_size + i] = loss.item()
        # save after each epoch
        torch.save({
            'epoch':epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, exp_config['model_path'])
        # print estimated time remaining
        elapsed = (time.clock() - start_time) / 60
        print(f'Epoch {epoch+1}/{n_epochs} complete. \
                Est {elapsed * (n_epochs-epoch):0.2f} minutes remaining. \
                Last loss: {performance[(epoch+1)*dataset_size-1]}')

    global_elapsed = (time.clock() - global_start) / 60
    print(f'Finished training in {global_elapsed:0.2f} minutes.')

    return model, performance

def validate(model, exp_config):
    # validation setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if exp_config['transform_mode']:
        transform = get_optical_flow_transform(exp_config['flow_index'],
                                               exp_config['integration_index'])
    else:
        transform = get_generic_transform(exp_config['integration_index'])
    data_loader = load_data(transform, train=False)
    criterion = nn.CrossEntropyLoss()

    # store results
    n_correct = 0
    n_total = 0
    avg_loss = 0

    # iterate over dataset
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            vid, _, labels = data
            vid, labels = vid.to(device=device), labels.to(device=device)
            out = model(vid).squeeze(3).squeeze(2)
            loss = criterion(out, labels)
            # top-1 accuracy
            n_correct += torch.sum(torch.argmax(out,dim=1)==labels)
            # other metrics
            avg_loss += loss.item()
            n_total += labels.shape[0]

    performance = np.array([avg_loss, n_correct]) / n_total

    return performance
