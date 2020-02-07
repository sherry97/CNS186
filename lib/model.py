import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

__all__ = ['get_model', 'train', 'validate']

#############################
#   Data
#############################

data_dir = r'data/input'
data_dir_ext = r'data/annotations'
frame_size = 240

def optical_flow(x):
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
    # TODO: use OpenCV to calculate LK flow
    x = x.mean(dim=0,keepdim=True)
    assert x.shape == (1,frame_size,frame_size)
    return x

def generic_transform(x):
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
    # average over time
    x = x.mean(dim=0,keepdim=True)
    assert x.shape == (1,frame_size,frame_size)
    return x

def get_optical_flow_transform():
    return transforms.Lambda(lambda x : optical_flow(x))

def get_generic_transform():
    return transforms.Lambda(lambda x : generic_transform(x))

def load_data(transform, train=True):
    print('Start loading data...')
    dataset = torchvision.datasets.HMDB51(data_dir,
                                        data_dir_ext,
                                        frames_per_clip=10,
                                        step_between_clips=2,
                                        train=train,
                                        transform=transform)
    print(f'Loaded {len(dataset)} entries from {data_dir}. Convert to DataLoader...')
    data_loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=128,
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

def train(model, exp_config):
    # training setup
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Training on {device}.')
    if exp_config['transform_mode']:
        transform = get_optical_flow_transform()
    else:
        transform = get_generic_transform()
    data_loader = load_data(transform, train=True)
    dataset_size = len(data_loader)

    # init auxiliary training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                        lr=exp_config['lr'],
                        momentum=0.9,
                        weight_decay=0.001)
    n_epochs = exp_config['n_epochs']

    # store results
    performance = np.zeros([dataset_size*n_epochs])

    # iterate over dataset
    global_start = time.clock()
    for epoch in range(n_epochs):
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
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if exp_config['transform_mode']:
        transform = get_optical_flow_transform()
    else:
        transform = get_generic_transform()
    data_loader = load_data(transform, train=False)
    criterion = nn.CrossEntropyLoss()

    # store results
    n_correct = 0
    n_total = 0
    avg_loss = 0

    # iterate over dataset
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            vid, _, labels = data
            vid, labels = vid.to(device=device), labels.to(device=device)
            out = model(vid).squeeze(3).squeeze(2)
            loss = criterion(out, labels)
            n_correct += torch.sum(torch.argmax(out,dim=1)==labels)
            n_total += labels.shape[0]
            avg_loss += loss.item()

    performance = np.array([avg_loss, n_correct]) / n_total

    return performance
