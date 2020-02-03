import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

__all__ = ['get_model', 'train', 'validate']

#############################
#   Data
#############################

data_dir = r'data/input'
data_dir_ext = r'data/annotation_path'

def get_optical_flow_transform():
    # TODO: transform 4d video to 3d optical flow tensor
    return None

def load_data(transform, train=True):
    dataset = torchvision.datasets.HMDB51(data_dir,
                                        data_dir_ext,
                                        frames_per_clip=5,
                                        step_between_clips=5,
                                        train=train)
    data_loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=32,
                                        shuffle=train)
    return data_loader

#############################
#   Model
#############################

def get_model():
    # TODO: define model
    model = nn.Sequential(
        nn.Conv3d(1,16,3),
        nn.ReLU(),
        nn.Conv3d(16,16,3),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1000,51)
    )
    return model

def train(model, exp_config):
    # training setup
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = get_optical_flow_transform()
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
            vid, labels = data
            vid, labels = vid.to(device=device), labels.to(device=device)
            optimizer.zero_grad()
            out = model(vid)
            loss = criterion(out, labels)
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

def validate(model):
    # validation setup
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = get_optical_flow_transform()
    data_loader = load_data(transform, train=False)
    criterion = nn.CrossEntropyLoss()

    # store results
    performance = np.zeros([len(data_loader)])

    # iterate over dataset
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            vid, labels = data
            vid, labels = vid.to(device=device), labels.to(device=device)
            out = model(vid)
            loss = criterion(out, labels)
            performance[i] = loss.item()

    return performance
