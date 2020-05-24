import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision import transforms, utils
import cv2
from PIL import Image
#plt.ion()

class VideoCarSpeedDataset(Dataset):

    def __init__(self, root_dir, augment_data=False):
        self.root_dir = root_dir
        speed_file = os.path.join(root_dir, 'train.txt')
        with open(speed_file, 'r') as f:
            self.speeds = [float(x) for x in f.readlines()]
        self.augment_data = augment_data
        if augment_data:
            self.dir_dict = {0: 'train_inputs', 1: 'vert_flip_inputs', 2: 'horiz_flip_inputs', 3: 'both_flip_inputs', 4: 'darker_inputs', 5: 'brighter_inputs'}

    def __len__(self):
        #return 10 # A way to get a simple baseline
        if self.augment_data:
            return 6*(len(self.speeds)-1)
        else:
            return len(self.speeds)-1

    def __getitem__(self, idx):
        # Originally loaded images every time getitem was called to save memory
        # all in init to hopefully speed things up dramatically.
        # This is not enough memory because it takes way more memory to store the
        # image array than the disk space required for the compressed JPEG image
        # Preprocessing all frame pairs into optical flow images and saving them
        # I have over 10GB of free memory, however, so I will try loading them
        # should still speed things up a lot. The only image processing required
        # before sending them to the neural network is transposing the dimensions
        # into CHW order instead of HWC, which OpenCV uses.
        if self.augment_data:
            dir_idx = idx // (len(self.speeds)-1)
            idx = idx % (len(self.speeds)-1)
            path = self.dir_dict[dir_idx]
            input_path = os.path.join(self.root_dir, f'{path}/input{idx}.png')
        else:
            input_path = os.path.join(self.root_dir, f'train_inputs/input{idx}.png')
        input = Image.open(input_path)
        preprocess = transforms.Compose([transforms.Resize((240, 320)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        input = preprocess(input)
        input = input.double()
        speed1 = self.speeds[idx]
        speed2 = self.speeds[idx+1]
        speeds = torch.tensor([speed1, speed2])
        return {'input': input, 'speeds': speeds.double()}


class Net(nn.Module):
    def __init__(self, p=0.7, mean_speed=25):
        super(Net, self).__init__()
        self.resnet = resnet.resnet50(pretrained=True, progress=True)
        self.fc = nn.Linear(1000, 2)
        self.fc.bias.data.fill_(mean_speed)

    def forward(self, x):
        #print(f'x.shape: {x.shape}')
        #x = F.interpolate(x, (240, 320))
        x = F.leaky_relu(self.resnet(x))
        # Trying to make an ad-hoc sort of model is not productive
        # Will try to modify a ResNet as baseline
        # ResNet18 successfully overfits a training set of 10 samples. Will next
        # try performance on whole training set over 5 epochs
        # I'm not actually using the expected inputs for the PyTorch ResNet
        # implementations. It expects larger input images that are normalized to
        # particular means and standard deviations. I will try that next, if the
        # larger image size will actually fit in GPU memory.
        x = self.fc(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    #plt.ion() # I kept getting fatal errors trying to plot in the middle of training
    root_dir = '/home/michael/speedchallenge/data'
    car_speed_dataset = VideoCarSpeedDataset('/home/michael/speedchallenge/data', augment_data=False)
    n = len(car_speed_dataset)
    print(f'Length: {n}')
    speed_file = os.path.join(root_dir, 'train.txt')
    with open(speed_file, 'r') as f:
        speeds = [float(x) for x in f.readlines()]
    mean_speed = np.mean(speeds)
    print(f'Mean speed: {mean_speed}')
    torch.manual_seed(0) # To make sure that the train/validation split is always the same
    train_set, validation_set = torch.utils.data.random_split(car_speed_dataset, [int(.8*n), n-int(.8*n)])
    #train_set = car_speed_dataset
    print(f'Training set length: {len(train_set)}')
    print(f'Validation set length: {len(validation_set)}')
    # Batch size of 200 seems to be largest that will fit in 8GB of GPU memory
    # Batch size of 160 for Resnet18 and no validation set
    # Batch size of 68 for ResNet50 and no validation set
    # Batch size of 60 for ResNet50 with validation set
    train_loader = DataLoader(train_set, batch_size=20, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(validation_set, batch_size=140, shuffle=True, num_workers=1, pin_memory=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    train_losses = {}
    val_losses = {}
    # Training loss of 3.173 after 50 epochs of ResNet18
    # Best Validation loss is 3.365 after 45 epochs of ResNet50
    # The specific hyperparameters have been edited several times in this file
    lrs = [0.00003, 0.0001, 0.0003]
    ps = [0.3] # Leftover from when I had dropout layers. I might put them back

    num_epochs = 3 # This is arbitrary
    for lr in lrs:
        for p in ps:
            net = Net(p, mean_speed=mean_speed)
            # This is the best model I've trained
            path = os.path.join(root_dir, 'models/45_800')
            net = torch.load(path)
            net = net.double()
            print(f'Network has {count_parameters(net)} parameters')
            net = net.to(device)
            #print(net)
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
            #optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
            criterion = nn.MSELoss()
            train_losses[(lr, p)] = []
            val_losses[(lr, p)] = []
            for epoch in range(num_epochs):
                running_loss = 0
                for i, sample in enumerate(train_loader, 0):
                    input = sample['input']
                    speeds = sample['speeds']
                    net.train()

                    optimizer.zero_grad()
                    input = input.to(device)
                    speeds = speeds.to(device)
                    outputs = net(input)
                    loss = criterion(outputs, speeds)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    if i % 50 == 49:
                        avg_loss = running_loss / 50
                        running_loss = 0.0
                        train_losses[(lr, p)].append(avg_loss)

                        running_val_loss = 0
                        # Testing the model on the validation set without updating
                        # weights
                        with torch.no_grad():
                            net.eval()
                            for val_sample in val_loader:
                                val_input = val_sample['input']
                                val_speeds = val_sample['speeds']
                                val_input = val_input.to(device)
                                val_speeds = val_speeds.to(device)

                                val_outputs = net(val_input)
                                val_loss = criterion(val_outputs, val_speeds)
                                running_val_loss += val_loss.item()
                        avg_val_loss = running_val_loss / len(val_loader)
                        val_losses[(lr, p)].append(avg_val_loss)
                        print(f'lr={lr}, [{epoch+1}, {i+1}] Training loss: {avg_loss:.3f}, Validation loss: {avg_val_loss:.3f}')



    for lr in lrs:
        for p in ps:
            fig = plt.figure()
            print(f'lr: {lr}, p: {p} len(train_losses[lr]): {len(train_losses[(lr, p)])}')
            p1 = plt.plot(train_losses[(lr, p)], label='Training Loss')
            p2 = plt.plot(val_losses[(lr, p)], label='Validation Loss')
            plt.title(f'Learning rate: {lr}, p: {p}')
            plt.legend()
    plt.show()
