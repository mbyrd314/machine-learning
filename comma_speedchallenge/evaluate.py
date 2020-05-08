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


class CarSpeedTestDataset(Dataset):

    def __init__(self, root_dir, length):
        self.root_dir = root_dir
        self.length = length

    def __len__(self):
        #return 10 # A way to get a simple baseline
        return self.length

    def __getitem__(self, idx):

        input_path = os.path.join(self.root_dir, f'test_inputs/input{idx}.png')
            # img1_name = os.path.join(self.root_dir, f'train_frames/frame{idx}.jpg')
            # img2_name = os.path.join(self.root_dir, f'train_frames/frame{idx+1}.jpg')
        input = Image.open(input_path)
        preprocess = transforms.Compose([transforms.Resize((240, 320)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        input = preprocess(input)
        input = input.double()
        return {'input': input, 'idx': idx}

# This has to be included for torch.load() to work correctly
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
    length=10797
    root_dir = '/home/michael/speedchallenge/data'
    test_set = CarSpeedTestDataset('/home/michael/speedchallenge/data', length=length)
    test_loader = DataLoader(test_set, batch_size=140, shuffle=False, num_workers=1, pin_memory=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    predicted_speeds = [0]*(length+1)

    path = os.path.join(root_dir, 'models/45_800')
    net = torch.load(path)
    net = net.double()
    print(f'Network has {count_parameters(net)} parameters')
    net = net.to(device)
    #print(net)
    net.eval()

    with torch.no_grad():
        for i, sample in enumerate(test_loader, 0):
            print(f'i: {i}')
            input = sample['input']
            idx = sample['idx']
            input = input.to(device)
            outputs = net(input)
            for j in range(len(outputs)):
                index = idx[j]
                predicted_speeds[index] = float(outputs[j][0])
                if index == length-1:
                    predicted_speeds[index+1] = float(outputs[j][1])

    outfile = '/home/michael/speedchallenge/data/test.txt'
    with open(outfile, 'w') as f:
        for speed in predicted_speeds:
            f.write(f'{speed:.6f}\n')
