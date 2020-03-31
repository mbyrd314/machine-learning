import os
import torch
#import pandas as pd
#import skimage
#from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
#from torchvision import transforms, utils
import cv2
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
            # Takes too much memory to store all images
            # Instead, decided to preprocess all images with optical flow in another function
            # self.inputs = {}
            # for i in range(6):
            #     self.inputs[i] = []
            #     images = []
            #     for j in range(len(self.speeds)):
            #         path = os.path.join(root_dir, f'/{self.dir_dict[i]}/frame{j}.jpg')
            #         image = cv2.imread(path)
            #         #self.images[i].append(image)
            #         images.append(image)
            #     for j in range(len(self.speeds)-1):
            #         image1 = images[j]
            #         image2 = images[j+1]
            #         hsv = np.zeros_like(image1)
            #         image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            #         image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            #         flow = cv2.calcOpticalFlowFarneback(image1, image2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            #         mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            #         hsv[..., 1] = 1 # Saturation is max
            #         hsv[..., 0] = ang*180/np.pi/2 # Converting hue to the right range
            #         hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            #         rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            #         rgb = skimage.img_as_float64(rgb.transpose(2, 0, 1))
            #         self.inputs[i].append(rgb)

        # else:
        #     self.inputs = {0: []}
        #     images = []
        #     for i in range(len(self.speeds)):
        #         path = os.path.join(root_dir, f'train_frames/frame{i}.jpg')
        #         image = cv2.imread(path)
        #         #self.images[0].append(image)
        #         images.append(image)
        #     for i in range(len(self.speeds)-1):
        #         image1 = images[i]
        #         image2 = images[i+1]
        #         hsv = np.zeros_like(image1)
        #         image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        #         image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        #         flow = cv2.calcOpticalFlowFarneback(image1, image2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        #         mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        #         hsv[..., 1] = 1 # Saturation is max
        #         hsv[..., 0] = ang*180/np.pi/2 # Converting hue to the right range
        #         hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        #         rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        #         rgb = skimage.img_as_float64(rgb.transpose(2, 0, 1))
        #         self.inputs[0].append(rgb)


    def __len__(self):
        if self.augment_data:
            return 6*(len(self.speeds)-1)
        else:
            return len(self.speeds)-1

    def __getitem__(self, idx):
        # Originally loaded images every time getitem was called to save memory
        # I have over 10GB of free memory, however, so I will try loading them
        # all in init to hopefully speed things up dramatically.
        # This is not enough memory because it takes way more memory to store the
        # image array than the disk space required for the compressed JPEG image
        # Preprocessing all frame pairs into optical flow images and saving them
        # should still speed things up a lot. The only image processing required
        # before sending them to the neural network is transposing the dimensions
        # into CHW order instead of HWC, which OpenCV uses.
        if self.augment_data:
            dir_idx = idx // (len(self.speeds)-1)
            idx = idx % (len(self.speeds)-1)
            # input = self.inputs[dir_idx][idx]
            path = self.dir_dict[dir_idx]
            input_path = os.path.join(self.root_dir, f'{path}/input{idx}.png')
            # img1_name = os.path.join(self.root_dir, f'{path}/frame{idx}.jpg')
            # img2_name = os.path.join(self.root_dir, f'{path}/frame{idx+1}.jpg')
            # #print(f'path: {path}, img1_name: {img1_name}, img2_name: {img2_name}')
        else:
            # input = self.inputs[0][idx]
            input_path = os.path.join(self.root_dir, f'train_inputs/input{idx}.png')
            # img1_name = os.path.join(self.root_dir, f'train_frames/frame{idx}.jpg')
            # img2_name = os.path.join(self.root_dir, f'train_frames/frame{idx+1}.jpg')
        # #image1 = io.imread(img1_name)
        # image1 = cv2.imread(img1_name)
        # #image1.resize(120,160)
        # #image1 = skimage.img_as_float64(image1.transpose(2,0,1))
        # #image2 = io.imread(img2_name)
        # image2 = cv2.imread(img2_name)
        # #image2.resize(120, 160)
        # #image2 = skimage.img_as_float64(image2.transpose(2,0,1))
        # #image1 = skimage.img_as_float64(io.imread(img1_name).transpose(2, 0, 1)) # Skimage loads images in H, W, C format
        # #image2 = skimage.img_as_float64(io.imread(img2_name).transpose(2, 0, 1))
        # hsv = np.zeros_like(image1)
        # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        input = cv2.imread(input_path)
        input = cv2.normalize(input.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        input = input.transpose(2, 0, 1) # Converting to CHW format
        speed1 = self.speeds[idx]
        speed2 = self.speeds[idx+1]
        speeds = torch.tensor([speed1, speed2])
        #speeds = torch.stack((speed1, speed2), 1)
        # This returned both images when I was just taking the difference, but
        # I should have just returned their difference
        #return {'image1': image1, 'image2': image2, 'speed1': speed1, 'speed2': speed2}
        #return {'image': image2-image1, 'speeds': speeds}

        # This computes the dense optical flow between the two frames, interprets it
        # as HSV, and then converts that to a BGR image that will be fed into the network
        #print(f'image1.size: {image1.size}')
        #print(f'image2.size: {image2.size}')
        #print(f'speeds.size: {speeds.size()}')
        # flow = cv2.calcOpticalFlowFarneback(image1, image2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # hsv[..., 1] = 1 # Saturation is max
        # hsv[..., 0] = ang*180/np.pi/2 # Converting hue to the right range
        # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # rgb = skimage.img_as_float64(rgb.transpose(2, 0, 1))
        return {'input': input, 'speeds': speeds.double()}

class Net(nn.Module):
    def __init__(self, p=0.9):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 7, padding=3)
        self.bc1 = nn.BatchNorm2d(3)
        #self.dropout1 = nn.Dropout(p=p)
        #self.conv2 = nn.Conv2d(32, 8, 5, padding=2)
        self.fc1 = nn.Linear(3*480*640, 32)
        self.bc2 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(p=p)
        self.fc2 = nn.Linear(32, 16)
        self.bc3 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(p=p)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        #print(f'x.shape: {x.shape}')
        #x = F.interpolate(x, (120, 160))
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.bc1(x))
        #print(f'x.shape: {x.shape}')
        #x = F.leaky_relu(self.dropout1(x))
        #x = F.leaky_relu(self.conv2(x))
        #print(f'x.shape: {x.shape}')
        x = x.reshape(-1, 3*480*640)
        #print(f'x.shape: {x.shape}')
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.bc2(x))
        #print(f'x.shape: {x.shape}')
        x = F.leaky_relu(self.dropout1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.bc3(x))
        x = F.leaky_relu(self.dropout2(x))
        #print(f'x.shape: {x.shape}')
        x = self.fc3(x)
        #print(f'x.shape: {x.shape}')
        return x


def load_image_batch(batched_sample):
    # Simple utility function used to test the custom dataloader
    # Not needed for the actual model.
    img1_batch = batched_sample['image1']
    img2_batch = batched_sample['image2']
    batch_size = len(img1_batch)
    #print(f'Batch size: {batch_size}')
    im_size = img1_batch.size(2)
    #print(f'Im size: {im_size}')
    grid_border_size = 2
    #img1_batch = img1_batch.permute(0, 3, 1, 2)
    #grid = utils.make_grid(img1_batch)
    #print(f'Grid size: {grid.size()}')
    #plt.imshow(grid)
    #plt.imshow(grid.numpy().transpose(1,2,0))
    for i in range(batch_size):
        img1 = img1_batch[i]
        img2 = img2_batch[i]
        ax = plt.subplot(1, batch_size, i+1)
        plt.tight_layout()
        ax.set_title(f'Sample #{i}')
        ax.axis('off')
        plt.imshow(img1)
        plt.pause(.001)


    #for i in range(batch_size):


if __name__ == '__main__':
    #plt.ion() # I kept getting fatal errors trying to plot in the middle of training
    car_speed_dataset = VideoCarSpeedDataset('/home/michael/speedchallenge/data', augment_data=False)
    n = len(car_speed_dataset)
    print(f'Length: {n}')
    train_set, validation_set = torch.utils.data.random_split(car_speed_dataset, [int(.8*n), n-int(.8*n)])
    print(f'Training set length: {len(train_set)}')
    print(f'Validation set length: {len(validation_set)}')
    #dataloader = DataLoader(car_speed_dataset, batch_size=16, shuffle=True, num_workers=8)
    # Batch size of 200 seems to be largest that will fit in 8GB of GPU memory
    train_loader = DataLoader(train_set, batch_size=100, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(validation_set, batch_size=100, shuffle=True, num_workers=1, pin_memory=True)
    #fig = plt.figure()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    # net = Net()
    # net = net.double()
    # net = net.to(device)
    # #print(net)
    # optimizer = torch.optim.Adam(net.parameters())
    # #optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    # criterion = nn.MSELoss()

    train_losses = {}
    val_losses = {}
    # Results for only using difference of two frames and only original frames
    # with no data augmentation.
    # .001 is the best of these from the first run
    #lrs = [10**(-x) for x in range(2, 5)]
    # Best performance on second run for lr=0.0025 and p=0.9 after 25 epochs
    # Training loss: 14.743, Validation loss: 15.882
    # Best performance on third run for lr=0.00075 and p=.7 after 25 epochs
    # Training loss: 5.851  Validation loss: 7.813
    # Could probably get better results by further tuning hyperparameters and
    # training for more epochs, but I determined that this would take too long
    # when my intended strategy was to use optical flow instead of just the difference
    # in the two frames.
    # Best performance on first run of 5 epochs without data augmentation using optical flow as input
    # p=0.5 and lr=0.0025 : Training loss=10.577 Validation loss=10.484
    # Best performance on second optical flow run of 5 epochs with p=0.4 and lr=0.003
    # Training Loss: 8.844  Validation Loss: 10.055
    # Best performance on second optical flow run of 5 epochs with p=0.4 and lr=0.003
    # Training loss: 8.075 Validation Loss: 8.611
    # I didn't get much better than 5 for training loss and validation loss at 25 epochs
    # of training. Performance might have improved with more epochs, but I am first trying to
    # to change the network architecture to not downsample the input image.
    #lrs = [0.00075, 0.001, 0.0025]
    # The larger lr values performed better on training and validation loss
    #lrs = [0.001, 0.003, 0.005]
    #lrs = [0.002, 0.003, 0.004]
    #lrs = [0.003]
    # Highest lr had best performance on second architecture
    #lrs = [0.0001, 0.001, 0.01]
    lrs = [0.01, 0.03, 0.1]
    #ps = [0.5, 0.7, 0.9]
    # Smaller p values tend to perform better on training and validation loss
    #ps = [0.4, 0.55, 0.7]
    #ps = [0.3, 0.4, 0.5]
    #ps = [0.4]
    # Lowest p value had best performance on second architecture
    #ps = [0.4, 0.5, 0.6]
    ps = [0.3, 0.4, 0.5]

    num_epochs = 5 # This is arbitrary
    for lr in lrs:
        for p in ps:
            net = Net(p)
            net = net.double()
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
                    #img1 = sample['image1']
                    #img2 = sample['image2']
                    input = sample['input']
                    #speed1 = sample['speed1']
                    #speed2 = sample['speed2']
                    speeds = sample['speeds']
                    #print(f'speed1: {speed1}')
                    #print(f'speed2: {speed2}')
                    #speeds = torch.stack((speed1, speed2), 1)

                    optimizer.zero_grad()

                    #input = img2 - img1 # Simplest idea. Difference of successive frames
                    #print(f'input.size: {input.size()}')
                    input = input.to(device)
                    speeds = speeds.to(device)
                    outputs = net(input)
                    loss = criterion(outputs, speeds)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    #losses[lr].append(loss.item())
                    if i % 25 == 24:
                        avg_loss = running_loss / 25
                        #print(f'lr={lr}, [{epoch+1}, {i+1}] loss: {avg_loss:.3f}')
                        running_loss = 0.0
                        train_losses[(lr, p)].append(avg_loss)

                        running_val_loss = 0
                        # Testing the model on the validation set without updating
                        # weights
                        with torch.no_grad():
                            for val_sample in val_loader:
                                #val_img1 = val_sample['image1']
                                #val_img2 = val_sample['image2']
                                val_input = val_sample['input']
                                #val_speed1 = val_sample['speed1']
                                #val_speed2 = val_sample['speed2']
                                val_speeds = val_sample['speeds']
                                #val_speeds = torch.stack((val_speed1, val_speed2), 1)

                                #val_input = val_img2 - val_img1
                                val_input = val_input.to(device)
                                val_speeds = val_speeds.to(device)

                                val_outputs = net(val_input)
                                val_loss = criterion(val_outputs, val_speeds)
                                running_val_loss += val_loss.item()
                        avg_val_loss = running_val_loss / len(val_loader)
                        val_losses[(lr, p)].append(avg_val_loss)
                        print(f'lr={lr}, p={p} [{epoch+1}, {i+1}] Training loss: {avg_loss:.3f}, Validation loss: {avg_val_loss:.3f}')
            # Plotting the training loss and validation loss for the model with
            # the given hyperparameters
            # fig = plt.figure()
            # print(f'lr: {lr}, p: {p} len(train_losses[lr, p]): {len(train_losses[(lr, p)])}')
            # #iters = [x+1 for x in range(len(train_losses[lr]))]
            # p1 = plt.plot(train_losses[(lr, p)], label='Training Loss')
            # p2 = plt.plot(val_losses[(lr, p)], label='Validation Loss')
            # plt.title(f'Learning rate: {lr}, p: {p}')
            # plt.legend()
            # #plt.show()
            # plt.pause(0.001)



    for lr in lrs:
        for p in ps:
            fig = plt.figure()
            print(f'lr: {lr}, p: {p} len(train_losses[lr]): {len(train_losses[(lr, p)])}')
            #iters = [x+1 for x in range(len(train_losses[lr]))]
            p1 = plt.plot(train_losses[(lr, p)], label='Training Loss')
            p2 = plt.plot(val_losses[(lr, p)], label='Validation Loss')
            plt.title(f'Learning rate: {lr}, p: {p}')
            plt.legend()

    # for i, sample in enumerate(dataloader):
    #     print(f"i: {i}, img1_size: {sample['image1'].size()}, img2_size: {sample['image2'].size()}")
    #     #load_image_batch(sample)
    #     # #sample = dataloader[i]
    #     # img1 = sample['image1']
    #     # img2 = sample['image2']
    #     # #print(f'i: {i}')
    #     # ax = plt.subplot(1, 4, i+1)
    #     # plt.tight_layout()
    #     # ax.set_title(f'Sample #{i}')
    #     # ax.axis('off')
    #     # plt.imshow(img1)
    #     # plt.pause(.001)
    #     # #plt.figure()
    #     # #plt.show()
    #     if i==3:
    #         break
    #     plt.figure()
    plt.show()
