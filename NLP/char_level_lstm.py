from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import sklearn
from imblearn.over_sampling import RandomOverSampler
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import unicodedata
import string

# Adapted from the PyTorch RNN Name Classification Tutorial
# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

class myLSTM(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers, dropout=0):
        super(myLSTM, self).__init__()
        self.output_size = output_size
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hc):
        L = len(input)
        #print(f'L: {L}')
        h, c = hc
        output, (h,c) = self.LSTM(input, (h,c))
        output = self.dropout(output)
        #print(f'output.shape: {output.shape}')
        #print(f'h.shape: {h.shape}')
        #print(f'c.shape: {c.shape}')
        #output = self.fc(output.view(L, -1))
        output = self.fc(output.view(L, -1))
        #print(f'output.shape: {output.shape}')
        output = self.softmax(output)
        return output[-1].view(1,-1), (h,c)

class CharLevelNameDataset(Dataset):
    def __init__(self, x, y):
        self.categories = y
        self.lines = x

    def __len__(self):
        return len(self.categories)

    def __getitem__(self, idx):
        category = self.categories[idx]
        line = self.lines[idx]
        return {'category': category, 'line': line}




def findFiles(path): return glob.glob(path)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )



# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

# Converts the output tensor from the network into the corresponding category
# Returns the category and index
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

# Returns a random item from the iterable l
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Generates a random category and line from that category. Both of these are
# converted to one-hot tensors to be used as inputs to the network. All of these
# are returned.
def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

# Returns the time elapsed since a reference time
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Returns the number of trainable parameters in a PyTorch model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print(findFiles('data/names/*.txt'))
    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)
    print(f'n_letters: {n_letters}')
    torch.manual_seed(0)
    rng = np.random.default_rng()

    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []

    for filename in findFiles('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines


    n_categories = len(all_categories)
    n_lines = sum(len(category_lines[category]) for category in all_categories)
    print(f'n_lines: {n_lines}')

    x = []
    y = []

    # Counting the number of samples in each category
    # The categories are very imbalanced, which is dealt with by using a random
    # subset of the more common categories and randomly resampling the less common
    # categories with replacement to make them all have the same count.
    y_counts = {category:len(category_lines[category]) for category in all_categories}
    max_count = max(y_counts.values())
    min_count = min(y_counts.values())
    mean_count = sum(y_counts.values()) // len(y_counts)
    print(f'y_counts: {y_counts}')
    print(f'max_count: {max_count}, min_count: {min_count}, mean_count: {mean_count}')

    for category in all_categories:
        # Downsampling the categories with the most examples to deal with imbalanced data
        rng.shuffle(category_lines[category])
        print(f'Before length of category {category}: {len(category_lines[category])}')
        new_len = min(mean_count, len(category_lines[category]))
        category_lines[category] = category_lines[category][:new_len]
        print(f'After length of category {category}: {len(category_lines[category])}')
        for line in category_lines[category]:
            x.append(line)
            y.append(category)
    x = np.array(x)
    y = np.array(y)
    # Splitting into training and validation sets, making sure to keep as close to
    # the same proportion in each category as the original dataset
    x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, stratify=y)
    # Randomly oversampling the categories with the least examples to deal with
    # imbalanced data
    ros = RandomOverSampler(random_state=42)
    x_train = x_train.reshape(-1,1)
    print(f'x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}')
    x_train, y_train = ros.fit_resample(x_train, y_train)
    print(f'x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}')
    # This fixed a recurring error: TypeError: default_collate: batch must
    # contain tensors, numpy arrays, numbers, dicts or lists; found <U19
    x_train = x_train.reshape(-1)
    print(f'x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}')
    # # Resampling to deal with unbalanced data
    # first = True
    # for category in all_categories:
    #     x_samples = x_train[y_train==category]
    #     y_samples = y_train[y_train==category]
    #
    #     #print(f'x_samples: {x_samples}, y_samples: {y_samples}')
    #     samples = np.vstack((x_samples,y_samples)).transpose()
    #     #print(f'x_samples.shape: {x_samples.shape}, y_samples.shape: {y_samples.shape}')
    #     #print(f'samples.shape {samples.shape}')
    #     upsampled_samples = sklearn.utils.resample(samples, replace=True, n_samples=max_count, random_state=42)
    #     #print(f'upsampled_samples.shape: {upsampled_samples.shape}')
    #     if not first:
    #         upsampled_x_train = np.concatenate((upsampled_x_train, upsampled_samples))
    #     else:
    #         first = False
    #         upsampled_x_train = upsampled_samples
    #
    # upsampled_x_train = np.array(upsampled_x_train)
    # print(f'upsampled_x_train.shape: {upsampled_x_train.shape}')
    # #print(f'upsampled_x_train: {upsampled_x_train}')
    #
    # x_train, y_train = np.hsplit(upsampled_x_train, 2)
    # #print(f'y_train: {y_train}')
    # print(f'x_train.shape: {x_train.shape}')
    # print(f'y_train.shape: {y_train.shape}')
    train_counts = {}
    val_counts = {}
    for label in y_train:
        if label in train_counts:
            train_counts[label] += 1
        else:
            train_counts[label] = 1
    for label in y_val:
        if label in val_counts:
            val_counts[label] += 1
        else:
            val_counts[label] = 1
    print(f'train_counts: {train_counts}')
    train_set = CharLevelNameDataset(x_train, y_train)
    val_set = CharLevelNameDataset(x_val, y_val)
    train_len = len(train_set)
    val_len = len(val_set)
    print(f'Training set length: {train_len}')
    print(f'Validation set length: {val_len}')


    batch_size = 1
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    print_every = 1000
    plot_every = 100
    print(f'n_categories: {n_categories}')

    lrs = [3e-2, 3e-1]
    ps = [0.3, 0.5]
    # hidden_size<=256 and num_layers<=2 works well on name language classification
    # Increasing these numbers prevents the network from learning
    # In certain configurations, it seems to always guess the same language
    hidden_sizes = [64, 96, 128]
    num_layers_list = [1, 2]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    for lr in lrs:
        for hidden_size in hidden_sizes:
            for num_layers in num_layers_list:
                for p in ps:
                    if num_layers == 1: # There are no dropout layers if there
                        p = 0           # is only one layer
                    model = myLSTM(output_size=n_categories, input_size=n_letters, hidden_size=hidden_size, num_layers=num_layers, dropout=p)
                    model = model.to(device)
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
                    criterion = nn.NLLLoss()
                    print(f'Model has {count_parameters(model)} parameters')
                    num_epochs = 4 # Originally using iters like in the example code. Will change to epochs
                    n_iters = num_epochs * n_lines
                    h = torch.zeros(num_layers, batch_size, hidden_size)
                    c = torch.zeros(num_layers, batch_size, hidden_size)


                    start = time.time()
                    losses = []
                    val_losses = []
                    running_loss = 0

                    torch.autograd.set_detect_anomaly(True)
                    #for iter in range(1, n_iters + 1):
                    for epoch in range(num_epochs):
                        print(f'Training epoch {epoch+1}, lr={lr}, h_size={hidden_size}, n_layers: {num_layers}, p={p}')
                        model.train()
                        iter = 1
                        for sample in train_loader:
                            category = sample['category'][0]
                            line = sample['line'][0]
                            #print(f'category: {category}, line: {line}')
                            #category = category[0]
                            #line = line[0]
                            #category = category[2:-2]
                            #category, line = sample
                        # rng.shuffle(all_categories)
                        # for category in all_categories:
                        #     print(f'Training category: {category}')
                        #     rng.shuffle(train_categories[category])
                        #     for line in train_categories[category]:
                            #print(f'category: {category}, line: {line}')
                            #print(f'type: {type(category)}, category: {str(category)[2:-2]}')
                            #print(f'all_categories: {all_categories}')
                            category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
                            line_tensor = lineToTensor(line)

                    #category, line, category_tensor, line_tensor = randomTrainingExample()

                            #print(f'category: {category}, line: {line}')
                            optimizer.zero_grad()
                            #model.zero_grad()
                            #print(f'category_tensor.shape: {category_tensor.shape}')
                            #print(f'category_tensor: {category_tensor}')
                            #print(f'line_tensor.shape: {line_tensor.shape}')
                            #print(f'line_tensor: {line_tensor}')
                            #print(f'h.shape: {h.shape}')
                            #print(f'c.shape: {c.shape}')
                            category_tensor = category_tensor.to(device)
                            line_tensor = line_tensor.to(device)
                            h = h.to(device)
                            c = c.to(device)
                            # for i in range(len(line_tensor)):
                            #     print(f'input shape: {line_tensor[i].shape}')
                            output, (h, c) = model(line_tensor, (h, c))
                            h = h.detach()
                            c = c.detach()
                            #print(f'output.shape: {output.shape}')
                            #print(f'h.shape: {h.shape}')
                            #print(f'c.shape: {c.shape}')
                            loss = criterion(output, category_tensor)
                            loss.backward()
                            optimizer.step()
                            #output, loss = train(category_tensor, line_tensor)
                            running_loss += loss.item()

                            # Print iter number, loss, name and guess
                            if iter % print_every == 0:
                                guess, guess_i = categoryFromOutput(output)
                                correct = '✓' if guess == category else '✗ (%s)' % category
                                print('lr=%f, h_size=%d, %d (%s) %.4f %s / %s %s' % (lr, hidden_size, iter, timeSince(start), loss, line, guess, correct))

                            # Add current loss avg to list of losses
                            if iter % plot_every == 0:
                                losses.append(running_loss / plot_every)
                                running_loss = 0
                            iter += 1

                        model.eval()
                        running_val_loss = 0
                        val_iter = 0
                        with torch.no_grad():
                            print(f'Validation epoch: {epoch+1}, lr={lr}, h_size={hidden_size}, n_layers={num_layers}, p={p}')
                            for sample in val_loader:
                                #category, line = sample
                                category = sample['category'][0]
                                line= sample['line'][0]
                            # for category in all_categories:
                            #     for line in val_categories[category]:
                                category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
                                line_tensor = lineToTensor(line)
                                optimizer.zero_grad()

                                category_tensor = category_tensor.to(device)
                                line_tensor = line_tensor.to(device)
                                h = h.to(device)
                                c = c.to(device)
                                # for i in range(len(line_tensor)):
                                #     print(f'input shape: {line_tensor[i].shape}')
                                output, (h, c) = model(line_tensor, (h, c))
                                val_loss = criterion(output, category_tensor)
                                running_val_loss += val_loss.item()
                                val_iter += 1
                            val_losses.append(running_val_loss / val_iter)
                            print(f'Validation loss in epoch {epoch+1}: {val_losses[-1]}')


                    num_per_epoch = len(losses) // len(val_losses)
                    val_xs = [num_per_epoch*(i+1) for i in range(num_epochs)]
                    plt.figure()
                    plt.plot(losses, label='Training Loss')
                    plt.plot(val_xs, val_losses, label='Validation Loss')
                    plt.title(f'Learning Rate: {lr}, Hidden Size: {hidden_size}, Num Layers: {num_layers}, Dropout: {p}')
    plt.show()
