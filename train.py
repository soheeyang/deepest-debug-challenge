'''
WARNING: This code is full of bugs. Can you squash them all?
We've created a super-awesome sentiment classification tool
that recognizes whether a movie review is good or bad.
However, it does not work as expected... Why?
'''

import sys
import random
import csv

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

from prepro import prepro_filename
import data_utils
from model import *


# this line automatically determines which device to use
# if you have a fancy NVIDIA GPU the code uses its horsepower.
# if not, it's fine: the code uses CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# we use Naver Sentiment Movie Corpus v1.0
# from Lucy Park's [nsmc](https://github.com/e9t/nsmc)
# which is a dataset for binary sentiment classification of movie reviews.
class NsmcDataset(torch.utils.data.Dataset):
    def __init__(self, dtype):
        filename = prepro_filename  # data/prepro.npz
        assert dtype in ['train', 'test']
        with np.load(filename) as f:
            self.x, self.y = f[f'x_{dtype}'], f[f'y_{dtype}']
        
    def __getitem__(self, ind):
        return self.x[ind], self.y[ind]

    def __len__(self):
        assert len(self.x) == len(self.y)
        return len(self.x)


# it is not necessary to change the number of epochs to make the code work.
# only one epoch is enough to see if the model works.
def main(epochs=1):
    # define the train & test pytorch DataLoader
    train_loader = torch.utils.data.DataLoader(
        NsmcDataset('train'),
        batch_size=64,
        shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        NsmcDataset('test'),
        batch_size=64,
        shuffle=False)

    ##########################################################################################

    ##############################  Neural Network Definition & Training  ##############################

    # define the model
    # .to(device) automatically loads the model to the pre-defined device (GPU or CPU)
    neural_net = CnnClassifier(len(data_utils.vocabs)).to(device)

    # we use an optimizer that trains the model
    # heard that Adam is good, so use it
    optimizer = optim.Adam(neural_net.parameters(), lr=1)

    # now we defined all the necessary things, let's train the model
    print('\n' + 'training phase')
    neural_net.train()
    for epoch in range(epochs):
        for batch_ind, (input_data, target_data) in enumerate(train_loader):
            # pytorch needs to "zero-fill" the gradients at each train step
            # otherwise, the model adds up the gradients: not what you would expect
            neural_net.zero_grad()

            # put the input & target data to the auto-defined device (GPU or CPU)
            input_data, target_data = input_data.to(device), target_data.to(device)

            # feed input data to the network
            output = neural_net(input_data)

            # we define how well the model performed by comparing the output to target data
            # cross entropy is a natural choice
            # first, convert the target data to one-hot
            target_data_onehot = torch.zeros(target_data.size(0), 2).to(device)
            target_data_onehot.scatter_(1, target_data.unsqueeze(1), 1)

            # then, calculate the cross entropy error
            loss = -torch.mean(torch.sum(torch.mul(target_data_onehot, torch.log(output)), dim=0))
            
            # train the model using backpropagation
            loss.backward()
            optimizer.step()

            # print the train log at every step
            if batch_ind % 1 == 0:
                train_log = 'Epoch {:2d}/{:2d}\tLoss: {:.6f}\tTrain: [{}/{} ({:.0f}%)]'.format(
                    epoch, epochs, loss.cpu().item(), batch_ind, len(train_loader),
                                100. * batch_ind / len(train_loader))
                print(train_log, end='\r')
                sys.stdout.flush()

    ##########################################################################################

    ##############################  Evaluation of the Trained Neural Network   ##############################
    print('\n' + 'evaluation phase')
    neural_net.eval()

    # let's test the trained AI: feed the test data and get the test accuracy
    correct = 0.
    test_loss = 0.

    # pytorch uses no_grad() context manager for evaluation phase: it does not store the history & grads
    # so it's much faster and memory-efficient
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as pbar:
            for batch_ind, (input_data, target_data) in enumerate(test_loader):
                # same as training phase
                input_data, target_data = input_data.to(device), target_data.to(device)

                output = neural_net(input_data)

                # get the index of the max probability
                pred = output.argmax(dim=-1)

                # add up prediction results
                correct += pred.eq(target_data.view_as(pred)).cpu().sum()

                # calculate cross entropy loss for target data: same as training
                target_data_onehot = torch.zeros(target_data.size(0), 2).to(device)
                target_data_onehot.scatter_(1, target_data.unsqueeze(1), 1)
                test_loss += -torch.sum(torch.sum(torch.mul(target_data_onehot, torch.log(output)), dim=0))
                pbar.update(1)
                # calculate cross entropy loss for target data: same as training

    # average out the test results
    print('test loss:', float(test_loss) / len(test_loader.dataset))
    print('test accuracy:', 100. * int(correct) / len(test_loader.dataset))


if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
    
    main(epochs=1)
