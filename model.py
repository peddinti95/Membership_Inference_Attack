import torch
import torch.nn as nn
import torch.nn.functional as F


class Net_mnist(nn.Module):
    def __init__(self):
        super(Net_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 128)
        self.fc2 = nn.Linear(128, 10)
        
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.tanh(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class Net_kmnist(nn.Module):
    def __init__(self):
        super(Net_kmnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 128)
        self.fc2 = nn.Linear(128, 10)
        
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.tanh(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class Net_fashionmnist(nn.Module):
    def __init__(self):
        super(Net_fashionmnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 128)
        self.fc2 = nn.Linear(128, 10)
        
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.tanh(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class Net_cifar10(nn.Module):
    def __init__(self):
        super(Net_cifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class Net_stl10(nn.Module):
    def __init__(self):
        super(Net_stl10, self).__init__()
        self.conv1 = nn.Conv1d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv1d(6, 16, 5)
        self.fc1= nn.Linear(49 * 96* 96, 128)
        #self.fc1 = nn.Linear(16 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.view(64,6,48*96)
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        #x = x.view(-1, 49*96*96)
        #x = x.view(-1, 16 * 5 * 5)
        x = x.view(x.size(0),1,1)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class Net_cifar100(nn.Module):
    def __init__(self):
        super(Net_cifar100, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 100)
        
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
