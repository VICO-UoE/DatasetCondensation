import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import get_dataset, get_network

parser = argparse.ArgumentParser(description='Parameter Processing')
parser.add_argument('--data_path', type=str, default='data', help='dataset path')
batch_size = 512
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


trasform_train = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation((-45,45)),
    transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5),
]

channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset("PCAM32FULL", args.data_path, batch_size=batch_size)

train_dataloader = DataLoader(dst_train, batch_size=batch_size, shuffle=True, num_workers=6)
test_dataloader = DataLoader(dst_test, batch_size=batch_size, shuffle=False, num_workers=6)

# Define the model
model = get_network("ConvNetW32", channel, num_classes, im_size=im_size)

# Define the loss function
criterion = nn.CrossEntropyLoss()

learning_rate = 0.01
# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_acc = 0
    for i, (images, labels) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += (outputs.argmax(1) == labels).sum().item()
    print('Epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}'.format(epoch, train_loss / len(train_dataloader), train_acc / len(dst_train)))

    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_acc += (outputs.argmax(1) == labels).sum().item()
    print('Epoch: {}, Test Loss: {:.4f}, Test Acc: {:.4f}'.format(epoch, test_loss / len(test_dataloader), test_acc / len(dst_test)))


