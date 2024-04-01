import os
import torch
import argparse

import data_setup, model, engine, utils
from torchvision import transforms

#SET UP HYPERPARAMETERS
BATCH_SIZE = 64
NUM_WORKERS = 0
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
HEIGHT = 320
WIDTH = 480
NUM_CLASSES = 21

#set up path data
voc_dir = 'VOCdevkit/VOC2012'

def main():
    #Set up device
    devices = 'cuda' if torch.cuda.is_available() else 'cpu'

    #Set up model
    net = model.FCN_Resnet18(NUM_CLASSES)

    #Set up optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    #Set up loss function
    loss_fn = utils.loss_fn

    #Set up data
    colormap2label = data_setup.voc_colormap2label()
    train_iter, test_iter = data_setup.create_dataloader(BATCH_SIZE, (HEIGHT, WIDTH), voc_dir, colormap2label, NUM_WORKERS)

    #Train model
    engine.train_epoch(net, train_iter, test_iter, loss_fn, optimizer, NUM_EPOCHS, devices)

if __name__ == '__main__':
    main()
    