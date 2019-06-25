from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np

from torchvision import datasets, models, transforms
from torch.autograd import Variable
from PIL import Image

import os
import ntpath
import sys
import glob
import random
from tqdm import tqdm
import time
import argparse
import torchvision
import copy
import pdb
import shutil
from shutil import copyfile



# Load pretrained base model

def load_model(model_path="base_model_1.pt"):
    # torch.load() needs pickle.py location. We will add code in future in the case that default does not work.
    try:
        model=torch.load(model_path)
    except Exception as e:
        print("Failed to load model")
        print(e)
    return model


# Save model

def save_model(model, model_path="new_model.pt"):
    # torch.save() needs pickle.py location. We will add code in future in the case that default does not work.
    try:
        torch.save(model, model_path)
    except Exception as e: 
        print("Failed to save model.")
        print(e)





##### Note: each img can only have 1 label for the following code

def eval_model(model, dataloaders, dataset_sizes,  use_gpu, batch_size, dict_i2c, feat_path='features'):
    print("===>Test begains...")
    #since = time.time()
    phase='eval'
    model.eval()
    #running_corrects = 0.0
    #out_list=[]
    out_arr=[]
    # Iterate over data
    i=0 
    # Create feature path folder
    if os.path.isdir(feat_path)==False:
        print('Make directory and class subdirectories for the output data!')
        return
    for data in tqdm(dataloaders[phase]):
        # get the inputs
        (inputs, labels), (paths, _) = data
        
        
        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            #labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
        print('inputs:')
        print(type(inputs))

        # forward
        outputs = model(inputs)
        if type(outputs) == tuple:
            outputs, _ = outputs
        #outputs=F.softmax(outputs,dim=1) 
        outputs=np.array(outputs.data.cpu())
        #outputs=outputs.sum(axis=0)
        labels=np.array(labels)
        for j in range(len(labels)):
            path=os.path.join(feat_path, dict_i2c[labels[j]], ntpath.basename(paths[j]))
                #np.save(str(i*batch_size+j),outputs[j,:,:,:])
            np.save(path,outputs[j,:,:,:])
        i=i+1

    
    print("output shape of last batch is: {}".format(outputs.shape))
    



"""""""""""""""""""""""""""""""""
Testing model 

"""""""""""""""""""""""""""""""""
def test_model(model, dataloaders, dataset_sizes,  use_gpu):
    print("===>Test begains...")
    since = time.time()
    phase='test'

    running_corrects = 0.0
    
    # Iterate over data.
    for data in tqdm(dataloaders[phase]):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # forward
        outputs = model(inputs)
        if type(outputs) == tuple:
            outputs, _ = outputs
        _, preds = torch.max(outputs.data, 1)

        running_corrects += preds.eq(labels).sum().item() 

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    acc = 100.* running_corrects / dataset_sizes[phase]
    print('Test Acc: {:.4f}'.format(acc))
    


"""""""""""""""""""""""""""""""""
Training model 

"""""""""""""""""""""""""""""""""
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, 
                use_gpu, num_epochs, output_dir, mixup = False, alpha = 0.1):
    #print("MIXUP".format(mixup))
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in tqdm(dataloaders[phase]):
                # get the inputs
                inputs, labels = data

                #augementation using mixup
                mixup=0
                if phase == 'train' and mixup:
                    #inputs = mixup_batch(inputs, alpha)
                    pass
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                if type(outputs) == tuple:
                    outputs, _ = outputs
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                #running_loss += loss.data[0]
                running_loss += loss.item()
                running_corrects += preds.eq(labels).sum().item() 
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model=copy.deepcopy(model)
                #best_model_wts = model.state_dict()
                
                print("Model Saving...")
                #state = {'net': model.state_dict()}
                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)
                torch.save(model, os.path.join(output_dir,'trained_model.t7'))
                print(r"Model Saved...")

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model.state_dict())
    return model







"""""""""""""""""""""""""""""""""
Training truncated model with only higher layers 

"""""""""""""""""""""""""""""""""

