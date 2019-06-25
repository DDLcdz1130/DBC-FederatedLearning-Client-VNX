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





# data_transforms for different types of tasks

# inception v3 preprocessing as default
def get_data_transforms(mode="train", model_origin="Inception V3", mean=None, std=None, scale=None, input_shape=299):
    # mode: "train", "test", or "eval".
    # model_origin: use given or custom ones.
    if model_origin== "Inception V3":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        scale = 299
    # Use other models, then add 
    #elif model_origin== "??":
    #    mean=....
    
    
    elif model_origin=="custom":
        # Your own parameters passed by mean, std, scale arguments
        pass
    
    if mode =="eval":  # eval is for getting simply feed forward results
        print('creating data_transform')
        data_transforms = {'eval':transforms.Compose([
                    transforms.Resize((scale,scale)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                    ])}
        print('data_transform finished')

    elif mode == "train":  # train is for model training
        data_transforms = {'train': transforms.Compose([
                    transforms.Resize((scale,scale)),
                    transforms.RandomCrop(input_shape),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(degrees=45),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)]),
                    'val': transforms.Compose([
                    transforms.Resize((scale, scale)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                    ])}
    elif mode == "test":  # test is for model testing
        data_transforms = {'test': transforms.Compose([
                    transforms.Resize((scale,scale)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                    ])}
    
    return data_transforms


def feature_model_data_transforms(mode="train"):
    if mode=="train":
        data_transforms = {'train': transforms.Compose([
                    transforms.ToTensor()]),
                    'val': transforms.Compose([
                    transforms.ToTensor()
                    ])}
    if mode=="test":
        data_transforms = {'test': transforms.Compose([
                    transforms.ToTensor()])}
    if mode=="eval":
        data_transforms = {'eval': transforms.Compose([
                    transforms.ToTensor()])}
    
    return data_transforms



# Data loading

# Class MyImageFolder replaces ImageFolder for 'eval' mode!
class MyImageFolder(datasets.ImageFolder):
            def __getitem__(self, index):
                print('iterating myimagefolder')
                return super(MyImageFolder, self).__getitem__(index), self.imgs[index] # return image path
    
# Data loading function


# Class MyImageFolder replaces ImageFolder for 'eval' mode!

def load_data(data_dir, data_transforms, mode = 'eval', 
              batch_size=16, num_workers=None):
    
    if mode == 'eval':
        print('step1')
        if num_workers==None:
            num_workers=0
        else:
            pass
        # Note that in this case, data_dir does not have the train and test subfolders
           
        image_datasets = {'eval': MyImageFolder(data_dir, data_transforms['eval'])}
        #print('got image_datasets')
        #print(image_datasets)
        dataloaders = {'eval': torch.utils.data.DataLoader(image_datasets['eval'], batch_size=batch_size,shuffle=False, num_workers=num_workers) }
        #print('step 3')
        print(dataloaders)
        dataset_sizes = {'eval': len(image_datasets['eval'])}
        print(dataset_sizes)
        class_names = image_datasets['eval'].classes
        print(class_names)
        dict_c2i=image_datasets['eval'].class_to_idx
        print(dict_c2i)

    elif mode == 'test':
        if num_workers==None:
            num_workers=0
        else:
            pass

        image_datasets = {'test': datasets.ImageFolder(data_dir, data_transforms['test'])}
        dataloaders = {'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size,shuffle=False, num_workers=num_workers) }
        dataset_sizes = {'test': len(image_datasets['test'])}
        class_names = image_datasets['test'].classes
        dict_c2i=image_datasets['test'].class_to_idx
        
    elif mode == 'train':
        if num_workers==None:
            num_workers=0
        else:
            pass
        
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x]) for x in ['train', 'val']}
        dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size,
                                                 shuffle=True, num_workers=num_workers), 
                       'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes
        dict_c2i=image_datasets['train'].class_to_idx
        
    return dataloaders, dataset_sizes, class_names, dict_c2i   #dict_c2i is the dictionary of class to index

        

        


"""""""""""""""""""""""""""""""""
Feature extraction data output setup

"""""""""""""""""""""""""""""""""

def feature_eval_prep(class_to_idx, feat_path="features"):

    feat_path="features"
    # Please delete features folder if it previously exists
    if os.path.isdir(feat_path)==False:
        os.mkdir(feat_path)
    else:
        sys.exit("Please rename or delete the previous "+feat_path+" folder for new run!")
    class_names=list(class_to_idx.keys())
    for class_name in class_names:
        os.mkdir(os.path.join(feat_path, class_name))
    # Get dict_i2c to be used later in getting class names of features
    dict_c2i=class_to_idx
    class_names, idx=zip(*dict_c2i.items())
    dict_i2c=dict(zip(idx, class_names))
    
    return dict_i2c


"""""""""""""""""""""""""""""""""
Rearranging files under source folder (data_folder) with class subfolders 
into train/val/test folders under target fold (targ_folder)

"""""""""""""""""""""""""""""""""
def refolder(data_folder, targ_folder, train_fraction=0.8, val_fraction=0.2, test_fraction=0.0, 
              remove_original=False):
    r=data_folder
    classes=[f for f in os.listdir(r) if os.path.isdir(os.path.join(r,f))]
    print('1 step')
    if os.path.isdir(targ_folder):
        shutil.rmtree(targ_folder)
    os.mkdir(targ_folder)
    print('step 2')
    sub_folder=os.path.join(targ_folder, 'train')
    os.mkdir(sub_folder)
    for c in classes:
        os.mkdir(os.path.join(sub_folder,c))
    
    sub_folder=os.path.join(targ_folder, 'val')
    os.mkdir(sub_folder)
    for c in classes:
        os.mkdir(os.path.join(sub_folder,c))

    if test_fraction!=0:
        sub_folder=os.path.join(targ_folder, 'test')
        os.mkdir(sub_folder)
        for c in classes:
            os.mkdir(os.path.join(sub_folder,c))
    
    for c in classes:
        files=glob.glob(os.path.join(r,c,"*"))
        random.shuffle(files)
        train_n=int(len(files)*train_fraction)
        for f in files[:train_n]:
            filename = os.path.basename(f)
            copyfile(f, os.path.join(targ_folder,'train', c,filename))
        
        if test_fraction==0:
            for f in files[train_n:]:
                filename = os.path.basename(f)
                copyfile(f, os.path.join(targ_folder,'val', c,filename))
        
        elif test_fraction!=0:
            val_n=int(len(files)*val_fraction)
            for f in files[train_n:(train_n+val_n)]:
                filename = os.path.basename(f)
                copyfile(f, os.path.join(targ_folder,'val', c,filename))
            for f in files[(train_n+val_n):]:
                filename = os.path.basename(f)
                copyfile(f, os.path.join(targ_folder,'test', c,filename))
        
        if remove_original==True:
            shutil.rmtree(data_folder)

        


