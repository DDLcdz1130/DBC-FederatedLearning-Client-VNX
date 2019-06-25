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


# Use first few layers as a model

def model_take_lower_layers(model, layer_number=3):

    try:
        feat_model=nn.Sequential(*list(model.children())[:layer_number])
    except Exception as e:
        print("Problem separating the model")
        print(e)
        feat_model=None
        
    return feat_model



def model_take_upper_layers(model, layer_number=3):

    try:
        feat_model=nn.Sequential(*list(model.children())[layer_number:])
    except Exception as e:
        print("Problem separating the model")
        print(e)
        upper_model=None
        
    return upper_model


