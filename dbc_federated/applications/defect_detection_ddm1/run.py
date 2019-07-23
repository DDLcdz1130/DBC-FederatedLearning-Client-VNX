import sys
import os
os.chdir(sys.path[0])
print(os.getcwd())
from client_train import node_training
import argparse



# Define stage
parser = argparse.ArgumentParser(description='PyTorch inception Training')
parser.add_argument('--train_stage', default=1, type=int, help='')
parser.add_argument('--batch_size', default=16, type=int, help='')

args = parser.parse_args()
train_stage = args.train_stage
batch_size = args.batch_size
# For each cycle, first train with freeze_layers =0, then send to server (train_stage==1). 
# After received new model from server, then train with freeze_layers=1, and 
# deploy (train_stage==2).
print("batch_size = {}".format(batch_size))
if train_stage==1:
	freeze_layers=0
	node_training(data_dir='./data/', restore=1, model_path='./checkpoint_i/avg_model.t7',freeze_layers=freeze_layers,n_class=2,
                       batch_size=batch_size, epochs = 25, device_ids=[0], output_dir = './checkpoint_o/', new_model_name='new_model3.t7' )

elif train_stage==2:
	freeze_layers=1
	node_training(data_dir='./data/', restore=1, model_path='./checkpoint_i/avg_model.t7',freeze_layers=freeze_layers,n_class=2,
                       batch_size=batch_size, epochs = 25, device_ids=[0], output_dir = './checkpoint_o/', new_model_name='new_model3.t7' )
	 
