# Defect detection using Federated Learning

This is a computing framework for developing models for visual inspection of defect products in manufacturing lines.

This is the basic version of a federated learning process, following this procedure for each training cycle:

Client model training -> Client model upload -> Server model integrate -> Server model download by client -> client transfer learning 

Client_train.py defines the node_training function to automated the training process. run.py is the actual execution code.

Since each cycle typically takes long time, the run only covers one training cycle and the client with run the code once each cycle.



