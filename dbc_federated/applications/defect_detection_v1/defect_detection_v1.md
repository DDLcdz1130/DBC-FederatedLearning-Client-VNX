# Defect detection using Federated Learning

This is a computing framework for developing models for visual inspection of defect products in manufacturing lines.

Our initial version as of 06/10/2019 enables feature evaluation and training of CNN models for defect visual detection by edge devices, as well as the training of base model in a server using the features extracted from edge devices, which ensures data privacy. Our framework is not concerned with bandwith problem at the current stage, since the application here does not require high data flow.

A training cycle typically operates like this:

First, the edge devices (clients) send a request to the server for model training.

Second, the server send a base model to the edges. The edges then extract features from their data using the deep learning model, which is showcased in Client_stage_2_modelling.py. Then the edges send the extracted feature data to the server. The server will train a new base model on the feature data (with labels), which is showcased in Server_stage_2_modelling.py.

Next, the server send the new base model to the edges. The edges can use the new base model and do transfer learning on each's own local data, as given in Client_stage_3_modelling.py. Now, each client has its own model. The model can be deployed.

The Federated Learning process can be in various forms. You can use the modules in lib folder in your own way. For convenience of package development, the edge devices and the server share the same modules and functions in lib folder.


## Client application scripts

Client_stage_2_modelling.py executes the computations to be done by the client at Stage 2. 

Client_stage_3_modelling.py executes the computations to be done by the client at Stage 3.


## Server application scripts

The server algorithm is showcased in Server_stage_2_modelling.py. This script is solely used for explanation of the FL process. The server has its own algorithms and libraries. 


## Program pipelines

All three scripts follow the pipeline of 1) setting up parameters; 2) loading data; 3) model training, testing, or evaluating; 3) save model and organize the output data if any


