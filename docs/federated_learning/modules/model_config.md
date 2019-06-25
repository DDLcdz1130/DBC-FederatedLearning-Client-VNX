# Configuring models

This library (model_config) gives you a simplified and tested model manipulation functions. 


One can either take the lower n layers from a well defined model or take the upper n layers as a separate model. This is useful for FL training with edge devices performing feature extraction and sending feature data to server for training, e.g. in the application of defect_detection_v1. 
