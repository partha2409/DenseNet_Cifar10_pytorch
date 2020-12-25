# DenseNet_Cifar10_pytorch

This repository is a Pytorch Implementation of DenseNet https://arxiv.org/abs/1608.06993 for classification of Cifar-10 images.

Requriements: Pytorch, Matplotlib, Pickle

Dataset: Download and extract the CIFAR-10 dataset (python version) to the project folder from https://www.cs.toronto.edu/~kriz/cifar.html

To Train the model:

Update the dataset path 'train_data_path' in config.py
Run Train.py

To run inference:

Update 'model_dir and 'model_path' of the trained model in 'run_inference.py'
Run run_inference.py
