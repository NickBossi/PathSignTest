# Signatory Library
# Nicholas Bossi
# 25 March 2024
# Testing to ensure that signatory.signature function works on generated random_walk


import torch
import signatory
import numpy as np
import Random_Walk as rw



#Creates a 2D random walk path over 100 time steps with a covariance of 0.2 and ensures the tensor is 3D as signatory requires a batch size (for us, simply 1)
path = torch.unsqueeze(torch.tensor(rw.random_walk(2,100,0.2)),0)

#Prtints path and ensures correct shape
print(path)
print(path.shape)

#Gets the signature of the paths
sig = signatory.signature(path=path, depth = 2)

#Prints the signature
print(sig.shape)