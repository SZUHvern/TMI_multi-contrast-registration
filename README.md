# A coarse-to-fine deformable transformation framework for unsupervised multi-contrast MR image registration with dual consistency constraint
The code of TMI paper 'A coarse-to-fine deformable transformation framework for unsupervised multi-contrast MR image registration with dual consistency constraint'

# Training steps:
Two steps for training are needed:
1. Perform affine_train.py to train the AT-Net in advance
2. Then perform registration.py with loding the weight of AT-Net

# Environment
Python >= 3.6

CUDA == 11.0

Tensorflow-gpu == 1.10

Keras == 2.2.0
