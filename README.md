# TMI_multi-contrast-registration
A coarse-to-fine deformable transformation framework for unsupervised multi-contrast MR image registration with dual consistency constraint

Two step for training are needed:
1. Perform affine_train.py to train the AT-Net in advance
2. Then perform registration.py with loding the weight of AT-Net
