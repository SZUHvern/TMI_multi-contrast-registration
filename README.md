# A coarse-to-fine deformable transformation framework for unsupervised multi-contrast MR image registration with dual consistency constraint
This is the code of TMI's paper 'A coarse-to-fine deformable transformation framework for unsupervised multi-contrast MR image registration with dual consistency constraint'.

![image](https://github.com/SZUHvern/TMI_multi-contrast-registration/blob/main/Framework.jpg)

# Abstract:
Multi-contrast magnetic resonance (MR) image registration is useful in the clinic to achieve fast and accurate imaging-based disease diagnosis and treatment planning. Nevertheless, the efficiency and performance of the existing registration algorithms can still be improved. In this paper, we propose a novel unsupervised learning-based framework to achieve accurate and efficient multi-contrast MR image registrations. Specifically, an end-to-end coarse-to-fine network architecture consisting of affine and deformable transformations is designed to improve the robustness and achieve end-to-end registration. Furthermore, a dual consistency constraint and a new prior knowledge-based loss function are developed to enhance the registration performances. The proposed method has been evaluated on a clinical dataset containing 555 cases, and encouraging performances have been achieved. Compared to the commonly utilized registration methods, including VoxelMorph, SyN, and LT-Net, the proposed method achieves better registration performance with a Dice score of 0.8397 in identifying stroke lesions. With regards to the registration speed, our method is about 10 times faster than the most competitive method of SyN (Affine) when testing on a CPU. Moreover, we prove that our method can still perform well on more challenging tasks with lacking scanning information data, showing the high robustness for the clinical application.

# Training:
Two steps for training are needed:
1. Perform affine_train.py to train the ATNet in advance.
2. Then perform registration.py with the weight of ATNet.

# Environmental dependences
Python >= 3.6

CUDA == 11.0

Tensorflow-gpu == 1.10

Keras == 2.2.0

# Cite this:
If you use this framework or some part of the code, please cite:

W. Huang et al., "A coarse-to-fine deformable transformation framework for unsupervised multi-contrast MR image registration with dual consistency constraint," in IEEE Transactions on Medical Imaging, doi: 10.1109/TMI.2021.3059282.
