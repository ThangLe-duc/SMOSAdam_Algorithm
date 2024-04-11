# SMOSAdam_Algorithm
We open-source a Matlab-based package for training neural networks by our proposed method called Sequential Motion Optimization with Short-term Adaptive Moment Estimation (SMO-SAdam). MNIST classification problem is chosen to demo in this package. The optimization function "smosadamupdate.m" provided in the folder “SMOSAdam_Package” has a similar format with the built-in "adamupdate" function, which helps the users to apply easily. Some detailed definitions and instructions are also presented into the codes to help the users easily modify it for personal research and uses. It is suggested that the users should read thoroughly our paper presented in the reference below before using the codes. We show below some necessary information of the SMO-SAdam package including
1. Structure of SMO-SAdam software package: • SMOSAdam_MNIST_Main.m: includes the main codes for running SMO-SAdam to solve MNIST problem. • Folder “SMOSAdam_Package”: includes the SMO-SAdam functions to optimize DNN parameters. • Folder “Datasets”: includes the problem datasets.
2. How to solve DNN training problems by SMO-SAdam software package: • Put the problem dataset into the “Datasets” folder and preprocess the problem in the main function • Specify the network structures in the function "Leader_Follower_Nets.m", general learning hyperparameters and SMO-SAdam settings in the main function and the relating ones in “SMOSAdam_Package” • Run main function and wait until the training process stops. • The network parameters and statistical results are recorded in the DNN.mat and results.mat, respectively.

# Programmer
Thang Le-Duc, Deep Learning Architecture Research Center, Sejong University, email: le.duc.thang0312@gmail.com

# Reference
Le-Duc, Thang, H. Nguyen-Xuan, and Jaehong Lee. "Sequential motion optimization with short-term adaptive moment estimation for deep learning problems." Engineering Applications of Artificial Intelligence 129 (2024): 107593.
https://www.sciencedirect.com/science/article/pii/S0952197623017773
