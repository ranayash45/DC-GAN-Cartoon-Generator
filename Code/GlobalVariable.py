import random
import torch


#Set random seed for reproduciblity
manualSeed = 999

#print("Random Seed: ",manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#Root Directory for dataset
dataroot = "D:\Dataset"

#Number of workers for dataloader
workers = 1

#Batch Size for training
batch_size = 128

#Spatial Size of training images
image_size = 64

#Number of channels in training images
nc = 3

#Size of generator input 
nz = 100

#Size of feature maps in generator
ngf = 64

#Size of feature maps in discriminator
ndf = 64

#Number of Epochs
num_epochs = 4

#Learning Rate for optimizers
lr = 0.002

#Beta1 hyperparam for adam optimizers
beta1 = 0.5

#Number of GPUs available
ngpu = 0