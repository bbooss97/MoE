import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import torchvision
from torchvision.models import resnet50
import wandb
from nn import *
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from vit_pytorch.distill import DistillableViT,DistillWrapper
import torchvision.transforms as transforms
import yaml
from vit_pytorch.vit import ViT


#device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


sweep_id=""
project_name="moeProfiler"
entity_name="aledevo"

# Load the YAML file as a dictionary
with open("sweep_config.yaml") as f:
  sweep_configuration = yaml.safe_load(f)

if sweep_id=="":
    #initialize wandb
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name,entity=entity_name)


#define the models and the optimizer
def load(num_classes=10):
    dim=wandb.config.dim
    depth=wandb.config.depth
    heads=wandb.config.heads
    mlp_dim=wandb.config.mlp_dim
    dropout=wandb.config.dropout

    #load the student model
    v = ViT(
        image_size = 32,
        patch_size = 4,
        num_classes = num_classes,
        dim = dim,
        depth = depth,
        heads = heads,
        mlp_dim = mlp_dim,
        dropout = dropout,
        emb_dropout = dropout
    )

    #put models on the device
    v=v.to(device)

    return v
#function to profile the model
def profile(model):
    pass

#main function
def run():
    wandb.init(id=sweep_id,project=project_name,entity=entity_name)

    v=load(100)

    profile(v)


# Start sweep jobs
wandb.agent(sweep_id, function=run,entity=entity_name,project=project_name)