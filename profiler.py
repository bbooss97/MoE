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
from torch.profiler import profile, record_function, ProfilerActivity
import time


#device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

first=True


sweep_id=""
project_name="moeProfiler"
entity_name="aledevo"

# Load the YAML file as a dictionary
with open("sweep_config_profiler.yaml") as f:
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

    input=torch.randn(1,3,32,32).to(device)

    return v,input

#function to profile the model
def profileModel(model,input):
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(input)
    res=prof.key_averages().table(sort_by="cuda_time_total")
    cpuTimeStr=res.split("\n")[-3][21:]
    cudaTimeStr=res.split("\n")[-2][22:]

    wandb.log({"profiler":res,"cpuTimeTotal":cpuTimeStr,"cudaTimeTotal":cudaTimeStr})

    print(res)
    print("cpu time: "+cpuTimeStr)
    print("cuda time: "+cudaTimeStr)
    return

#main function
def run():
    wandb.init(id=sweep_id,project=project_name,entity=entity_name)
    global first

    v,input=load(100)

    profileModel(v,input)

    #if firt run standard
    if first:
        print("its the first run wait 60 sec")
        time.sleep(60)
        first=False

# Start sweep jobs
wandb.agent(sweep_id, function=run,entity=entity_name,project=project_name)