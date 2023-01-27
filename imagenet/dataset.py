import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
from classes import *

#set the seeds to make everything reproducible
torch.manual_seed(0)
np.random.seed(0)

#dataset class
class ImagenDataset(Dataset):
    def __init__(self,w,h,isVal):
        self.w=w
        self.h=h
        self.isVal=str(isVal)
        self.counter=0
        self.items=[]
        with open("./imagenet/imagenette2-320/noisy_imagenette.csv", "r") as f:
            header=f.readline().split(",")
            for line in f.readlines():
                line=line.strip().split(",")
                if line[-1]==self.isVal:
                    self.counter+=1
                    self.items.append([line[0],int(classes[line[1]])])
        
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        itemPath="./imagenet/imagenette2-320/"+self.items[idx][0].strip()
        itemClass=self.items[idx][1]

        #one hot encoding of the class of the item
        label=torch.zeros(len(classes))
        label[itemClass]=1

        #load the image as pil image
        image=Image.open(itemPath).convert('RGB')
        image = image.resize((self.w, self.h)) 

        #convert it to a tensor
        image=torch.tensor(np.array(image),dtype=torch.float32)
        
        return image, label ,itemPath