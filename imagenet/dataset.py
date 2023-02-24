import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
from classes import *
from torchvision import transforms
from torchvision.transforms import RandAugment

#set the seeds to make everything reproducible
torch.manual_seed(0)
np.random.seed(0)

#dataset class
class ImagenDataset(Dataset):
    def __init__(self,w,h,isVal,augment=False):
        self.w=w
        self.h=h
        self.isVal=str(isVal)
        self.counter=0
        self.augment=augment
        self.items=[]
        self.transform = transforms.Compose([
            # transforms.RandomResizedCrop(size=160),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(180),
            RandAugment(num_ops=2, magnitude=9)  # Apply 2 operations with magnitude up to 9
            ])
        self.baseTransform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
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
        # label=torch.zeros(len(classes))
        # label[itemClass]=1
        label=torch.tensor(itemClass,dtype=torch.long)

        #load the image as pil image
        image=Image.open(itemPath).convert('RGB')
        image = image.resize((self.w, self.h)) 

        #augment
        if self.augment:
            image=self.transform(image)
        image=self.baseTransform(image)
        # if not self.augment:
        # #convert it to a tensor
        #     image=torch.tensor(np.array(image),dtype=torch.float32)

        return image, label 