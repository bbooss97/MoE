import torch
from torch.utils.data import DataLoader
from torch import nn
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

#device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


sweep_id=""
project_name="moeCifar100"
entity_name="aledevo"

# Load the YAML file as a dictionary
with open("sweep_config.yaml") as f:
  sweep_configuration = yaml.safe_load(f)

if sweep_id=="":
    #initialize wandb
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name,entity=entity_name)


#load the cifar100 dataset with 3 randaug
def loadDatasetCifar100():
    #define dataset
    datasetTraining=torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.RandAugment(num_ops=2, magnitude=14),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]))
    datasetTest=torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]))

    batch_size=wandb.config.batch_size

    #define dataloader
    train_dataloader = DataLoader(datasetTraining, batch_size=batch_size, shuffle=True , drop_last=True )
    test_dataloader = DataLoader(datasetTest, batch_size=batch_size, shuffle=False , drop_last=True )
    
    return train_dataloader, test_dataloader , 100

def loadDatasetCifar10():
    #define dataset
    datasetTraining=torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.RandAugment(num_ops=3, magnitude=6),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))
    datasetTest=torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))

    batch_size=wandb.config.batch_size

    #define dataloader
    train_dataloader = DataLoader(datasetTraining, batch_size=batch_size, shuffle=True , drop_last=True )
    test_dataloader = DataLoader(datasetTest, batch_size=batch_size, shuffle=False , drop_last=True )

    return train_dataloader, test_dataloader , 10

#define the models and the optimizer
def load(num_classes=10):
    dim=wandb.config.dim
    depth=wandb.config.depth
    heads=wandb.config.heads
    mlp_dim=wandb.config.mlp_dim
    lr=wandb.config.lr

    #load the teacher model
    teacher = resnet50(pretrained = True )
    #modify here to change the number of classes
    teacher.fc = nn.Linear(2048, num_classes)

    #load the student model
    v = DistillableViT(
        image_size = 32,
        patch_size = 4,
        num_classes = num_classes,
        dim = dim,
        depth = depth,
        heads = heads,
        mlp_dim = mlp_dim,
        dropout = 0.0,
        emb_dropout = 0.0
    )

    distiller = DistillWrapper(
        student = v,
        teacher = teacher,
        temperature = 3,           # temperature of distillation
        alpha = 0.5,               # trade between main loss and distillation loss
        hard = False               # whether to use soft or hard distillation
    )


    #put models on the device
    v=v.to(device)
    distiller=distiller.to(device)
    
    #define loss and the optimizer
    loss=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(v.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)

    return v, distiller, loss, optimizer , scheduler

def trainLoop(epoch, num_epochs, train_dataloader, v, distiller, optimizer, scheduler):
    #train
    v.train()
    wandb.log({"epoch":epoch})

    # #train loop
    for i, (images, labels) in enumerate(train_dataloader):

        #move the data to the device
        images=images.to(device)
        labels=labels.to(device)

        #loss function
        l=distiller(images,labels)
            
        #backpropagation
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        scheduler.step()

        #log
        if i%20==0:
            print("epoch: {}/{}, step: {}/{}, loss: {}".format(epoch+1,num_epochs,i+1,int(len(train_dataloader)),l.item()))
            wandb.log({"loss_train":l.item()})

def testLoop(epoch, num_epochs, test_dataloader, v, loss, optimizer):
    #test
    v.eval()

    # Initialize variables to store metrics
    l = 0.0
    accuracy = 0.0
    f1 =0.0
    precision = 0.0
    recall = 0.0

    # Initialize lists to store predictions and labels
    testOutputs=[]
    testLabels=[]

    # Loop over the data in the test set
    with torch.no_grad():
        for i,(images, labels) in enumerate(test_dataloader):

            # Move the data to the device
            images = images.to(device)
            labels = labels.to(device)
           
            outputs=v(images)

            # Store predictions and labels
            testOutputs.append(outputs)
            testLabels.append(labels)

            ls = loss(outputs, labels)

            # Compute running metrics
            l += ls.item()
        

    # Concatenate predictions and labels
    testOutputs = torch.cat(testOutputs, dim=0)
    testLabels = torch.cat(testLabels, dim=0)

    testLabels=testLabels.cpu()
    testOutputs=testOutputs.cpu()

    # Compute average loss
    avg_loss = l / len(test_dataloader)

    # Compute accuracy
    accuracy = accuracy_score(testLabels, testOutputs.argmax(dim=1))

    # Compute f1 score
    f1 = f1_score(testLabels, testOutputs.argmax(dim=1), average='macro')

    # Compute precision
    precision = precision_score(testLabels, testOutputs.argmax(dim=1), average='macro')

    # Compute recall
    recall = recall_score(testLabels, testOutputs.argmax(dim=1), average='macro')

    # Print the metrics
    print(f'Test loss: {avg_loss:.4f}')
    print(f'Test accuracy: {accuracy:.4f}')
    print(f'Test f1: {f1:.4f}')
    print(f'Test precision: {precision:.4f}')
    print(f'Test recall: {recall:.4f}')

    
    wandb.log({"avg_loss_test":avg_loss,"accuracy_test":accuracy,"f1_test":f1,"precision_test":precision,"recall_test":recall})

    #save the model with the id from wandb
    # torch.save(v,"./"+ "vit" +".pt")

    return accuracy

#main function
def run():
    wandb.init(id=sweep_id,project=project_name,entity=entity_name)

    #change the commented lines to change the dataset
    train_dataloader,test_dataloader,num_classes =loadDatasetCifar100()
    # train_dataloader,test_dataloader,num_classes =loadDatasetCifar10()

    v, distiller, loss, optimizer , scheduler =load(num_classes)

    num_epochs=wandb.config.num_epochs
    
    #simple early stopping
    topAccuracy=0
    epochsWithoutImprovements=0
    stopIfNoImprovementFor=5

    for epoch in range(num_epochs):
        trainLoop(epoch, num_epochs, train_dataloader, v, distiller, optimizer , scheduler)

        if epoch%3==0:
            accuracy=testLoop(epoch, num_epochs, test_dataloader, v, loss, optimizer)
            
            #if there are no improvements stop the training
            if accuracy>topAccuracy:
                topAccuracy=accuracy
                epochsWithoutImprovements=0
            else:
                epochsWithoutImprovements+=1
            if epochsWithoutImprovements>=stopIfNoImprovementFor:
                return


# Start sweep jobs
wandb.agent(sweep_id, function=run,entity=entity_name,project=project_name)