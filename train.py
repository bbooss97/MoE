import torch
from torch.utils.data import DataLoader
from torch import nn
import torchvision
import wandb
from nn import *
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from vit_pytorch.vit import ViT
# from vit_pytorch.vit_for_small_dataset import ViT
import torchvision.transforms as transforms
from torchsummary import summary

#declare parameters
num_epochs=20000
batch_size=64
w_and_b=False
nn_type="vit"

#use additional loss from the transformer layers modified
rl=False
w,h=32,32

#read the dataset cifar10 in this case
datasetTraining=torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(w),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
]))
datasetTest=torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
    transforms.Resize(w),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
]))

#device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


#if wandb is used initialize it
if w_and_b:
    wandb.init(project='moe', entity='bbooss97',name=nn_type)

#dataloader
train_dataloader = DataLoader(datasetTraining, batch_size=batch_size, shuffle=True , drop_last=True )
test_dataloader = DataLoader(datasetTest, batch_size=batch_size, shuffle=False , drop_last=True )


model = ViT(
    image_size=w,
    patch_size=4,
    depth=4,
    heads=8,
    dim=128,
    mlp_dim=128,
    dropout=0.0,
    emb_dropout=0.0,
    num_classes=10
)
    
if w_and_b:
    wandb.watch(model)

model=model.to(device)

#define loss and the optimizer
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)


for epoch in range(num_epochs):
    #train
    model.train()
    if w_and_b:
        wandb.log({"epoch":epoch})

    #train loop
    for i, (images, labels) in enumerate(train_dataloader):

        #move the data to the device
        images=images.to(device)
        labels=labels.to(device)

       
        outputs=model(images)

        #calculate the loss
        l=loss(outputs,labels)
        
        #add the additional loss from the transformer layers modified if rl is true
        if rl:
            rLoss=torch.zeros(1).to(device)
            for ind,layer in enumerate(model.transformer.layers):
                if ind % 1 == 0:
                    rLoss+=layer[1].fn.net.rlLoss
            l+=rLoss.item()
            
            
        #backpropagation
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        print("epoch: {}/{}, step: {}/{}, loss: {}".format(epoch+1,num_epochs,i+1,int(len(train_dataloader)),l.item()))
        
        if w_and_b:
            if i%20==0:
                wandb.log({"loss_train":l.item()})

    #test
    model.eval()
    if w_and_b:
        wandb.watch(model)

    # Initialize variables to store metrics
    l = 0.0
    accuracy = 0.0
    f1 =0.0
    precision =0.0
    recall =0.0

    # Initialize lists to store predictions and labels
    testOutputs=[]
    testLabels=[]

    # Loop over the data in the test set
    with torch.no_grad():
        for i,(images, labels) in enumerate(test_dataloader):

            # Move the data to the device
            images = images.to(device)
            labels = labels.to(device)
           
            outputs=model(images)

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

    if w_and_b:
        wandb.log({"avg_loss_test":avg_loss,"accuracy_test":accuracy,"f1_test":f1,"precision_test":precision,"recall_test":recall})

    #save the model
    torch.save(model,"./"+nn_type+".pt")

if w_and_b:
    wandb.finish()

print("finished")