import torch
from torch.utils.data import DataLoader
from torch import nn
import torchvision
import wandb
from dataset import ImagenDataset
from nn import *
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

w,h=160,160

#read the dataset
datasetTraining=ImagenDataset(w,h,False)
datasetTest=ImagenDataset(w,h,True)

#device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device= torch.device("cpu")
print(device)

#declare parameters
num_epochs=200
batch_size=32
nOfPatches=10
w_and_b=False
nn_type="moe"

balanceTheLoss=False

if w_and_b:
    wandb.init(project='moe', entity='bbooss97',name=nn_type)

#dataloader
train_dataloader = DataLoader(datasetTraining, batch_size=batch_size, shuffle=True , drop_last=True)
test_dataloader = DataLoader(datasetTest, batch_size=batch_size, shuffle=True , drop_last=True)

if nn_type=="mlp":
    model=Mlp(w,h)
elif nn_type=="moe":
    model=MoE(w,h,3,30,nOfPatches,useTokenBasedApproach=True,useAttention=False)
elif nn_type=="mlp_patches":
    model=MlpPatches(w,h,nOfPatches)
elif nn_type=="resnetFrom0":
    model=torch.hub.load('pytorch/vision:v0.6.0', 'resnet18',pretrained=False)
    model.fc=torch.nn.Linear(512,10)
elif nn_type=="resnetPretrainedFineTuneFc":
    model=torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    model.fc=torch.nn.Linear(512,10)
    toFreeze=[j for i,j in model.named_parameters()][:-2]
    for i in toFreeze:
        i.requires_grad=False
elif nn_type=="vit":
    model = torchvision.models.VisionTransformer(
        image_size=160,
        patch_size=32,
        num_layers=1,
        num_heads=1,
        hidden_dim=32,
        mlp_dim=32
    )
    model.heads=torch.nn.Linear(32,10)
elif nn_type=="moeTransformerFc":
    model=moeTransformerFc()
elif nn_type=="moeStack":
    model=moeStack()

    

if w_and_b:
    wandb.watch(model)

model=model.to(device)

#define loss and the optimizer
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters())


for epoch in range(num_epochs):

    #train
    model.train()
    if w_and_b:
        wandb.log({"epoch":epoch})

    #normal loop for training
    #train loop
    for i, (images, labels,paths) in enumerate(train_dataloader):

        #move the data to the device
        images=images.to(device)
        labels=labels.to(device)

        if nn_type=="moe" or nn_type=="mlp_patches" or nn_type=="moeTransformerFc" or nn_type=="moeStack":
            #get the patches
            images=images/255
            images=torch.einsum("abcd->adbc",images)
            size=int(images.shape[2]/nOfPatches)
            unfold=torch.nn.Unfold(kernel_size=(size,size),stride=size)
            patches=unfold(images)
            patches=patches.transpose(1,2)
            patches=patches.to(device)
            outputs=model(patches)
            #balancingLoss=model.moefc.balancingLoss
        elif nn_type=="mlp":
            images=images/255
            images=images.view(images.shape[0],-1)
            outputs=model(images)
        elif nn_type =="resnetFrom0":
            images=images/255
            images=torch.einsum("abcd->adbc",images)
            outputs=model(images)
        elif nn_type =="resnetPretrainedFineTuneFc":
            images=images/255
            images=torch.einsum("abcd->adbc",images)
            outputs=model(images)
        elif nn_type =="vit":
            images=images/255
            images=torch.einsum("abcd->adbc",images)
            outputs=model(images)
 
       
    

        
        #calculate the loss
        l=loss(outputs,labels)
        if nn_type=="moe" and balanceTheLoss:
            l=l+balancingLoss/(100)
            print("epoch: {}   ,loss: {}   ,  balancingLoss: {} ".format(epoch+1,l.item()-balancingLoss.item()/100,balancingLoss.item()/100))
            
            
        
        #backpropagation
        optimizer.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        if not balanceTheLoss:
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
        for i,(images, labels,paths) in enumerate(test_dataloader):

            # Move the data to the device
            images = images.to(device)
            labels = labels.to(device)

            if nn_type=="moe":
                #get the patches
                images=torch.einsum("abcd->adbc",images)
                size=int(images.shape[2]/nOfPatches)
                unfold=torch.nn.Unfold(kernel_size=(size,size),stride=size)
                patches=unfold(images)
                patches=patches.transpose(1,2)
                patches=patches.to(device)
                outputs=model(patches)
            elif nn_type=="mlp":
                images=images.view(images.shape[0],-1)
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
    accuracy = accuracy_score(testLabels.argmax(dim=1), testOutputs.argmax(dim=1))

    # Compute f1 score
    f1 = f1_score(testLabels.argmax(dim=1), testOutputs.argmax(dim=1), average='macro')

    # Compute precision
    precision = precision_score(testLabels.argmax(dim=1), testOutputs.argmax(dim=1), average='macro')

    # Compute recall
    recall = recall_score(testLabels.argmax(dim=1), testOutputs.argmax(dim=1), average='macro')

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