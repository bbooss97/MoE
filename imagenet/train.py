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
from mlpMixer import MLPMixer
from vit_pytorch.vit import ViT
# from vit_pytorch.vit_for_small_dataset import ViT
import torchvision.transforms as transforms

w,h=32,32
#declare parameters
augment=True
num_epochs=20000
batch_size=32
nOfPatches=10
w_and_b=False 
nn_type="vit"

rl=False


#read the dataset
# datasetTraining=ImagenDataset(w,h,False,augment=augment)
# datasetTest=ImagenDataset(w,h,True,augment=False)

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
# device= torch.device("cpu")
print(device)


if w_and_b:
    wandb.init(project='moe', entity='bbooss97',name=nn_type)

#dataloader
train_dataloader = DataLoader(datasetTraining, batch_size=batch_size, shuffle=True , drop_last=True )
test_dataloader = DataLoader(datasetTest, batch_size=batch_size, shuffle=False , drop_last=True )

if nn_type=="mlp":
    model=Mlp(w,h)
elif nn_type=="moe":
    model=MoE(w,h,1,20,nOfPatches,useTokenBasedApproach=True,useAttention=False)
elif nn_type=="mlp_patches":
    model=MlpPatches(w,h,nOfPatches)
elif nn_type=="resnetFrom0":
    model=torch.hub.load('pytorch/vision:v0.6.0', 'resnet18',pretrained=False)
    model.fc=torch.nn.Linear(512,10)
elif nn_type=="resnetPretrainedFineTuneAll":
    model=torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    model.fc=torch.nn.Linear(512,10)
elif nn_type=="resnetPretrainedFineTuneFc":
    model=torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    model.fc=torch.nn.Linear(512,10)
    toFreeze=[j for i,j in model.named_parameters()][:-2]
    for i in toFreeze:
        i.requires_grad=False
elif nn_type=="vit":
    model = ViT(
        image_size=w,
        patch_size=4,
        depth=4,
        heads=8,
        dim=128,
        mlp_dim=128,
        dropout=0.1,
        emb_dropout=0.1,
        num_classes=10
    )
elif nn_type=="moeTransformerFc":
    model=moeTransformerFc()
elif nn_type=="moeStack":
    model=moeStack()
elif nn_type=="mixerMoe":
    model = MLPMixer(in_channels=3, image_size=w, patch_size=4, num_classes=10,
                     dim=128, depth=6, token_dim=128, channel_dim=128)
elif nn_type=="moeConvolution":
    model=MoeConvolution(w,h,5,128,nOfPatches,useTokenBasedApproach=True,useAttention=False)
elif nn_type=="moeCombination":
    model=MoeCombination(w,h,5,32,5,nOfPatches,useTokenBasedApproach=True,useAttention=True)
elif nn_type=="moeMix":
    model=MoeMix(w,h,4,128,nOfPatches,useTokenBasedApproach=True,useAttention=False)
elif nn_type=="moeProbabilities":
    model=MoeProbabilities(w,h,5,64,nOfPatches,useTokenBasedApproach=True,useAttention=False)
elif nn_type=="moeRl":
    model=MoeRl(w,h,1,64,nOfPatches,useTokenBasedApproach=False,useAttention=False)
elif nn_type=="moeTransformer":
    model=MoeTransformer(w,h,2,50,nOfPatches,useTokenBasedApproach=True,useAttention=False)
elif nn_type=="moeWide":
    model=MoeWide(w,h,3,1000,nOfPatches,useTokenBasedApproach=True,useAttention=False)
    
if w_and_b:
    wandb.watch(model)

model=model.to(device)

#define loss and the optimizer
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)


for epoch in range(num_epochs):

    #train
    model.train()
    if w_and_b:
        wandb.log({"epoch":epoch})

    #normal loop for training
    #train loop
    for i, (images, labels) in enumerate(train_dataloader):

        #move the data to the device
        images=images.to(device)
        labels=labels.to(device)

        if nn_type=="moe" or nn_type=="mlp_patches" or nn_type=="moeTransformerFc" or nn_type=="moeStack" or nn_type=="moeConvolution" or nn_type=="moeCombination" or nn_type=="moeMix" or nn_type=="moeProbabilities" or nn_type=="moeRl" :
            #get the patches
            # images=images/255
            # images=torch.einsum("abcd->adbc",images)
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
        elif nn_type =="resnetPretrainedFineTuneAll":
            images=images/255
            images=torch.einsum("abcd->adbc",images)
            outputs=model(images)
        elif nn_type =="vit":
            # images=images/255
            # images=torch.einsum("abcd->adbc",images)
            outputs=model(images)
        elif nn_type =="mixerMoe":
            # images=images/255
            # images=torch.einsum("abcd->adbc",images)
            outputs=model(images)
        elif nn_type=="moeTransformer":
            # images=images/255
            # images=torch.einsum("abcd->adbc",images)
            outputs=model(images)
        elif nn_type=="moeWide":
            # images=images/255
            # images=torch.einsum("abcd->adbc",images)
            outputs=model(images)
       
    

        
        #calculate the loss
        l=loss(outputs,labels)
        if nn_type=="vit" and rl:
            rLoss=torch.zeros(1).to(device)
            for ind,layer in enumerate(model.transformer.layers):
                if ind % 1 == 0:
                    rLoss+=layer[1].fn.net.rlLoss
            l+=rLoss.item()
            
            
        #backpropagation
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        l.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
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

            if nn_type=="moe" or nn_type=="mlp_patches" or nn_type=="moeTransformerFc" or nn_type=="moeStack" or nn_type=="moeConvolution"or nn_type=="moeCombination" or nn_type=="moeMix" or nn_type=="moeProbabilities" or nn_type=="moeRl" :   
                #get the patches
                # images=images/255
                # images=torch.einsum("abcd->adbc",images)
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
                # images=images/255
                # images=torch.einsum("abcd->adbc",images)
                outputs=model(images)
            elif nn_type =="mixerMoe":
                # images=images/255
                # images=torch.einsum("abcd->adbc",images)
                outputs=model(images)
            elif nn_type=="moeTransformer":
                # images=images/255
                # images=torch.einsum("abcd->adbc",images)
                outputs=model(images)
            elif nn_type=="moeWide":
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
    
    # # Compute accuracy
    # accuracy = accuracy_score(testLabels.argmax(dim=1), testOutputs.argmax(dim=1))

    # # Compute f1 score
    # f1 = f1_score(testLabels.argmax(dim=1), testOutputs.argmax(dim=1), average='macro')

    # # Compute precision
    # precision = precision_score(testLabels.argmax(dim=1), testOutputs.argmax(dim=1), average='macro')

    # # Compute recall
    # recall = recall_score(testLabels.argmax(dim=1), testOutputs.argmax(dim=1), average='macro')

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