import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Expert(nn.Module):
    def __init__(self,input,output):
        self.input=input
        self.output=output
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input,output)
        self.fc2 = nn.Linear(output, output)
        self.fc3 = nn.Linear(output, output)


    def forward(self, x):
        x = self.fc1(x)
        x=nn.ReLU()(x)

        x = self.fc2(x)
        x=nn.ReLU()(x)

        x=self.fc3(x)
        x=nn.ReLU()(x)

        return x

#self attention module to otain the attention map from the patches
class SelfAttention(torch.nn.Module):
    def __init__(self, inputDimension,hiddenDimension,nOfExperts):
        super(SelfAttention, self).__init__()
        self.hiddenDimension = hiddenDimension
        #query and key linear layers
        self.q = torch.nn.Linear(inputDimension, hiddenDimension*nOfExperts)
        self.k = torch.nn.Linear(inputDimension, hiddenDimension*nOfExperts)
        self.inputDimension = inputDimension
        self.nOfExperts=nOfExperts

    def forward(self, input):
        #input=input.double()
        #get query and keys
        #shape num, number of patches, qDimension or kDimension
        q=self.q(input).view(input.shape[0],input.shape[1],self.hiddenDimension,self.nOfExperts)
        k=self.k(input).view(input.shape[0],input.shape[1],self.nOfExperts,self.hiddenDimension)
        #batched matrix multiplication
        #shape num , number of patches, number of patches in this case 225
        #attention=torch.einsum('bij,bjk->bik', q,k)
        attention=torch.einsum("bijk,bilj->bikl", q,k)
        #scaling factor sqrt of dimension of key vector like in normal self attention
        attention=attention/((input.shape[2])**0.5)
        #softmax along the last dimension
        #shape num , number of patches, number of patches in this case 225
        attention=torch.softmax(attention,dim=-1)
        attention=attention.sum(dim=-2)
        return attention
        
class Mlp(nn.Module):
    def __init__(self,w,h):
        self.w=w
        self.h=h
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(self.w*self.h*3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc41 = nn.Linear(128, 128)
        self.fc42 = nn.Linear(128, 128)
        self.fc43 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 18)

    def forward(self, x):
        x = self.fc1(x)
        x=nn.ReLU()(x)

        x = self.fc2(x)
        x=nn.ReLU()(x)

        x=self.fc3(x)
        x=nn.ReLU()(x)

        x=self.fc4(x)
        x=nn.ReLU()(x)

        
        x=self.fc41(x)
        x=nn.ReLU()(x)

        
        x=self.fc42(x)
        x=nn.ReLU()(x)

        
        x=self.fc43(x)
        x=nn.ReLU()(x)

        x=self.fc5(x)
        x=nn.ReLU()(x)

        x=self.fc6(x)
        
        #do not need to use softmax or sigmoid because we use cross entropy loss and it does it for us
        return x


class MoeFc(nn.Module):
    def __init__(self,inputDimension, outputDimension,nOfExperts,k,useAttention=False):
        super(MoeFc, self).__init__()
        self.inputDimension=inputDimension
        self.outputDimension=outputDimension
        self.nOfExperts=nOfExperts
        self.k=k
        self.counter=0
        self.experts=nn.ModuleList([Expert(self.inputDimension,self.outputDimension) for i in range(self.nOfExperts)])
        self.useAttention=useAttention
        self.hiddenAttentionDimension=3
        if self.useAttention:
            self.selfAttention=SelfAttention(self.inputDimension,self.hiddenAttentionDimension,self.nOfExperts)
        else:
            self.gate=nn.Linear(self.inputDimension, self.nOfExperts)

    def forward(self, x):
        self.counter+=1
        if self.useAttention:
            gateProbabilities=self.selfAttention(x)
        else:
            #compute the logits of the gate
            gateLogits=self.gate(x)
        
            #compute the probability of each expert
            gateProbabilities=nn.Softmax(dim=-1)(gateLogits)

        self.balancingLoss=gateProbabilities.sum(dim=-2)
        self.balancingLoss=nn.MSELoss()(self.balancingLoss,torch.ones(self.balancingLoss.shape).to(device)*x.shape[1]/self.nOfExperts).to(device)

        #get the topk
        topKvalues, topKindices=torch.topk(gateProbabilities,self.k,dim=-1)

        outputs=torch.zeros(x.shape[0],x.shape[1],self.outputDimension).to(device)
        
        #compute the output of each expert
        for i in range(self.nOfExperts):
            x_e=(topKindices==i).nonzero()
            outputs[x_e[:,0],x_e[:,1]]+=(self.experts[i](x[x_e[:,0],x_e[:,1]]).T  * gateProbabilities[x_e[:,0],x_e[:,1],x_e[:,2]]).T

        #outputs=nn.Softmax(dim=-1)(outputs)
        return outputs

class MoeFcTokens(nn.Module):
    def __init__(self,inputDimension, outputDimension,nOfExperts,k,useAttention=False):
        super(MoeFcTokens, self).__init__()
        self.inputDimension=inputDimension
        self.outputDimension=outputDimension
        self.nOfExperts=nOfExperts
        self.k=k
        self.counter=0
        self.useAttention=useAttention
        self.experts=nn.ModuleList([Expert(self.inputDimension,self.outputDimension) for i in range(self.nOfExperts)])
        self.hiddenAttentionDimension=3

        if self.useAttention:
            self.selfAttention=SelfAttention(self.inputDimension,self.hiddenAttentionDimension,self.nOfExperts)

        else:
            self.gate=nn.Linear(self.inputDimension, self.nOfExperts)

    def forward(self, x):
        self.counter+=1
        #compute the logits of the gate
        if self.useAttention:
            gateProbabilities=self.selfAttention(x)
        else:
            gateLogits=self.gate(x)
            #compute the probability of each expert
            gateProbabilities=nn.Softmax(dim=-2)(gateLogits)

        #get the topk
        topKvalues, topKindices=torch.topk(gateProbabilities,self.k,dim=-2)

        self.balancingLoss=gateProbabilities.sum(dim=-2)
        self.balancingLoss=nn.MSELoss()(self.balancingLoss,torch.ones(self.balancingLoss.shape).to(device)*x.shape[1]/self.nOfExperts).to(device)

        outputs=torch.zeros(x.shape[0],x.shape[1],self.outputDimension).to(device)
        #compute the output of each expert
        for i in range(self.nOfExperts):
            batch_indices=torch.arange(x.shape[0]).reshape(-1,1).expand(x.shape[0],self.k).reshape(-1)
            outputs[batch_indices,topKindices[:,:,i].reshape(-1)]+=(self.experts[i](x[batch_indices,topKindices[:,:,i].reshape(-1)]).T * gateProbabilities[batch_indices,topKindices[:,:,i].reshape(-1),i]).T
            
        return outputs

class MoE(nn.Module):
    def __init__(self,w,h,k,nOfExperts,nOfPatches,useTokenBasedApproach=False,useAttention=False):
        super(MoE, self).__init__()
        self.w=w
        self.h=h
        self.nOfPatches=nOfPatches
        self.k=k
        self.useAttention=useAttention
        self.nOfExperts=nOfExperts
        self.tokenSize=int(3*(self.w/self.nOfPatches)*(self.h/self.nOfPatches))

        if useTokenBasedApproach:
            self.moefc=MoeFcTokens(self.tokenSize,128,self.nOfExperts,self.k,useAttention=self.useAttention)
        else:
            self.moefc=MoeFc(self.tokenSize,128,self.nOfExperts,self.k,useAttention=self.useAttention)

        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
       
        self.fc4= nn.Linear(128*self.nOfPatches**2,128)

        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 18)


    def forward(self, x):

        x=self.moefc(x)
        x=nn.ReLU()(x)

        x = self.fc2(x)
        x=nn.ReLU()(x)

        x=self.fc3(x)
        x=nn.ReLU()(x)

        x=x.view(x.shape[0],-1)

        x=self.fc4(x)
        x=nn.ReLU()(x)


        x=self.fc5(x)
        x=nn.ReLU()(x)

        x=self.fc6(x)

        return x


