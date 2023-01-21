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
            self.keys=nn.Linear(self.inputDimension, self.hiddenAttentionDimension*self.nOfExperts)
            self.queries=nn.Linear(self.inputDimension, self.hiddenAttentionDimension*self.nOfExperts)
        else:
            self.gate=nn.Linear(self.inputDimension, self.nOfExperts)

    def forward(self, x):
        self.counter+=1
        if self.useAttention:
            keys=self.keys(x).view(x.shape[0],x.shape[1],self.hiddenAttentionDimension,self.nOfExperts)
            queries=self.queries(x).view(x.shape[0],x.shape[1],self.hiddenAttentionDimension,self.nOfExperts)
            queries=torch.einsum("abcd->acbd",queries)
            gateLogits=torch.einsum("abcd,ackd->abkd",keys,queries)
            gateProbabilities=nn.Softmax(dim=-2)(gateLogits)
            gateProbabilities=gateProbabilities.sum(dim=-3)
        else:
            #compute the logits of the gate
            gateLogits=self.gate(x)
        
            #compute the probability of each expert
            gateProbabilities=nn.Softmax(dim=-1)(gateLogits)

        self.balancingLoss=gateProbabilities.sum(dim=-2)
        self.balancingLoss=nn.MSELoss()(self.balancingLoss,torch.ones(self.balancingLoss.shape).to(device)*x.shape[1]/self.nOfExperts).to(device)

        #get the topk
        topKvalues, topKindices=torch.topk(gateProbabilities,self.k,dim=-1)

        # if self.counter%100==0:
        #     print(topKvalues)

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
            self.keys=nn.Linear(self.inputDimension, self.hiddenAttentionDimension*self.nOfExperts)
            self.queries=nn.Linear(self.inputDimension, self.hiddenAttentionDimension*self.nOfExperts)
        else:
            self.gate=nn.Linear(self.inputDimension, self.nOfExperts)

    def forward(self, x):
        self.counter+=1
        #compute the logits of the gate
        if self.useAttention:
            keys=self.keys(x).view(x.shape[0],x.shape[1],self.hiddenAttentionDimension,self.nOfExperts)
            queries=self.queries(x).view(x.shape[0],x.shape[1],self.hiddenAttentionDimension,self.nOfExperts)
            queries=torch.einsum("abcd->acbd",queries)
            gateLogits=torch.einsum("abcd,ackd->abkd",keys,queries)
            gateProbabilities=nn.Softmax(dim=-2)(gateLogits)
            gateProbabilities=gateProbabilities.sum(dim=-3)

        else:
            gateLogits=self.gate(x)
            #compute the probability of each expert
            gateProbabilities=nn.Softmax(dim=-2)(gateLogits)

        #get the topk
        topKvalues, topKindices=torch.topk(gateProbabilities,self.k,dim=-2)

        self.balancingLoss=gateProbabilities.sum(dim=-2)
        self.balancingLoss=nn.MSELoss()(self.balancingLoss,torch.ones(self.balancingLoss.shape).to(device)*x.shape[1]/self.nOfExperts).to(device)

        # if self.counter%100==0:
        #     print(topKvalues)

        outputs=torch.zeros(x.shape[0],x.shape[1],self.outputDimension).to(device)
        #compute the output of each expert
        for i in range(self.nOfExperts):
            batch_indices=torch.arange(x.shape[0]).reshape(-1,1).expand(x.shape[0],self.k).reshape(-1)
            outputs[batch_indices,topKindices[:,:,i].reshape(-1)]+=self.experts[i](x[batch_indices,topKindices[:,:,i].reshape(-1)]).T @ gateProbabilities[batch_indices,topKindices[:,:,i].reshape(-1),i]
            
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


