import torch
import torch.nn as nn



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device= torch.device("cpu")

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

#self attention module to otain the attention map from the patches my idea
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
        #get query and keys
        q=self.q(input).view(input.shape[0],input.shape[1],self.hiddenDimension,self.nOfExperts)
        k=self.k(input).view(input.shape[0],input.shape[1],self.hiddenDimension,self.nOfExperts)

        k=torch.permute(k,(0,2,1,3))

        #batched matrix multiplication
        attention=torch.einsum("bijl,bjkl->bikl", q,k)
        #scaling factor sqrt of dimension of key vector like in normal self attention
        attention=attention/(self.hiddenDimension**0.5)
        #softmax along the last dimension
        attention=torch.softmax(attention,dim=-2)
        attention=attention.sum(dim=-3)

        return attention

class Attention(torch.nn.Module):
    def __init__(self, inputDimension,hiddenDimension,nOfExperts):
        super(Attention, self).__init__()
        self.hiddenDimension = hiddenDimension
        #query and key linear layers
        self.q = torch.nn.Linear(inputDimension, hiddenDimension*nOfExperts)
        self.k = torch.nn.Linear(inputDimension, hiddenDimension*nOfExperts)
        self.v= torch.nn.Linear(inputDimension, nOfExperts)
        self.toFinal= torch.nn.Linear(hiddenDimension, 1)
        self.inputDimension = inputDimension
        self.nOfExperts=nOfExperts

    def forward(self, input):
        #get query and keys
        q=self.q(input).view(input.shape[0],input.shape[1],self.hiddenDimension,self.nOfExperts)
        k=self.k(input).view(input.shape[0],input.shape[1],self.hiddenDimension,self.nOfExperts)
        v=self.v(input).view(input.shape[0],input.shape[1],self.nOfExperts)

        k=torch.permute(k,(0,2,1,3))

        #batched matrix multiplication
        attention=torch.einsum("bijl,bjkl->bikl", q,k)
        #scaling factor sqrt of dimension of key vector like in normal self attention
        attention=attention/(self.hiddenDimension**0.5)
        #softmax along the last dimension
        attention=torch.softmax(attention,dim=-2)
        #multiply attention with values
        attention=torch.einsum("bikl,bkl->bil", attention,v)


        return attention

class MoeFcTokensParallel(nn.Module):
    def __init__(self,inputDimension, outputDimension,nOfExperts,k,useAttention=False):
        super(MoeFcTokensParallel, self).__init__()
        self.inputDimension=inputDimension
        self.outputDimension=outputDimension
        self.nOfExperts=nOfExperts
        self.k=k
        self.counter=0
        self.useAttention=useAttention
        self.experts=nn.ModuleList([Expert(self.inputDimension,self.outputDimension) for i in range(self.nOfExperts)])
        self.hiddenAttentionDimension=3
        #self.w = torch.ones(self.nOfExperts,self.inputDimension,self.outputDimension)
        
        self.w = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(self.nOfExperts,self.inputDimension,self.outputDimension)),requires_grad=True).to(device)
        self.b = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(self.nOfExperts,1)),requires_grad=True).to(device)


        if self.useAttention:
            self.selfAttention=SelfAttention(self.inputDimension,self.hiddenAttentionDimension,self.nOfExperts)
            #self.selfAttention=Attention(self.inputDimension,self.hiddenAttentionDimension,self.nOfExperts)
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

        outputs=torch.zeros(x.shape[0],x.shape[1],self.outputDimension).to(device)

        i=torch.ones_like(topKindices).nonzero()
        inp=x[i[:,0],topKindices[i[:,0],i[:,1],i[:,2]]]
        exp=self.w[i[:,2],:,:]
        b=self.b[i[:,2],:]
        out=torch.einsum("ab,abc->ac", inp,exp)
        out=out+b

        prob=gateProbabilities[i[:,0],topKindices[i[:,0],i[:,1],i[:,2]],i[:,2]]

        out=out/prob.view(-1,1)

        outputs[i[:,0],topKindices[i[:,0],i[:,1],i[:,2]],:]+=out
        
        return outputs
class MlpPatches(nn.Module):
    def __init__(self,w,h,nOfPatches):
        super(MlpPatches, self).__init__()
        self.w=w
        self.h=h
        self.nOfPatches=nOfPatches
        self.tokenSize=int(3*(self.w/self.nOfPatches)*(self.h/self.nOfPatches))

        self.moefc=Expert(self.tokenSize,128)

        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
       
        self.fc4= nn.Linear(128*self.nOfPatches**2,128)

        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 10)


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
        self.fc6 = nn.Linear(128, 10)

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
            #self.selfAttention=Attention(self.inputDimension,self.hiddenAttentionDimension,self.nOfExperts)
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
            self.moefc=MoeFcTokensParallel(self.tokenSize,128,self.nOfExperts,self.k,useAttention=self.useAttention)
        else:
            self.moefc=MoeFc(self.tokenSize,128,self.nOfExperts,self.k,useAttention=self.useAttention)

        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
       
        self.fc4= nn.Linear(128*self.nOfPatches**2,128)

        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 10)


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

class moeTransformerFc(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mha=torch.nn.MultiheadAttention(768, 8,batch_first=True)
        self.norm1=nn.LayerNorm(768)
        self.norm2=nn.LayerNorm(768)
        self.fc=MoeFcTokens(768,768,60,3,useAttention=False)
        self.lastLayer=nn.Linear(768*100,10)
    
    def forward(self, x):
        x=x.view(x.shape[0],x.shape[1],-1)
        x_1=x
        x=self.mha(x,x,x)[0]
        x=self.norm1(x+x_1)

        x_2=x
        x=self.fc(x)
        x=self.norm2(x+x_2)
        
        x=self.lastLayer(x.view(x.shape[0],-1))

        return x

class vit(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model=vit_b_16()
    
    def forward(self, x):
        return self.model(x)

class moeStack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.moe1=MoeFcTokens(768,768,20,3,useAttention=False)
        self.moe2=MoeFcTokens(768,768,20,3,useAttention=False)
        self.moe3=MoeFcTokens(768,768,20,3,useAttention=False)
        self.lastLayer=nn.Linear(768*100,128)
        self.lastLayer2=nn.Linear(128,10)

    
    def forward(self, x):
        x=x.view(x.shape[0],x.shape[1],-1)
        x=self.moe1(x)
        x=self.moe2(x)
       # x=self.moe3(x)
        x=self.lastLayer(x.view(x.shape[0],-1))
        x=self.lastLayer2(x)

        return x
