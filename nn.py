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

        x=self.fc5(x)
        x=nn.ReLU()(x)

        x=self.fc6(x)
        
        #do not need to use softmax or sigmoid because we use cross entropy loss and it does it for us
        return x


class MoeFc(nn.Module):
    def __init__(self,inputDimension, outputDimension,nOfExperts,k):
        super(MoeFc, self).__init__()
        self.inputDimension=inputDimension
        self.outputDimension=outputDimension
        self.nOfExperts=nOfExperts
        self.k=k
        self.experts=nn.ModuleList([Expert(self.inputDimension,self.outputDimension) for i in range(self.nOfExperts)])
        self.gate=nn.Linear(self.inputDimension, self.nOfExperts)

    def forward(self, x):
        #compute the logits of the gate
        gateLogits=self.gate(x)

        #compute the probability of each expert
        gateProbabilities=nn.Softmax(dim=1)(gateLogits)

        #get the topk
        topKvalues, topKindices=torch.topk(gateProbabilities,self.k,dim=-1)

        outputs=torch.zeros(x.shape[0],x.shape[1],self.outputDimension).to(device)
        #compute the output of each expert
        for i in range(self.nOfExperts):
            x_e=(topKindices==i).nonzero()
            outputs[x_e[:,0],x_e[:,1]]+=self.experts[i](x[x_e[:,0],x_e[:,1]]).T @ topKvalues[x_e[:,0],x_e[:,1],x_e[:,2]]

        return outputs
        

class MoE(nn.Module):
    def __init__(self,w,h,k,nOfExperts,nOfPatches):
        super(MoE, self).__init__()
        self.w=w
        self.h=h
        self.nOfPatches=nOfPatches
        self.k=k
        self.nOfExperts=20
        self.tokenSize=int(3*(self.w/self.nOfPatches)*(self.h/self.nOfPatches))

        self.moefc=MoeFc(self.tokenSize,128,self.nOfExperts,self.k)
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


