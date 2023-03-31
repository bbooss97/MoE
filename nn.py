import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
#module to implement the prof idea considering for each expert all the tokens and passing to the fc their weighed sum
class MoeMuxExpertChoiceAllTokens(nn.Module):
    def __init__(self,inputDimension,hiddenDimension, outputDimension,nOfExperts,dropout=0.):
        super(MoeMuxExpertChoiceAllTokens, self).__init__()

        self.inputDimension=inputDimension
        self.outputDimension=outputDimension
        self.nOfExperts=nOfExperts
        self.hiddenDimension=hiddenDimension
        self.first=True
        self.dropout=dropout

        self.counter=0
        
        #initialize the weights of the experts
        self.weight1 = torch.nn.Parameter(torch.empty([self.nOfExperts,self.inputDimension+1,self.hiddenDimension]))
        self.weight2 = torch.nn.Parameter(torch.empty([self.nOfExperts,self.hiddenDimension+1,self.outputDimension]))
        torch.nn.init.kaiming_uniform_(self.weight1, a=5**0.5)
        torch.nn.init.kaiming_uniform_(self.weight2, a=5**0.5)
        
        self.gate=nn.Linear(self.inputDimension, self.nOfExperts)

    def forward(self, x):
        self.counter+=1
        #if its the first time i enter the forward i initialize the ones tensor
        if self.first:
            self.first=False
            #this tensor is later added in the inputs to include the bias in the multiplication with the experts
            self.ones=torch.ones([self.nOfExperts,x.shape[0],1]).to(device)

        #compute the logits of the gate
        gateLogits=self.gate(x)
        
        #x has shape batchsize x tokens x inputDimension

        #compute the probability of each expert
        #normalize over the tokens or over the experts
        gateProbabilitiesTokens=nn.Softmax(dim=-2)(gateLogits)
        gateProbabilitiesExperts=nn.Softmax(dim=-1)(gateLogits)

        #gate probabilities have shape batchsize x tokens x experts
        
        #take the weighted sum of the inputs for each expert
        #it multiplies the x by the probability of the expert to choose the token and i sum over the tokens dimension to take the weighted sum
        inp=torch.einsum('btk,bta->bak',x,gateProbabilitiesTokens)
        #inp has shape batchsize x experts x inputDimension

        #average of the cos sim of each input with all the others this is a loss to distantiate the expert's inputs 
        #it can be used or not used in the training using rlLoss variable in the training 
        #distance
        # norm=torch.norm(inp,dim=-1,keepdim=True)
        # norm_tensor=inp/norm
        # distance=torch.einsum('bak,btk->bat',norm_tensor,norm_tensor)
        # self.rlLoss=distance.mean()
        averageInputs=inp.mean(dim=0)
        #l2 distance
        self.norm=torch.norm(averageInputs,dim=-1)
        self.rlLoss=averageInputs/self.norm.unsqueeze(-1)
        self.rlLoss=torch.einsum('ab,cb->ac',self.rlLoss,self.rlLoss)
        self.rlLoss=(self.rlLoss.sum()-self.rlLoss.diag().sum())/(self.rlLoss.shape[0]**2-self.rlLoss.shape[0])


        inp=inp.permute([1,0,2])
        #inp has shape experts x batchsize x inputDimension
        
        #pass them to the experts
        #i add a one to include the bias in the multiplication
        #i use the batched matrix multiplication to parallelize the computation
        #this is the same block of the default vit with 2 layers dropout and gelu
        inp=torch.cat((inp,self.ones),dim=-1)
        out = torch.bmm(inp,self.weight1)
        out=nn.GELU()(out)
        out=nn.Dropout(self.dropout)(out)
        out=torch.cat((out,self.ones),dim=-1)
        out = torch.bmm(out,self.weight2)
        #out has shape experts x batchsize x outputDimension
        
        out=out.permute([1,0,2])
        #out has shape batchsize x experts x outputDimension

        #multiply by the probabilities and reorganize the outputs
        #it multiplies the output of the experts by the probability of the expert to choose the token and i sum over the experts dimension to take the weighted sum of the outputs
        out=torch.einsum('btk,bat->bak',out,gateProbabilitiesExperts)

        #out has shape batchsize x tokens x outputDimension
        return out
    

#module to implement the prof idea considering for each expert the weighted sum of the tokens routed to it
class MoeMuxExpertChoiceKTokens(nn.Module):
    def __init__(self,inputDimension,hiddenDimension, outputDimension,nOfExperts,k=8,dropout=0.):
        super(MoeMuxExpertChoiceKTokens, self).__init__()

        self.inputDimension=inputDimension
        self.outputDimension=outputDimension
        self.nOfExperts=nOfExperts
        self.hiddenDimension=hiddenDimension
        self.first=True
        self.dropout=dropout
        self.k=k

        self.counter=0
        
        #initialize the weights of the experts
        self.weight1 = torch.nn.Parameter(torch.empty([self.nOfExperts,self.inputDimension+1,self.hiddenDimension]))
        self.weight2 = torch.nn.Parameter(torch.empty([self.nOfExperts,self.hiddenDimension+1,self.outputDimension]))
        torch.nn.init.kaiming_uniform_(self.weight1, a=5**0.5)
        torch.nn.init.kaiming_uniform_(self.weight2, a=5**0.5)
        
        self.gate=nn.Linear(self.inputDimension, self.nOfExperts)

    def forward(self, x):
        self.counter+=1
        #if its the first time i enter the forward i initialize the ones tensor
        if self.first:
            self.first=False
            #this tensor is later added in the inputs to include the bias in the multiplication with the experts
            self.ones=torch.ones([self.nOfExperts,x.shape[0],1]).to(device)

        #compute the logits of the gate
        gateLogits=self.gate(x)
        
        #x has shape batchsize x tokens x inputDimension

        #compute the probability of each expert normalize over the tokens 
        gateProbabilitiesTokens=nn.Softmax(dim=-2)(gateLogits)
        #gate probabilities have shape batchsize x tokens x experts
        
        #take the top k tokens for each expert
        topKvalues,topKindices=torch.topk(gateProbabilitiesTokens,self.k,dim=-2)

        #take as inputs for the experts the weighted sum of the top k tokens
        indexes=topKindices.reshape(x.shape[0],-1).unsqueeze(2).repeat(1,1,128)
        inp=torch.gather(x,1,indexes)
        inp=inp.reshape(x.shape[0],self.k,self.nOfExperts,x.shape[-1])
        #inp has shape batchsize x k x experts x inputDimension

        #multiply by the probabilities
        inp=torch.einsum('abcd,abc->acd',inp,topKvalues)
        #inp has shape batchsize x experts x inputDimension

        #average of the cos sim of each input with all the others this is a loss to distantiate the expert's inputs 
        #it can be used or not used in the training using rlLoss variable in the training 
        #distance
        averageInputs=inp.mean(dim=0)
        #l2 distance
        self.norm=torch.norm(averageInputs,dim=-1)
        self.rlLoss=averageInputs/self.norm.unsqueeze(-1)
        self.rlLoss=torch.einsum('ab,cb->ac',self.rlLoss,self.rlLoss)
        self.rlLoss=(self.rlLoss.sum()-self.rlLoss.diag().sum())/(self.rlLoss.shape[0]**2-self.rlLoss.shape[0])


        inp=inp.permute([1,0,2])
        #inp has shape experts x batchsize x inputDimension
        
        #pass them to the experts
        #i add a one to include the bias in the multiplication
        #i use the batched matrix multiplication to parallelize the computation
        #this is the same block of the default vit with 2 layers dropout and gelu
        inp=torch.cat((inp,self.ones),dim=-1)
        out = torch.bmm(inp,self.weight1)
        out=nn.GELU()(out)
        out=nn.Dropout(self.dropout)(out)
        out=torch.cat((out,self.ones),dim=-1)
        out = torch.bmm(out,self.weight2)
        #out has shape experts x batchsize x outputDimension
        
        out=out.permute([1,0,2])
        #out has shape batchsize x experts x outputDimension

        #multiply by the probabilities and reorganize the outputs
        out=torch.einsum("abc,adb->adbc",out,topKvalues)
        #out has shape batchsize x k x experts x outputDimension

        #recombine the outputs of the experts
        out=out.reshape(x.shape[0],-1,self.outputDimension)
        outputs=torch.zeros(x.shape[0],x.shape[1],self.outputDimension)
        outputs=outputs.to(device)

        #add the outputs of the experts to the token
        outputs.scatter_add_(1,indexes,out)
        return outputs

class MoeExpertChoice(nn.Module):
    def __init__(self,inputDimension,hiddenDimension, outputDimension,nOfExperts,k=8,dropout=0.):
        super(MoeExpertChoice, self).__init__()

        self.inputDimension=inputDimension
        self.outputDimension=outputDimension
        self.nOfExperts=nOfExperts
        self.hiddenDimension=hiddenDimension
        self.first=True
        self.dropout=dropout
        self.k=k

        self.counter=0
        
        #initialize the weights of the experts
        self.weight1 = torch.nn.Parameter(torch.empty([self.nOfExperts,self.inputDimension+1,self.hiddenDimension]))
        self.weight2 = torch.nn.Parameter(torch.empty([self.nOfExperts,self.hiddenDimension+1,self.outputDimension]))
        torch.nn.init.kaiming_uniform_(self.weight1, a=5**0.5)
        torch.nn.init.kaiming_uniform_(self.weight2, a=5**0.5)
        
        self.gate=nn.Linear(self.inputDimension, self.nOfExperts)

    def forward(self, x):
        self.counter+=1
        #if its the first time i enter the forward i initialize the ones tensor
        if self.first:
            self.first=False
            #this tensor is later added in the inputs to include the bias in the multiplication with the experts
            self.ones=torch.ones([self.nOfExperts,x.shape[0]*self.k,1]).to(device)

        #compute the logits of the gate
        gateLogits=self.gate(x)
        
        #x has shape batchsize x tokens x inputDimension

        #compute the probability of each expert normalize over the tokens 
        gateProbabilitiesTokens=nn.Softmax(dim=-2)(gateLogits)
        #gate probabilities have shape batchsize x tokens x experts
        
        #take the top k tokens for each expert
        topKvalues,topKindices=torch.topk(gateProbabilitiesTokens,self.k,dim=-2)

        #take as inputs for the experts the weighted sum of the top k tokens
        indexes=topKindices.reshape(x.shape[0],-1).unsqueeze(2).repeat(1,1,128)
        inp=torch.gather(x,1,indexes)
        inp=inp.reshape(x.shape[0],self.k,self.nOfExperts,x.shape[-1])
        #inp has shape batchsize x k x experts x inputDimension


        inp=inp.permute([2,0,1,3])
        #inp has shape experts x batchsize x k x inputDimension

        inp=inp.reshape(self.nOfExperts,-1,x.shape[-1])
        #inp has shape experts x batchsize*k x inputDimension
        
        #pass them to the experts
        #i add a one to include the bias in the multiplication
        #i use the batched matrix multiplication to parallelize the computation
        #this is the same block of the default vit with 2 layers dropout and gelu
        inp=torch.cat((inp,self.ones),dim=-1)
        out = torch.bmm(inp,self.weight1)
        out=nn.GELU()(out)
        out=nn.Dropout(self.dropout)(out)
        out=torch.cat((out,self.ones),dim=-1)
        out = torch.bmm(out,self.weight2)
        #out has shape experts x batchsize x outputDimension
        
        out=out.permute([1,0,2])
        out=out.reshape(x.shape[0],self.k,self.nOfExperts,self.outputDimension)
        #out has shape batchsize x k x experts x outputDimension

        #multiply by the probabilities and reorganize the outputs
        out=torch.einsum("abcd,abc->abcd",out,topKvalues)
        #out has shape batchsize x k x experts x outputDimension

        #recombine the outputs of the experts
        out=out.reshape(x.shape[0],-1,self.outputDimension)
        outputs=torch.zeros(x.shape[0],x.shape[1],self.outputDimension)
        outputs=outputs.to(device)

        #add the outputs of the experts to the token
        outputs.scatter_add_(1,indexes,out)
        return outputs