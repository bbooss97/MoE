import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
#experts
class Expert(nn.Module):
    def __init__(self,input,hidden_dim,dropout=0.):
        self.input=input
        self.hidden_dim=hidden_dim
        self.dropout=dropout
        super(Expert, self).__init__()
        self.fc1=nn.Linear(self.input,self.hidden_dim)
        self.fc2=nn.Linear(self.hidden_dim,self.input)

    def forward(self, x):
        x=self.fc1(x)
        x=nn.GELU()(x)
        x=nn.Dropout(self.dropout)(x)
        x=self.fc2(x)
        x=nn.Dropout(self.dropout)(x)
        return x

#self attention
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
#moe tokens
class MoeFcTokens(nn.Module):
    def __init__(self,inputDimension, outputDimension,dropout,nOfExperts,k,useAttention=False):
        super(MoeFcTokens, self).__init__()
        self.inputDimension=inputDimension
        self.outputDimension=outputDimension
        self.dropout=dropout
        self.nOfExperts=nOfExperts
        self.k=k
        self.counter=0
        self.useAttention=useAttention
        self.experts=nn.ModuleList([Expert(self.inputDimension,self.outputDimension,self.dropout) for i in range(self.nOfExperts)])
        self.hiddenAttentionDimension=1

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

        # self.balancingLoss=gateProbabilities.sum(dim=-2)
        # self.balancingLoss=nn.MSELoss()(self.balancingLoss,torch.ones(self.balancingLoss.shape).to(device)*x.shape[1]/self.nOfExperts).to(device)

        outputs=torch.zeros(x.shape[0],x.shape[1],self.outputDimension).to(device)
        #compute the output of each expert
        for i in range(self.nOfExperts):
            batch_indices=torch.arange(x.shape[0]).reshape(-1,1).expand(x.shape[0],self.k).reshape(-1)
            outputs[batch_indices,topKindices[:,:,i].reshape(-1)]+=(self.experts[i](x[batch_indices,topKindices[:,:,i].reshape(-1)]).T * gateProbabilities[batch_indices,topKindices[:,:,i].reshape(-1),i]).T
            # outputs[batch_indices,topKindices[:,:,i].reshape(-1)]+=self.experts[i](x[batch_indices,topKindices[:,:,i].reshape(-1)])
        return outputs

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, dim),
        #     nn.Dropout(dropout)
        # )
        self.net = MoeFcTokens(dim, hidden_dim, dropout, 32, 1, useAttention=True)
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
