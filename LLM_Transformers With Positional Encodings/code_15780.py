import torch
import re
from torch.nn import Module, Parameter, ModuleList
import numpy as np

def one_hot(Y, num_classes):
    """ Return one hot embedding vectors """
    return (torch.arange(num_classes)[None,:] == Y[:,None]).float()


def loss_01(H, Y):
    """ 0-1 Loss """
    return (H.argmax(-1) != Y).float().mean()




class SimpleTokenizer:
    """ Simple class to handle tokenization of inputs including whitespace. """

    def _process_whitespace(self, content):
        """ Remove duplicate whitespace and replace non-whitespace separation with special char"""
        content = re.sub(r"[^\S\n]+", " ", content)
        content = re.sub(r"[\n]{3,}", "\n\n", content)
        punctuation = set(re.findall(r"[^\w \t]+", " ".join(content)))

        # replace punctuation not surrounded by whitespace with special anti-space token
        char_set = "".join(re.escape(p) for p in punctuation)
        for _ in range(2):
            content = re.sub(r"([^ ġ])(["+ char_set + "])", r"\1 ġ\2", content)
        for _ in range(2):
            content = re.sub(r"(["+ char_set + "])([^ ġ])", r"\1ġ \2", content)
        return content

    def __init__(self, filename, lowercase=True):
        """ Build tokenization from file """
        with open(filename, "rt",encoding="utf8") as f:
            content = f.read()
        content = self._process_whitespace(content)
        if lowercase:
            content = content.lower()
        self.lowercase = lowercase

        tokens = content.split(" ")
        self.token_encode = dict(zip(set(tokens), range(len(set(tokens)))))
        self.token_decode = {v:k for k,v in self.token_encode.items()}

    def encode(self, text):
        """ Convert text to tokens """
        text = self._process_whitespace(text)
        if self.lowercase:
            text = text.lower()
        return torch.tensor([self.token_encode[t] for t in text.split(" ")])
    
    def decode(self, tokens):
        """ Convert tokens to text """
        text = " ".join([self.token_decode[int(t)] for t in tokens])
        return text.replace(" ġ", "").replace("ġ ", "").replace("ġ", "")
    
    @property
    def vocab_size(self):
        """ Get number of tokens """
        return len(self.token_encode)
    

def ema(x, dim=-1, beta = 0.99):
    """ Compute exponential moving average over tensor """
    x0 = x.split(1,dim=dim)
    out = [x0[0]]
    for val in x0[1:]:
        out.append(beta * out[-1] + (1-beta) * val)
    return torch.cat(out, dim=dim)



#### Custom PyTorch Modules

class Linear(Module):
    """ Matrix multiplication plus bias"""
    def __init__(self, in_dim, out_dim, bias=True, init_factor=2.0):
        super().__init__()
        self.weight = Parameter(torch.randn(in_dim, out_dim) * np.sqrt(init_factor / in_dim))
        if bias:
            self.bias = Parameter(torch.zeros(out_dim))
        else:
            self.bias = None
        
    def forward(self, X):
        out = X @ self.weight
        if self.bias is not None:
            out += self.bias[..., :]
        return out
    

class Embedding(Module):
    """ Embedding that transforms indices to embedding vectors. """
    def __init__(self, num_tokens, dim):
        super().__init__()
        self.weight = Parameter(torch.randn(num_tokens, dim))
    
    def forward(self, Y):
        return self.weight[Y]

    
class ReLU(Module):
    """ Layer for ReLU """
    def forward(self, X):
        return torch.maximum(X, torch.tensor(0.))
    

class CrossEntropyLoss(Module):
    """ Layer for Cross Entropy """
    def forward(self, H, Y):
        return -H[torch.arange(len(Y)),Y].mean() + torch.logsumexp(H, -1).mean()


class Sequential(Module):
    """ Layer for applying a sequence of sub layers"""
    def __init__(self, *layers):
        super().__init__()
        self.layers = ModuleList(layers)

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer(out)
        return out
    

class LayerNorm(Module):
    """ Layer Normalization with additional scaling and bias"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.w = Parameter(torch.ones(dim))
        self.b = Parameter(torch.zeros(dim))
        self.eps = eps
    
    def forward(self, z):
        out = z - torch.mean(z, -1, keepdim=True)
        out = out / torch.sqrt(torch.var(z,-1,keepdim=True) + self.eps)
        return out * self.w[...,:] + self.b[...,:]
        

class SelfAttention(Module):
    """ Multi-head self attention"""
    def __init__(self, d, num_heads, max_seq_length=None):
        super().__init__()
        self.wq = Linear(d,d)
        self.wk = Linear(d,d)
        self.wv = Linear(d,d)
        self.wo = Linear(d,d)
        self.dim = d // num_heads
        self.num_heads = num_heads
        self.max_seq_length=max_seq_length
        assert(self.dim * num_heads == d)

        self.i=0
        self.k_cache=None
        self.v_cache=None
    
    def clear_cache(self):
        self.k_cache=self.k_cache[:,-self.max_seq_length:,:]
        self.v_cache=self.v_cache[:,-self.max_seq_length:,:]

    def forward(self, X, mask = None, use_kv_cache=False):
        # X in (B x T x d)
        B, T, d = X.shape

        # Q, K, V => (B x h x T x d/h)
        Q = self.wq(X).view(B, T, self.num_heads, self.dim).transpose(1,2)
        K = self.wk(X).view(B, T, self.num_heads, self.dim).transpose(1,2)
        V = self.wv(X).view(B, T, self.num_heads, self.dim).transpose(1,2)

        if use_kv_cache and self.k_cache is not None:
            K=torch.cat((self.k_cache,K),dim=2)
            V=torch.cat((self.v_cache,V),dim=2)
            if self.max_seq_length is not None:
                #print("KV Cache")
                self.clear_cache()
            self.k_cache=K
            self.v_cache=V

        #Q @ K.T => B x h x T x T
        scores = Q @ K.transpose(2,3) / np.sqrt(self.dim)
        if mask is not None:
            scores += mask
        
        A = torch.softmax(scores, -1)
        
        return self.wo((A @ V).transpose(1,2).contiguous().view_as(X))
    

class TransformerBlockPostNorm(Module):
    """ Single transformer block with attention plus a feedforward network"""
    def __init__(self, d, num_heads, d_ff):
        super().__init__()
        self.attn = SelfAttention(d, num_heads)
        self.w1 = Linear(d, d_ff)
        self.w2 = Linear(d_ff, d)
        self.norm1 = LayerNorm(d)
        self.norm2 = LayerNorm(d)
        self.relu = ReLU()

    def forward(self, X, mask=None):
        Y = self.norm1(X + self.attn(X, mask))
        return self.norm2(Y + self.w2(self.relu(self.w1(Y))))
    
    
class PositionalEncoding(Module):
    """ Add a sine/cosine positional encoding to an input"""
    def __init__(self, dim):
        super().__init__()
        self.c = 10000. ** (-2 * torch.arange(dim//2) / dim)

    def forward(self, X):
        freq = torch.arange(X.shape[-2])[:,None] * self.c[None,:]
        return X + torch.stack([torch.cos(freq), torch.sin(freq)],2).view(X.shape[-2], X.shape[-1])
        


class SGD:
    def __init__(self, params, lr=1.0):
        self.params = list(params)
        self.lr = lr

    def step(self):
        with torch.no_grad():
            for param in self.params:
                param -= self.lr * param.grad

    def zero_grad(self):
        with torch.no_grad():
            for param in self.params:
                if param.grad is not None:
                    param.grad.zero_()

class Adam:
    def __init__(self, params, lr=1e-3, beta=0.9, gamma=0.999, eps=1e-8):
        self.params=list(params)
        self.lr=lr
        self.beta=beta
        self.gamma=gamma
        self.eps=eps
        self.i=1

        self.u={}
        self.v={}

        for p in self.params:
            self.u[p]=torch.zeros_like(p)
            self.v[p]=torch.zeros_like(p)
    
    def step(self):
        with torch.no_grad():
            for p in self.params:
                self.u[p]=self.beta*self.u[p] + (1-self.beta)*p.grad
                self.v[p]=self.gamma*self.v[p] + (1-self.gamma)*(p.grad**2)

                uhat=self.u[p]/(1-self.beta**self.i)
                vhat=self.v[p]/(1-self.gamma**self.i)

                p -= (self.lr * uhat)/(torch.sqrt(vhat)+self.eps)
    
    def zero_grad(self):
        with torch.no_grad():
            for p in self.params:
                if p.grad is not None:
                    p.grad.zero_()