#CLIP is an Open AI tool that utilizes AI to turn text into embeddings
import torch 
from torch import nn as nn
from nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    
    # (variable name: varaible type)
    def __init__(self, n_vocab: int, n_embd: int, n_tokens: int):
         super().__init__()
         
         self.token_embedding = nn.Embedding(n_vocab, n_embd)   
         self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embd))
         
    def forward(self, tokens):
        # (Batch size, sequence length) -> (Batch__size, seq_len, Dimension)
        x = self.token_embedding(tokens)
        
        #add the positional encodings of each position to each token
        x += self.position_embedding
        
        return x
        
class CLIPLayer(nn.Module):
    #constructor with three paramters
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        
        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd) #another layer normalization
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        #first residual connection
        residue = x
        
        # apply the layer normalization and then self attention
        
        x = self.layernorm_1(x)
        
        #casual mask -> every tokens can not watch the next tokens so it can't be related to future tokens but only to the one to the next of it
        x = self.attention(x, casual_mask=True)
        
        x += residue
        
        # feed forward layer
        
        #firs the residue connection
        residue = x
        
        #second layer normalization
        x = self.layernorm_2(x)
        
        #first linear transformation
        x = self.linear_1(x)
        
        #activation function
        x = x * torch.sigmoid(1.702 * x) #QuickGELU activation function
        
        #second linear transformation
        x = self.linear_2(x)
        
        x += residue
        
        return x
        
        
class CLIP(nn.Module):
    
    #constructor 
    #self is the instance of the class being used
    def __init__(self):
        #CLIPEmbedding(vocabulary size, embedding size, max sequence length for padding)
        self.embedding = CLIPEmbedding(49408, 768, 77)
        
        self.layers = nn.Module([
            # (number of heads of the mutliheaded attention layer, embedding size, layers)
            CLIPLayer(12, 768, 12)
        ])
        
        self.layernorm = nn.LayerNorm(768)
        
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        
        #first covert tokens into embeddings
        tokens = tokens.type(torch.long)
        
        #(Batch_size, sequence length) -> (Batch_size, seq_len, dimensions)
        state = self.embedding(tokens)
        
        for layer in self.layers:
            state = layer(state)
        
        #(Batch_size, seq_len, dimensions)
        output = self.layernorm(state)
        
        return output
