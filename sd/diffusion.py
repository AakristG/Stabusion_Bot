import torch
from torch import nn as nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

#
class Diffusion(nn.Module):
    
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
       
    #latent is output of the variational autoencoder  
    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent: (Batch_Size, 4, Height /8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, 320) (time, embedding size)
        
        #(1, 320) -> (1, 1280)
        time = self.time_embedding(time)
        
        # (Batch, 4, Height /8 , Width / 8 ) -> (Batch, 320, Height / 8, Width / 8)
        output = self.unet(latent, context, time)
        
        # (Batch, 320, Height /8 , Width / 8 ) -> (Batch, 4, Height / 8, Width / 8)
        output = self.final(output)
        
       # (Batch, 4, Height / 8, Width / 8)
        return output

#encodes information about the time step that we currently are in    
class TimeEmbedding(nn.Module):
    
    def __init__(self, n_embd: int): 
        super().__init__()
        #linear variable = nn.Linear function (the embedding variable, map into 4 * embedding)
        self.linear1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear2 = nn.Linear(4 * n_embd, 4 * n_embd)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x: (1, 320)
        
        #first linear transformation
        x = self.linear1(x)
        
        #apply silu function
        x = F.silu(x)
        
        #second linear transformation
        x = self.linear2(x)
        
        # returns (1, 1280)
        return x
   
class SwitchSequential(nn.Module):
    
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                # this attention block will compute the cross attention between our latents and the prompt
                x = layer(x, context) 
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)    
        return x
                
                
class UNET(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # every step of the encoder is connect to its equivalent step in decoder 
        
        self.encoders = nn.Module([
            #given a list of layers, we can look at the layers one by one
            
            # (Batch_Size, 4, Height / 8, Width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            
            SwitchSequential(UNET_ResidualBlock(320,320), UNET_AttentionBlock(8, 40)),
            
            SwitchSequential(UNET_ResidualBlock(320,320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 16, Width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            
            #increases the AttentionBlock(number of heads, embedding size) to (number of heads, embedding size * 2)
            SwitchSequential(UNET_ResidualBlock(320,640), UNET_AttentionBlock(8, 80)),
            
            SwitchSequential(UNET_ResidualBlock(640,640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(nn.Conv2d(640,640, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(UNET_ResidualBlock(640,1280), UNET_AttentionBlock(8, 160)),
            
            SwitchSequential(UNET_ResidualBlock(1280,1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(nn.Conv2d(1280,1280, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(UNET_ResidualBlock(1280,1280)),
            
            #(Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280,1280))
        ])
        
        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            
            UNET_AttentionBlock(8, 160),
            
            UNET_ResidualBlock(1280, 1280),
        )
        
        self.decoders = nn.ModuleList([
            
            # (Batch_Size, 2560, Height / 64, Width /64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UpSample(1280)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), UpSample(1280)),
            
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 160), UpSample(640)),
            
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 80)),
            
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            
            
           
        ])
        

class UNET_ResidualBlock(nn.Module):
    {
        
    }
    
class UNET_AttentionBlock(nn.Module):
    {
        
    }
        
class UNET_OutputLayer(nn.Module):
    {
        
    }
    