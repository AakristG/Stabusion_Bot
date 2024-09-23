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
        
class UpSample(nn.Module):
    
    def __init__(self, channels: int):
        super().__init__() 
        self.conv = nn.Conv2d(channels, channels, kernel_siz=3, padding=1)
        
    def forward(self, x):
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * 2, Width * 2 )
        x = F.interpolate (x, scale_factor=2, mode="nearest") #doubles the size 
        
        return self.conv(x)
class UNET_ResidualBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, n_time=1280):
        super().__init__()
        self.groupnom_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)
        
        self.groupnom_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # can connect them with each other using the residual connection
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            #if not then we concer thte input into the output otherwise we can't add the tensors together
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padidng=1)
            
    def forward(self, feature, time):
        # feature: (Batch_Size, In_Channels, Heigth, Width)
        # time: (1, 1280)
        
        #we need the way to combine the latent, time, and the prompt embeddings together into the unit and pass it inot the UNET
        
        residue = feature
        
        feature = self.groupnom_feature(feature)
        
        feature  = F.silu(feature)
        
        feature = self.conv_feature(feature)
        
        time = F.silu(time)
        
        time = self.linear_time(time)
        
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        merged = self.groupnom_merged(merged)
        
        merged = F.silu(merged)
        
        merged = self.conv_merged(merged)
        
        return merged + self.residual_layer(residue)
        
        
    
class UNET_AttentionBlock(nn.Module):
    
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_head
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.lienar_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)
        
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
    def forward(self, x, context):
        # x: (Batch_Size, Features, Height, Width)
        # context: (Batch_Size, Seq_len, Dim)
        
        residue_long = x
        
        x = self.groupnorm(x)
        
        x = self.conv_input(x)
        
        #batch size, num of features, height, and width
        n, c, h, w = x.shape
        
        x = x.view(n, c, h * w)

        x = x.transpose(-1, -2)
        
        # apply normalization + self attention with skip connection
        
        residue_short = x
        
        x = self.layernorm_1(x)
        self.attention_1(x)
        x += residue_short
        
        
        residue_short = x
        
        #Normalizaiton + cross attention with skip connection
        x = self.layernorm_2(x)
        
        # Cross Attention
        self.attention_2(x, context)
        
        x += residue_short
        
        residue_short = x
        
        #Normalization = feed forward with GeGLU and skip connection
        x = self.layernorm_3(x)
        
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        
        
        x = self.linear_geglu_2(x)
        
        x += residue_short

        # (Batch_size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)
        
        x = x.view((n,c,h,w))
        
        #defines the long skip connection
        
        return self.conv_output(x) + residue_long
class UNET_OutputLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    #@param x (Bath_Size, 320, Height /8, Width /8)
    def forward(self, x):
        
        #apply group normalization
        x = self.groupnorm(x)
        
        #apply silu method 
        x = F.silu(x)
        
        # apply convolution
        x = self.conv(x)
        
        return x
    