import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

#We decrease the size of the image but also at the same time we keep on adding more channels (specific component of the image ) to the image
class VAE_Encoder(nn.Sequential):
    
    def __init__(self):
        
        #We keep on decreasing the size of the image but each pixel holds more information while # of pixels are decreasing
        super().__init__(
            # (Batch_Size, Channel, Height, Width) -> (Batch_Size, 128, Height, Width)
            #A kernel slides over the 2d input data and performs multiplication -> creates layers in neural networks
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
            #(input channels, output channels) - Wont change the size of the image                      
            VAE_ResidualBlock(128,128),
                        
            #(input channels, output channels) - Wont the size of the image
            VAE_ResidualBlock(128,128),
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height / 2, Width / 2) (changes the size of the image)
            nn.Conv2d(128,128, kernel_size=3, stride=2, padding=0),
            
            # (Batch_Size, 128, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2) -> Channels increase
            VAE_ResidualBlock(128,256),
             # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2) -> Chanenls stays the same
            VAE_ResidualBlock(256,256),
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 4, Width / 4) -> Size gets halfed again
            nn.Conv2d(256,256, kernel_size=3, stride=2, padding=0),
            
            # (Batch_Size, 256, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4) -> Channels increase
            VAE_ResidualBlock(256,512),
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4) -> Channels stays the same
            VAE_ResidualBlock(512,512),
             
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8) -> Size gets halfed again
            nn.Conv2d(512,512, kernel_size=3, stride=2, padding=0),
            
            VAE_ResidualBlock(512,512), 
            
            VAE_ResidualBlock(512,512),
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8) --> stays the same
            VAE_ResidualBlock(512,512),
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8) -> stays the same
            VAE_AttentionBlock(512),
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8) --> stays the same
            VAE_ResidualBlock(512,512),
            
            #caclulates the standard deviation -> (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8) --> stays the same
            nn.GroupNorm(32, 512),
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8) -> stays the same
            nn.SiLU(),
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8) -> changes from the max to 8 and bottle neck of the encoder
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8) -> stays the same 
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )
        
        #runs through __init__() -> each convolution block 
        #@return torch.Tensor (variable x which is a tensor)
        #@param x, noise -> both tensor values
        def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
            #x : (Batch_size, Channnel, Height, Width)
            #noise: (Batch_Size, Output_Channels, Height /8, Width /8)
            
            #runs the modules in init sequentially
            for module in self:
                #for convulations that we have destroyed we need a special embed padding 
                if getattr(module, 'stride', None) == (2,2):
                    #pad ( left, right, top, bottom) -> applies padding 
                    x = F.pad(x, (0,1,0,1))
                x = module(x)
                
            #Output of the variotional encoder is the mean and the log variance 
            # (Batch_size, 8, Height, Height / 8, Width /8 )
            mean, log_variance = torch.chunk(x, 2, dim=1) # chunk -> divide into 2 tensors along this dimension  (Batch_Size, 4, height / 8, width / 8)
            
            # (Batch_Size, 4, height / 8, width / 8) -> (Batch_Size, 4, height / 8, width / 8) -> stays the same
            log_variance = torch.clamp(log_variance, -30, 20)
            
            # (Batch_Size, 4, height / 8, width / 8) -> (Batch_Size, 4, height / 8, width / 8) -> stays the same
            variance = log_variance.exp()  #returns tensor with exponential of the elements
            
            # (Batch_Size, 4, height / 8, width / 8) -> (Batch_Size, 4, height / 8, width / 8) -> stays the same
            stdev = variance.sqrt() # #returns tensor with sqrt of the elements
            
            # Z = N(0,1 ) -> X = N(mean, variance) ;  How do we convert (0,1) to the mean and variance that we are looking for
            # X = mean + stdev * Z
            x = mean + stdev * noise
            
            # Scale the output by a constant
            x *= 0.18215
            
            return x
            