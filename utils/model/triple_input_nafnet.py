"""
Triple Input NAFNet: NAFNet that takes 3 inputs for enhanced MRI reconstruction
- VarNet reconstructed image
- GRAPPA reconstructed image  
- Aliased input image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class TripleInputNAFNet(nn.Module):
    """
    NAFNet that takes three inputs and produces enhanced reconstruction
    """
    def __init__(self, img_channel=1, width=32, middle_blk_num=8, 
                 enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]):
        super().__init__()
        
        # Input channels = 3 (varnet + grappa + input)
        total_input_channels = 3 * img_channel
        
        # Initial convolution to process 3-channel input
        self.intro = nn.Conv2d(in_channels=total_input_channels, out_channels=width, 
                              kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        
        # Output convolution back to single channel
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, 
                               kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, varnet_output, grappa_image, input_image):
        """
        Args:
            varnet_output: Reconstructed image from VarNet [B, H, W] or [B, 1, H, W]
            grappa_image: GRAPPA reconstructed image [B, H, W] or [B, 1, H, W]  
            input_image: Aliased input image [B, H, W] or [B, 1, H, W]
        Returns:
            enhanced_image: Final enhanced reconstruction [B, H, W] or [B, 1, H, W]
        """
        # Ensure all inputs have channel dimension
        if varnet_output.dim() == 3:
            varnet_output = varnet_output.unsqueeze(1)
        if grappa_image.dim() == 3:
            grappa_image = grappa_image.unsqueeze(1)
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(1)
            
        # Concatenate along channel dimension
        inp = torch.cat([varnet_output, grappa_image, input_image], dim=1)
        
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        
        # Skip connection from VarNet output (primary reconstruction)
        x = x + varnet_output

        result = x[:, :, :H, :W]
        
        # Return to original shape if input was 3D
        if result.shape[1] == 1:
            result = result.squeeze(1)
            
        return result

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class VarNetWithTripleInputNAFNet(nn.Module):
    """
    Combined model: VarNet + Triple Input NAFNet
    """
    def __init__(self, varnet_config=None, nafnet_config=None):
        super().__init__()
        
        # VarNet configuration
        if varnet_config is None:
            varnet_config = {
                'num_cascades': 12,
                'sens_chans': 8,
                'sens_pools': 4,
                'chans': 18,
                'pools': 4
            }
        
        # NAFNet configuration  
        if nafnet_config is None:
            nafnet_config = {
                'img_channel': 1,
                'width': 32,
                'middle_blk_num': 8,
                'enc_blk_nums': [2, 2, 4, 8],
                'dec_blk_nums': [2, 2, 2, 2]
            }
            
        # Import VarNet
        from varnet import VarNet
        
        self.varnet = VarNet(**varnet_config)
        self.nafnet = TripleInputNAFNet(**nafnet_config)
        
    def forward(self, masked_kspace, mask, grappa_image, input_image):
        """
        Args:
            masked_kspace: Input k-space data
            mask: Sampling mask
            grappa_image: GRAPPA reconstructed image
            input_image: Aliased input image
        """
        # First stage: VarNet reconstruction
        varnet_output = self.varnet(masked_kspace, mask)
        
        # Second stage: Triple Input NAFNet enhancement
        enhanced_output = self.nafnet(varnet_output, grappa_image, input_image)
        
        return enhanced_output


class TripleInputNAFNetStandalone(nn.Module):
    """
    Standalone Triple Input NAFNet for enhancing pre-reconstructed images
    """
    def __init__(self, img_channel=1, width=32, middle_blk_num=8, 
                 enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]):
        super().__init__()
        self.nafnet = TripleInputNAFNet(
            img_channel=img_channel,
            width=width, 
            middle_blk_num=middle_blk_num,
            enc_blk_nums=enc_blk_nums,
            dec_blk_nums=dec_blk_nums
        )
        
    def forward(self, varnet_output, grappa_image, input_image):
        """
        Args:
            varnet_output: Reconstructed image from VarNet
            grappa_image: GRAPPA reconstructed image  
            input_image: Aliased input image
        Returns:
            enhanced_image: Enhanced image
        """
        return self.nafnet(varnet_output, grappa_image, input_image)
