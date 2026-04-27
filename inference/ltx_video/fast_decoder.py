# Source Generated with Decompyle++
# File: fast_decoder.pyc (Python 3.10)

'''
Standalone distilled VAE decoder (no diffusers dependency).

Usage:
    decoder = load_fast_decoder(
        config_path="/path/to/config.json",
        checkpoint_path="/path/to/decoder.pth",
        device="cuda",
        dtype=torch.float16,
    )
    # latents: [B, 128, 5, 16, 16] (un-normalized)
    output = decoder(latents)  # [B, 3, 33, 512, 512]
'''
import json
from typing import Tuple
import torch
from torch.nn import nn

class RMSNorm(nn.Module):
    
    def __init__(self = None, dim = None, eps = None, elementwise_affine = None):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            return None
        self.weight = None

    
    def forward(self, x):
        input_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(-1, True, **('keepdim',))
        x = x * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            if self.weight.dtype in (torch.float16, torch.bfloat16):
                x = x.to(self.weight.dtype)
            x = x * self.weight
            return x
        x = x.to(input_dtype)
        return x



class TurboVAEDConv2dSplitUpsampler(nn.Module):
    
    def __init__(self = None, in_channels = None, kernel_size = None, stride = None):
        super().__init__()
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        k = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        padding = (k[0] // 2, k[1] // 2)
        self.conv = nn.Conv2d(in_channels, in_channels, k, 1, padding, **('stride', 'padding'))

    
    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.pixel_shuffle(x, self.stride[0])
        return x



class TurboVAEDConv2dUpsampler(nn.Module):
    
    def __init__(self = None, in_channels = None, kernel_size = None, stride = None):
        super().__init__()
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        k = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        padding = (k[0] // 2, k[1] // 2)
        self.conv = nn.Conv2d(in_channels, in_channels, k, 1, padding, **('stride', 'padding'))

    
    def forward(self, x):
        (B, C, T, H, W) = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.conv(x)
        x = nn.functional.pixel_shuffle(x, self.stride[0])
        (_, _, oH, oW) = x.shape
        x = x.reshape(B, T, -1, oH, oW).permute(0, 2, 1, 3, 4)
        return x



class CausalConv3d(nn.Module):
    
    def __init__(self = None, in_ch = None, out_ch = None, kernel_size = None, stride = None, dilation = None, groups = None, is_causal = None):
        super().__init__()
        self.is_causal = is_causal
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        stride = stride if isinstance(stride, tuple) else (stride,) * 3
        dilation = dilation if isinstance(dilation, tuple) else (dilation, 1, 1)
        padding = (0, self.kernel_size[1] // 2, self.kernel_size[2] // 2)
        self.conv = nn.Conv3d(in_ch, out_ch, self.kernel_size, stride, dilation, groups, padding, **('stride', 'dilation', 'groups', 'padding'))

    
    def forward(self, x):
        t_k = self.kernel_size[0]
        if self.is_causal:
            if t_k > 1:
                pad = x[:, :, :1].repeat(1, 1, t_k - 1, 1, 1)
                x = torch.cat([
                    pad,
                    x], 2, **('dim',))
            elif t_k > 1:
                pad_l = x[:, :, :1].repeat(1, 1, (t_k - 1) // 2, 1, 1)
                pad_r = x[:, :, -1:].repeat(1, 1, (t_k - 1) // 2, 1, 1)
                x = torch.cat([
                    pad_l,
                    x,
                    pad_r], 2, **('dim',))
        return self.conv(x)



class DepthwiseSeparableConv3d(nn.Module):
    
    def __init__(self = None, in_ch = None, out_ch = None, kernel_size = None, stride = None, dilation = None, is_causal = None):
        super().__init__()
        self.is_causal = is_causal
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        stride = stride if isinstance(stride, tuple) else (stride,) * 3
        dilation = dilation if isinstance(dilation, tuple) else (dilation, 1, 1)
        padding = (0, self.kernel_size[1] // 2, self.kernel_size[2] // 2)
        self.depthwise_conv = nn.Conv3d(in_ch, in_ch, self.kernel_size, stride, dilation, in_ch, padding, **('stride', 'dilation', 'groups', 'padding'))
        self.pointwise_conv = nn.Conv3d(in_ch, out_ch, 1, **('kernel_size',))

    
    def forward(self, x):
        t_k = self.kernel_size[0]
        if t_k > 1:
            pad_n = (t_k - 1) // 2
            pad_l = x[:, :, :1].repeat(1, 1, pad_n, 1, 1)
            pad_r = x[:, :, -1:].repeat(1, 1, pad_n, 1, 1)
            x = torch.cat([
                pad_l,
                x,
                pad_r], 2, **('dim',))
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x



class ResnetBlock3d(nn.Module):
    
    def __init__(self = None, in_ch = None, out_ch = None, eps = None, is_causal = None, is_dw_conv = None, dw_kernel_size = None, is_upsampler_modified = None):
        super().__init__()
        if not out_ch:
            pass
        out_ch = in_ch
        conv_cls = DepthwiseSeparableConv3d if is_dw_conv else CausalConv3d
        k = dw_kernel_size if is_dw_conv else 3
        self.is_upsampler_modified = is_upsampler_modified
        self.nonlinearity = nn.SiLU()
        self.replace_nonlinearity = nn.ReLU()
        self.norm1 = RMSNorm(in_ch, 1e-08, **('eps',))
        self.conv1 = conv_cls(in_ch, out_ch, k, is_causal, **('kernel_size', 'is_causal'))
        self.norm2 = RMSNorm(out_ch, 1e-08, **('eps',))
        self.conv2 = conv_cls(out_ch, out_ch, k, is_causal, **('kernel_size', 'is_causal'))
        self.norm3 = None
        self.conv_shortcut = None
        if in_ch != out_ch:
            self.norm3 = nn.LayerNorm(in_ch, eps, True, True, **('eps', 'elementwise_affine', 'bias'))
            self.conv_shortcut = conv_cls(in_ch, out_ch, 1, 1, is_causal, **('kernel_size', 'stride', 'is_causal'))
            return None

    
    def forward(self, x):
        residual = x
        x = self.norm1(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        if self.is_upsampler_modified:
            x = self.replace_nonlinearity(x)
        else:
            x = self.nonlinearity(x)
        x = self.conv1(x)
        x = self.norm2(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        if self.norm3 is not None:
            residual = self.norm3(residual.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return x + residual



class MidBlock3d(nn.Module):
    
    def __init__(self = None, in_ch = None, num_layers = None, eps = None, is_causal = None, is_dw_conv = None, dw_kernel_size = None):
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock3d(in_ch, in_ch, eps=eps, is_causal=is_causal, is_dw_conv=is_dw_conv, dw_kernel_size=dw_kernel_size) for _ in range(num_layers)])

    
    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        return x



class Upsampler3d(nn.Module):
    
    def __init__(self = None, in_ch = None, stride = None, is_causal = None, upscale_factor = None):
        super().__init__()
        self.stride = stride if isinstance(stride, tuple) else (stride,) * 3
        out_ch = in_ch * stride[0] * stride[1] * stride[2] // upscale_factor
        self.conv = CausalConv3d(in_ch, out_ch, 3, 1, is_causal, **('kernel_size', 'stride', 'is_causal'))

    
    def forward(self, x):
        (B, C, T, H, W) = x.shape
        x = self.conv(x)
        x = x.reshape(B, -1, self.stride[0], T, H, W)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, -1, T * self.stride[0], H, W)
        uT = T * self.stride[0]
        x = x.permute(0, 2, 1, 3, 4).reshape(B * uT, -1, H, W)
        x = nn.functional.pixel_shuffle(x, self.stride[1])
        (_, c, h, w) = x.shape
        x = x.reshape(B, uT, c, h, w).permute(0, 2, 1, 3, 4)
        x = x[:, :, self.stride[0] - 1:]
        return x



class UpBlock3d(nn.Module):
    
    def __init__(self, in_ch = None, out_ch = None, num_layers = None, eps = None, spatio_temporal_scale = None, is_causal = None, upscale_factor = None, is_dw_conv = None, dw_kernel_size = ((None, 1, 1e-06, True, True, 1, False, 3),)):
        super().__init__()
        if not out_ch:
            pass
        out_ch = in_ch
        self.conv_in = None
        if in_ch != out_ch:
            self.conv_in = ResnetBlock3d(in_ch, out_ch, eps, is_causal, is_dw_conv, dw_kernel_size, **('eps', 'is_causal', 'is_dw_conv', 'dw_kernel_size'))
        self.upsamplers = None
        if spatio_temporal_scale:
            self.upsamplers = nn.ModuleList([
                Upsampler3d(out_ch * upscale_factor, (2, 2, 2), is_causal, upscale_factor, **('stride', 'is_causal', 'upscale_factor'))])
        self.resnets = nn.ModuleList([ResnetBlock3d(out_ch, out_ch, eps=eps, is_causal=is_causal, is_dw_conv=is_dw_conv, dw_kernel_size=dw_kernel_size, spatio_temporal_scale=spatio_temporal_scale) for _ in range(num_layers)])

    
    def forward(self, x):
        if self.conv_in is not None:
            x = self.conv_in(x)
        if self.upsamplers is not None:
            for up in self.upsamplers:
                x = up(x)
        for resnet in self.resnets:
            x = resnet(x)
        return x



class TurboVAEDDecoder3d(nn.Module):
    '''FastDecoder: lightweight distilled LTX-VAE decoder.'''
    
    def __init__(self, in_channels, out_channels = None, block_out_channels = None, spatio_temporal_scaling = None, layers_per_block = None, patch_size = None, resnet_norm_eps = None, is_causal = None, decoder_is_dw_conv = None, decoder_dw_kernel_size = ((128, 3, (128, 256, 512, 512), (True, True, True, False), (3, 2, 3, 1, 1), 4, 1e-06, False, (False, False, False, True, True), 5),)):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        block_out_channels = tuple(reversed(block_out_channels))
        spatio_temporal_scaling = tuple(reversed(spatio_temporal_scaling))
        layers_per_block = tuple(reversed(layers_per_block))
        decoder_is_dw_conv = tuple(reversed(decoder_is_dw_conv))
        output_channel = block_out_channels[0]
        self.conv_in = CausalConv3d(in_channels, output_channel, 3, 1, is_causal, **('kernel_size', 'stride', 'is_causal'))
        self.mid_block = MidBlock3d(output_channel, layers_per_block[0], resnet_norm_eps, is_causal, decoder_is_dw_conv[0], decoder_dw_kernel_size, **('num_layers', 'eps', 'is_causal', 'is_dw_conv', 'dw_kernel_size'))
        self.up_blocks = nn.ModuleList()
        for i in range(len(block_out_channels)):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            self.up_blocks.append(UpBlock3d(input_channel, output_channel, layers_per_block[i + 1], resnet_norm_eps, spatio_temporal_scaling[i], is_causal, decoder_is_dw_conv[i + 1], decoder_dw_kernel_size, **('num_layers', 'eps', 'spatio_temporal_scale', 'is_causal', 'is_dw_conv', 'dw_kernel_size')))
        if patch_size >= 2:
            self.norm_up_1 = RMSNorm(output_channel, 1e-08, **('eps',))
            self.upsampler2d_1 = TurboVAEDConv2dSplitUpsampler(output_channel, 3, (2, 2), **('kernel_size', 'stride'))
            output_channel = output_channel // 4
        if patch_size >= 4:
            self.norm_up_2 = RMSNorm(output_channel, 1e-08, **('eps',))
            self.upsampler2d_2 = TurboVAEDConv2dUpsampler(output_channel, 3, (2, 2), **('kernel_size', 'stride'))
            output_channel = output_channel // 4
        self.conv_act = nn.SiLU()
        self.conv_out = CausalConv3d(output_channel, out_channels, 3, 1, is_causal, **('kernel_size', 'stride', 'is_causal'))

    
    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid_block(x)
        for up_block in self.up_blocks:
            x = up_block(x)
        if self.patch_size >= 2:
            x = self.norm_up_1(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            x = self.conv_act(x)
            frames = []
            for t in range(x.shape[2]):
                frames.append(self.upsampler2d_1(x[:, :, t]))
            x = torch.stack(frames, 2, **('dim',))
        if self.patch_size >= 4:
            x = self.norm_up_2(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            x = self.conv_act(x)
            x = self.upsampler2d_2(x)
        variance = x.pow(2).mean(1, True, **('keepdim',))
        x = x * torch.rsqrt(variance + 1e-08)
        x = self.conv_act(x)
        x = self.conv_out(x)
        return x



def load_fast_decoder(config_path, checkpoint_path, device, dtype = ('cuda', torch.float16)):
    '''Load FastDecoder from config + checkpoint.

    Args:
        config_path: Path to FastDecoder-LTX.json
        checkpoint_path: Path to FastDecoder-LTX.pth
        device: Target device
        dtype: Target dtype (float16 recommended for H100)

    Returns:
        TurboVAEDDecoder3d model ready for inference
    '''
    with open(config_path) as f:
        config = json.load(f)
        None(None, None, None)
    with None:
        if not None:
            pass
    decoder = TurboVAEDDecoder3d(config.get('latent_channels', 128), config.get('out_channels', 3), tuple(config.get('decoder_block_out_channels', [
        64,
        128,
        256,
        512])), tuple(config.get('spatio_temporal_scaling', [
        True,
        True,
        True,
        False])), tuple(config.get('decoder_layers_per_block', [
        3,
        2,
        3,
        1,
        1])), config.get('patch_size', 4), config.get('resnet_norm_eps', 1e-06), config.get('decoder_causal', False), tuple(config.get('decoder_is_dw_conv', [
        False,
        False,
        False,
        True,
        True])), config.get('decoder_dw_kernel_size', 5), **('in_channels', 'out_channels', 'block_out_channels', 'spatio_temporal_scaling', 'layers_per_block', 'patch_size', 'resnet_norm_eps', 'is_causal', 'decoder_is_dw_conv', 'decoder_dw_kernel_size'))
    checkpoint = torch.load(checkpoint_path, 'cpu', True, **('map_location', 'weights_only'))
    decoder_keys = {k: v for k, v in checkpoint.items()}
    (missing, unexpected) = decoder.load_state_dict(decoder_keys, False, **('strict',))
    if missing:
        raise RuntimeError(f'''Missing keys in FastDecoder: {missing}''')
    decoder.requires_grad_(False)
    decoder.to(device, dtype, **('device', 'dtype'))
    return decoder

