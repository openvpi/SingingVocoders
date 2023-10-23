
import torch
import logging
# from modules import LVCBlock
import torch.nn.functional as F
from torch import nn

from modules.univ_ddsp.block import LVCBlock

LRELU_SLOPE = 0.1

from modules.ddsp.vocoder import CombSub, Sins


class DDSP(nn.Module):
    def __init__(self,config):
        super().__init__()
        if config['model_args']['type']=='CombSub':
            self.ddsp = CombSub(
                sampling_rate=config['audio_sample_rate'],
                block_size=config['hop_size'],
                win_length=config['win_size'],
                n_mag_harmonic=config['model_args']['n_mag_harmonic'],
                n_mag_noise=config['model_args']['n_mag_noise'],
                n_mels=config['audio_num_mel_bins'])
        elif config['model_args']['type']=='Sins':
            self.ddsp = Sins(
                sampling_rate=config['audio_sample_rate'],
                block_size=config['hop_size'],
                win_length=config['win_size'],
                n_harmonics=config['model_args']['n_harmonics'],
                n_mag_noise=config['model_args']['n_mag_noise'],
                n_mels=config['audio_num_mel_bins'])

    def forward(self,mel,f0,infer=False):
        signal, _, (s_h, s_n) = self.ddsp(mel.transpose(1,2), torch.unsqueeze(f0,dim=-1), infer=infer)
        return signal.unsqueeze(1),s_h,s_n

class downblock(nn.Module):
    def __init__(self, down, indim, outdim):
        super().__init__()
        self.c = nn.Conv1d(indim, outdim * 2, kernel_size=down * 2, stride=down, padding=down // 2)
        self.act = GLU(1)
        self.out = nn.Conv1d(outdim, outdim, kernel_size=3, padding=1)
        self.act1 = nn.GELU()

    def forward(self, x):
        return self.act1(self.out(self.act(self.c(x))))

class ddsp_down(nn.Module):
    def __init__(self,dims,downs:list,):
        super().__init__()

        dl=[]
        ppl=[]
        downs.reverse()
        self.fistpp=nn.Conv1d(1,dims,kernel_size=1)
        for idx,i in enumerate(downs[:-1]):
            if idx==0:
                dl.append(downblock(i,1,dims))
                ppl.append(nn.Conv1d(dims,dims,kernel_size=1))
            else:
                dl.append(downblock(i,dims*idx,dims*(idx + 1)))
                ppl.append(nn.Conv1d(dims*(idx + 1), dims, kernel_size=1))
        self.downs = nn.ModuleList(dl)
        self.ppls = nn.ModuleList(ppl)
    def forward(self,x):
        spec=[]
        spec.append(self.fistpp(x))

        for dl,ppl in zip(self.downs,self.ppls ):
            x=dl(x)
            spec.append(ppl(x))
        spec.reverse()
        return spec



class GLU(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class Upspamper(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv2d(1, 8, kernel_size=1)
        self.UP = torch.nn.ConvTranspose2d(4, 8, [3, 32], stride=[1, 2], padding=[1, 15])
        self.Glu = GLU(1)
        self.c2 = torch.nn.Conv2d(4, 8, kernel_size=3,padding=1)
        self.c3 = torch.nn.Conv2d(4, 2, kernel_size=1)



    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x=self.Glu (self.c1(x))
        # x=self.net(x)

        x = self.Glu (self.UP(x))
        x=self.Glu(self.c2(x))+x
        x =self.Glu(self.c3(x))

        spectrogram = torch.squeeze(x, 1)
        return spectrogram

class ddspUnivNet(torch.nn.Module):
    """Parallel WaveGAN Generator module."""

    def __init__(self, h, use_weight_norm=True):

        super().__init__()

        self.ddsp=DDSP(h)



        in_channels = h['model_args']['cond_in_channels']
        out_channels = h['model_args']['out_channels']
        inner_channels = h['model_args']['cg_channels']
        cond_channels =  h['audio_num_mel_bins']
        upsample_ratios =  h['model_args']['upsample_rates']
        lvc_layers_each_block =  h['model_args']['num_lvc_blocks']
        lvc_kernel_size = h['model_args']['lvc_kernels']
        kpnet_hidden_channels =  h['model_args']['lvc_hidden_channels']
        kpnet_conv_size =  h['model_args']['lvc_conv_size']
        dropout =  h['model_args']['dropout']
        # upsample_ratios:list
        self.ddspd = ddsp_down(dims=inner_channels,downs=upsample_ratios.copy(),)

        upmel=h['model_args'].get('upmel')
        self.upblocke=torch.nn.Sequential(*[Upspamper() for i in range(upmel//2)]) if upmel is not None else torch.nn.Identity()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_channels = cond_channels
        self.lvc_block_nums = len(upsample_ratios)

        # define first convolution
        self.first_conv = torch.nn.Conv1d(in_channels, inner_channels,
                                        kernel_size=7, padding=(7 - 1) // 2,
                                        dilation=1, bias=True)

        # define residual blocks
        self.lvc_blocks = torch.nn.ModuleList()
        cond_hop_length = 1
        for n in range(self.lvc_block_nums):
            cond_hop_length = cond_hop_length * upsample_ratios[n]
            lvcb = LVCBlock(
                in_channels=inner_channels,
                cond_channels=cond_channels,
                upsample_ratio=upsample_ratios[n],
                conv_layers=lvc_layers_each_block,
                conv_kernel_size=lvc_kernel_size,
                cond_hop_length=cond_hop_length,
                kpnet_hidden_channels=kpnet_hidden_channels,
                kpnet_conv_size=kpnet_conv_size,
                kpnet_dropout=dropout,
            )
            self.lvc_blocks += [lvcb]

        # define output layers
        self.last_conv_layers = torch.nn.ModuleList([
            torch.nn.Conv1d(inner_channels, out_channels, kernel_size=7, padding=(7 - 1) // 2,
                                        dilation=1, bias=True),

        ])

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x, c,f0,infer=False):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, 1, T).
            c (Tensor): Local conditioning auxiliary features (B, C ,T').
        Returns:
            Tensor: Output tensor (B, out_channels, T)
        """
        ddspwav,s_h,s_n=self.ddsp(mel=c,f0=f0,infer=infer)
        specl=self.ddspd(ddspwav)

        x = self.first_conv(x)
        c=self.upblocke(c)

        for n in range(self.lvc_block_nums):
            x = self.lvc_blocks[n](x, c,specl[n])

        # apply final layers
        for f in self.last_conv_layers:
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = f(x)
        x = torch.tanh(x)
        return x,ddspwav,s_h,s_n

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    @staticmethod
    def _get_receptive_field_size(layers, stacks, kernel_size,
                                  dilation=lambda x: 2 ** x):
        assert layers % stacks == 0
        layers_per_cycle = layers // stacks
        dilations = [dilation(i % layers_per_cycle) for i in range(layers)]
        return (kernel_size - 1) * sum(dilations) + 1

    @property
    def receptive_field_size(self):
        """Return receptive field size."""
        return self._get_receptive_field_size(self.layers, self.stacks, self.kernel_size)

    def inference(self, c=None, x=None):
        """Perform inference.
        Args:
            c (Union[Tensor, ndarray]): Local conditioning auxiliary features (T' ,C).
            x (Union[Tensor, ndarray]): Input noise signal (T, 1).
        Returns:
            Tensor: Output tensor (T, out_channels)
        """
        if x is not None:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float).to(next(self.parameters()).device)
            x = x.transpose(1, 0).unsqueeze(0)
        else:
            assert c is not None
            x = torch.randn(1, 1, len(c) * self.upsample_factor).to(next(self.parameters()).device)
        if c is not None:
            if not isinstance(c, torch.Tensor):
                c = torch.tensor(c, dtype=torch.float).to(next(self.parameters()).device)
            c = c.transpose(1, 0).unsqueeze(0)
            c = torch.nn.ReplicationPad1d(self.aux_context_window)(c)
        return self.forward(x, c).squeeze(0).transpose(1, 0)
