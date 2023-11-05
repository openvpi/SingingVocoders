import numpy as np
import torch
import logging
# from modules import LVCBlock
import torch.nn.functional as F
from torch import nn

from modules.univ_ddsp.block import LVCBlock

LRELU_SLOPE = 0.1

from modules.ddsp.vocoder import CombSub, Sins


class SineGen(torch.nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-waveform (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_threshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(self, samp_rate, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values, upp):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        rad_values = (f0_values / self.sampling_rate).fmod(1.)  # %1意味着n_har的乘积无法后处理优化
        rand_ini = torch.rand(1, self.dim, device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] += rand_ini
        is_half = rad_values.dtype is not torch.float32
        tmp_over_one = torch.cumsum(rad_values.double(), 1)  # % 1  #####%1意味着后面的cumsum无法再优化
        if is_half:
            tmp_over_one = tmp_over_one.half()
        else:
            tmp_over_one = tmp_over_one.float()
        tmp_over_one *= upp
        tmp_over_one = F.interpolate(
            tmp_over_one.transpose(2, 1), scale_factor=upp,
            mode='linear', align_corners=True
        ).transpose(2, 1)
        rad_values = F.interpolate(rad_values.transpose(2, 1), scale_factor=upp, mode='nearest').transpose(2, 1)
        tmp_over_one = tmp_over_one.fmod(1.)
        diff = F.conv2d(
            tmp_over_one.unsqueeze(1), torch.FloatTensor([[[[-1.], [1.]]]]).to(tmp_over_one.device),
            stride=(1, 1), padding=0, dilation=(1, 1)
        ).squeeze(1)  # Equivalent to torch.diff, but able to export ONNX
        cumsum_shift = (diff < 0).double()
        cumsum_shift = torch.cat((
            torch.zeros((f0_values.size()[0], 1, self.dim), dtype=torch.double).to(f0_values.device),
            cumsum_shift
        ), dim=1)
        sines = torch.sin(torch.cumsum(rad_values.double() + cumsum_shift, dim=1) * 2 * np.pi)
        if is_half:
            sines = sines.half()
        else:
            sines = sines.float()
        return sines

    @torch.no_grad()
    def forward(self, f0, upp):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        f0 = f0.unsqueeze(-1)
        fn = torch.multiply(f0, torch.arange(1, self.dim + 1, device=f0.device).reshape((1, 1, -1)))
        sine_waves = self._f02sine(fn, upp) * self.sine_amp
        uv = (f0 > self.voiced_threshold).float()
        uv = F.interpolate(uv.transpose(2, 1), scale_factor=upp, mode='nearest').transpose(2, 1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves


class SourceModuleHnNSF(torch.nn.Module):
    """ SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshold=0):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num,
                                 sine_amp, add_noise_std, voiced_threshold)

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x, upp):
        sine_wavs = self.l_sin_gen(x, upp)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge

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

class nsfUnivNet(torch.nn.Module):
    """Parallel WaveGAN Generator module."""

    def __init__(self, h, use_weight_norm=True):

        super().__init__()

        # self.ddsp=DDSP(h)
        self.m_source = SourceModuleHnNSF(
            sampling_rate=h['audio_sample_rate'],
            harmonic_num=8
        )
        self.upp = int(np.prod(h['hop_size']))



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
        self.upblocke=torch.nn.Sequential(*[Upspamper() for i in range(upmel//2)]) if upmel is not None or upmel==1 else torch.nn.Identity()

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
        pass
        # ddspwav,s_h,s_n=self.ddsp(mel=c,f0=f0,infer=infer)
        har_source = self.m_source(f0, self.upp).transpose(1, 2)
        specl=self.ddspd(har_source)

        x = self.first_conv(x)
        c=self.upblocke(c)

        for n in range(self.lvc_block_nums):
            x = self.lvc_blocks[n](x, c,specl[n])

        # apply final layers
        for f in self.last_conv_layers:
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = f(x)
        x = torch.tanh(x)
        return x,har_source

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
