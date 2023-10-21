import os
import numpy as np
import yaml
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from .mel2control import Mel2Control
from .core import frequency_filter, upsample, remove_above_fmax
from ..lvc.lvcnet import LVCNetGenerator


class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__
    
def load_model(
        model_path,
        device='cpu'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    
    # load model
    print(' [Loading] ' + model_path)
    if model_path.split('.')[-1] == 'jit':
        model = torch.jit.load(model_path, map_location=torch.device(device))
    else:
        if args.model.type == 'Sins':
            model = Sins(
                sampling_rate=args.data.sampling_rate,
                block_size=args.data.block_size,
                win_length=args.data.n_fft,
                n_harmonics=args.model.n_harmonics,
                n_mag_noise=args.model.n_mag_noise,
                n_mels=args.data.n_mels)
    
        elif args.model.type == 'CombSub':
            model = CombSub(
                sampling_rate=args.data.sampling_rate,
                block_size=args.data.block_size,
                win_length=args.data.n_fft,
                n_mag_harmonic=args.model.n_mag_harmonic,
                n_mag_noise=args.model.n_mag_noise,
                n_mels=args.data.n_mels)
                        
        else:
            raise ValueError(f" [x] Unknown Model: {args.model.type}")
        model.to(device)
        ckpt = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(ckpt['model'])
        model.eval()
    return model, args
    
class Audio2Mel(torch.nn.Module):
    def __init__(
        self,
        hop_length,
        sampling_rate,
        n_mel_channels,
        win_length,
        n_fft=None,
        mel_fmin=0,
        mel_fmax=None,
        clamp = 1e-5
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        mel_basis = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=n_fft, 
            n_mels=n_mel_channels, 
            fmin=mel_fmin, 
            fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

    def forward(self, audio, keyshift=0, speed=1):
        '''
              audio: B x C x T
        log_mel_spec: B x T_ x C x n_mel 
        '''
        factor = 2 ** (keyshift / 12)       
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))
        
        keyshift_key = str(keyshift)+'_'+str(audio.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_length_new).to(audio.device)
            
        B, C, T = audio.shape
        audio = audio.reshape(B * C, T)
        fft = torch.stft(
            audio,
            n_fft=n_fft_new,
            hop_length=hop_length_new,
            win_length=win_length_new,
            window=self.hann_window[keyshift_key],
            center=True,
            return_complex=True)
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        
        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size-resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new
            
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=self.clamp))

        # log_mel_spec: B x C, M, T
        T_ = log_mel_spec.shape[-1]
        log_mel_spec = log_mel_spec.reshape(B, C, self.n_mel_channels ,T_)
        log_mel_spec = log_mel_spec.permute(0, 3, 1, 2)

        # print('og_mel_spec:', log_mel_spec.shape)
        log_mel_spec = log_mel_spec.squeeze(2) # mono
        return log_mel_spec


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

class Sins(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            win_length,
            n_harmonics,
            n_mag_noise,
            n_mels=80,lvc_noise_channels=8,lvc_latent_channels=8,  lvc_block_nums=2,
                 lvc_layers_each_block=5,
                 lvc_kernel_size=3,
                 kpnet_hidden_channels=128,
                 kpnet_conv_size=1,
                 dropout=0.0,up_condition=2):
        super().__init__()

        print(' [DDSP Model] Sinusoids Additive Synthesiser')

        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("win_length", torch.tensor(win_length))
        self.register_buffer("window", torch.hann_window(win_length))
        # Mel2Control
        split_map = {
            'harmonic_phase': win_length // 2 + 1,
            'amplitudes': n_harmonics,
            'noise_magnitude': n_mag_noise,
        }
        self.mel2ctrl = Mel2Control(n_mels, split_map)
        lvc_con=block_size//up_condition if up_condition is not None else block_size
        self.lvc=LVCNetGenerator(in_channels=lvc_noise_channels,
                 out_channels=1,
                 inner_channels=lvc_latent_channels,
                 cond_channels=n_mag_noise,
                 cond_hop_length=lvc_con,
                 lvc_block_nums=lvc_block_nums,
                 lvc_layers_each_block=lvc_layers_each_block,
                 lvc_kernel_size=lvc_kernel_size,
                 kpnet_hidden_channels=kpnet_hidden_channels,
                 kpnet_conv_size=kpnet_conv_size,
                 dropout=dropout)
        self.lvc_noise_channels = lvc_noise_channels
        self.upnet=torch.nn.Sequential(*[Upspamper() for i in range(up_condition//2)]) if up_condition is not None else torch.nn.Identity()

    def forward(self, mel_frames, f0_frames, initial_phase=None, infer=True, max_upsample_dim=32):
        '''
            mel_frames: B x n_frames x n_mels
            f0_frames: B x n_frames x 1
        '''
        # exciter phase
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / np.pi    
        x = x - torch.round(x)
        x = x.to(f0)
        
        phase = 2 * np.pi * x
        phase_frames = phase[:, ::self.block_size, :]
        
        # parameter prediction
        ctrls = self.mel2ctrl(mel_frames, phase_frames)
        
        src_allpass = torch.exp(1.j * np.pi * ctrls['harmonic_phase'])
        src_allpass = torch.cat((src_allpass, src_allpass[:,-1:,:]), 1)
        amplitudes_frames = torch.exp(ctrls['amplitudes'])/ 128
        # noise_param = torch.exp(ctrls['noise_magnitude']) / 128
        noise_param = ctrls['noise_magnitude']
        
        # sinusoids exciter signal 
        amplitudes_frames = remove_above_fmax(amplitudes_frames, f0_frames, self.sampling_rate / 2, level_start = 1)
        n_harmonic = amplitudes_frames.shape[-1]
        level_harmonic = torch.arange(1, n_harmonic + 1).to(phase)
        sinusoids = 0.
        for n in range(( n_harmonic - 1) // max_upsample_dim + 1):
            start = n * max_upsample_dim
            end = (n + 1) * max_upsample_dim
            phases = phase * level_harmonic[start:end]
            amplitudes = upsample(amplitudes_frames[:,:,start:end], self.block_size)
            sinusoids += (torch.sin(phases) * amplitudes).sum(-1)
        
        # harmonic part filter (all pass)
        harmonic_spec = torch.stft(
                            sinusoids,
                            n_fft = self.win_length,
                            win_length = self.win_length,
                            hop_length = self.block_size,
                            window = self.window,
                            center = True,
                            return_complex = True)
        harmonic_spec = harmonic_spec * src_allpass.permute(0, 2, 1)

        harmonic = torch.istft(
                        harmonic_spec,
                        n_fft = self.win_length,
                        win_length = self.win_length,
                        hop_length = self.block_size,
                        window = self.window,
                        center = True)
                        
        # noise part filter (using constant-windowed LTV-FIR) 
        # noise = torch.rand_like(harmonic).to(noise_param) * 2 - 1
        # noise = frequency_filter(
        #                 noise,
        #                 torch.complex(noise_param, torch.zeros_like(noise_param)),
        #                 hann_window = True)

        noise=torch.randn(harmonic.size()[0],self.lvc_noise_channels,harmonic.size()[1]).to(noise_param)
        noise=self.lvc(noise,self.upnet(noise_param.transpose(1,2))).squeeze(1)
        signal = harmonic + noise

        return signal, phase, (harmonic, noise)
        
class CombSub(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            win_length,
            n_mag_harmonic,
            n_mag_noise,
            n_mels=80,lvc_noise_channels=8,lvc_latent_channels=8,  lvc_block_nums=2,
                 lvc_layers_each_block=5,
                 lvc_kernel_size=3,
                 kpnet_hidden_channels=128,
                 kpnet_conv_size=1,
                 dropout=0.0,up_condition=2):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("win_length", torch.tensor(win_length))
        self.register_buffer("window", torch.hann_window(win_length))
        # Mel2Control
        split_map = {
            'harmonic_phase': win_length // 2 + 1,
            'harmonic_magnitude': n_mag_harmonic, 
            'noise_magnitude': n_mag_noise
        }
        self.mel2ctrl = Mel2Control(n_mels, split_map)
        lvc_con = block_size // up_condition if up_condition is not None else block_size
        self.lvc=LVCNetGenerator(in_channels=lvc_noise_channels,
                 out_channels=1,
                 inner_channels=lvc_latent_channels,
                 cond_channels=n_mag_noise,
                 cond_hop_length=lvc_con,
                 lvc_block_nums=lvc_block_nums,
                 lvc_layers_each_block=lvc_layers_each_block,
                 lvc_kernel_size=lvc_kernel_size,
                 kpnet_hidden_channels=kpnet_hidden_channels,
                 kpnet_conv_size=kpnet_conv_size,
                 dropout=dropout)

        self.lvc_noise_channels=lvc_noise_channels
        self.upnet = torch.nn.Sequential(
            *[Upspamper() for i in range(up_condition // 2)]) if up_condition is not None else torch.nn.Identity()

    def forward(self, mel_frames, f0_frames, initial_phase=None, infer=True, **kwargs):
        '''
            mel_frames: B x n_frames x n_mels
            f0_frames: B x n_frames x 1
        '''
        # exciter phase
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / np.pi    
        x = x - torch.round(x)
        x = x.to(f0)
        
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
        
        # parameter prediction
        ctrls = self.mel2ctrl(mel_frames, phase_frames)
        
        src_allpass = torch.exp(1.j * np.pi * ctrls['harmonic_phase'])
        src_allpass = torch.cat((src_allpass, src_allpass[:,-1:,:]), 1)
        src_param = torch.exp(ctrls['harmonic_magnitude'])
        # noise_param = torch.exp(ctrls['noise_magnitude']) / 128
        noise_param = ctrls['noise_magnitude']
        # combtooth exciter signal
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3))
        combtooth = combtooth.squeeze(-1) 
        
        # harmonic part filter (using dynamic-windowed LTV-FIR)
        harmonic = frequency_filter(
                        combtooth,
                        torch.complex(src_param, torch.zeros_like(src_param)),
                        hann_window = True,
                        half_width_frames = 1.5 * self.sampling_rate / (f0_frames + 1e-3))
               
        # harmonic part filter (all pass)
        harmonic_spec = torch.stft(
                            harmonic,
                            n_fft = self.win_length,
                            win_length = self.win_length,
                            hop_length = self.block_size,
                            window = self.window,
                            center = True,
                            return_complex = True)
        harmonic_spec = harmonic_spec * src_allpass.permute(0, 2, 1)
        harmonic = torch.istft(
                        harmonic_spec,
                        n_fft = self.win_length,
                        win_length = self.win_length,
                        hop_length = self.block_size,
                        window = self.window,
                        center = True)
        
        # noise part filter (using constant-windowed LTV-FIR)
        # noise = torch.rand_like(harmonic).to(noise_param) * 2 - 1
        # noise = frequency_filter(
        #                 noise,
        #                 torch.complex(noise_param, torch.zeros_like(noise_param)),
        #                 hann_window = True)

        noise=torch.randn(harmonic.size()[0],self.lvc_noise_channels,harmonic.size()[1]).to(noise_param)
        noise=self.lvc(noise,self.upnet(noise_param.transpose(1,2))).squeeze(1)
                        
        signal = harmonic + noise

        return signal, phase_frames, (harmonic, noise)