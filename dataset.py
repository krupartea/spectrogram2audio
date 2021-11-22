import torch
import torchaudio
from torch.utils.data import Dataset
import os
import random
from hparams import SPECTROGRAM_PARAMS




class SpeechAndNoiseDataset(Dataset):

    def __init__(self,
                 phase,
                 audio_dir,
                 target_sample_rate,
                 num_samples,
                 frame_offset,
                 device,
                 return_orig=False,
                 normalize=False):
        
        self.phase = phase
        self.audio_dir = audio_dir
        self.device = device
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.return_orig = return_orig
        self.frame_offset = frame_offset
        self.clean_filenames = self._get_filenames('clean_', self.phase) # todo: calling every time is redundand
        self.noise_filenames = self._get_filenames('noise_', self.phase)
        self.normalize = normalize

    def __len__(self):
        return len(self.clean_filenames)

    def __getitem__(self, index):
        clean_sample_path = self._get_audio_sample_path(index, 'clean_', self.phase, self.clean_filenames)
        noise_sample_path = self._get_audio_sample_path(index, 'noise_', self.phase, self.noise_filenames)
        noise_signal, noise_sr = torchaudio.load(noise_sample_path)
        clean_signal, clean_sr = torchaudio.load(clean_sample_path, frame_offset=self.frame_offset)
        clean_signal = clean_signal.to(self.device)
        noise_signal = noise_signal.to(self.device)
        clean_signal = self._to_mono_if_necessary(clean_signal)
        noise_signal = self._to_mono_if_necessary(noise_signal)
        clean_signal = self._resample_if_necessary(clean_signal, clean_sr)
        noise_signal = self._resample_if_necessary(noise_signal, noise_sr)
        clean_signal = self._cut_if_necessary(clean_signal)
        noise_signal = self._cut_if_necessary(noise_signal)
        clean_signal = self._right_pad_if_necessary(clean_signal)
        noise_signal = self._right_pad_if_necessary(noise_signal)
        # noise_signal is a pure noise and noisy_signal is a clean_signal combined with noise_signal
        noisy_signal = self._mix_clean_with_noise(clean_signal, noise_signal)
        if self.return_orig:
            return noisy_signal, clean_signal
        clean_spectrogram = self._make_spectrogram(clean_signal) # target
        noisy_spectrogram = self._make_spectrogram(noisy_signal) # data
        if self.normalize:
            clean_spectrogram = self._normalize(clean_spectrogram)
            noisy_spectrogram = self._normalize(noisy_spectrogram)
        return noisy_spectrogram, clean_spectrogram

    # TODO: OCP violation. Pull transforamiton (spectrogram) creation out of dataset class
    def _make_spectrogram(self, signal):
        transformation = torchaudio.transforms.Spectrogram(**SPECTROGRAM_PARAMS).to(self.device)
        spectrogram = transformation(signal)
        return spectrogram
    
    def _get_filenames(self, prefix, phase):
        # prefix is eihter 'clean_' or 'noise_'
        path = os.path.join(self.audio_dir, prefix+phase)
        filenames = os.listdir(path)
        return filenames

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal

    def _to_mono_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index, prefix, phase, filenames):
        if prefix == 'clean_':
            path = os.path.join(self.audio_dir, prefix+phase, filenames[index])
        if prefix == 'noise_':
            path = os.path.join(self.audio_dir, prefix+phase, random.choice(filenames))
        return path
    
    def _mix_clean_with_noise(self, clean, noise):
        mixed = clean + noise*2.3
        # TODO: test if spectrogram of a signal > 1 is equal to normalized signal spectrogram
#         max_amplitude = max([abs(max(mixed)), abs(min(mixed))])
#         if max_amplitude > 1:
#             mixed = mixed / max_amplitude
        return mixed
    
    def _normalize(self, signal):
        return signal / 20

    