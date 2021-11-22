import torch
from torch.utils import data
import torchaudio
import dataset
from hparams import *
import matplotlib.pyplot as plt
from denoise import denoise
import os

dataset = dataset.SpeechAndNoiseDataset('test', AUDIO_DIR, SAMPLE_RATE, N_SAMPLES, FRAME_OFFSET, DEVICE, return_orig=True)


noisy, clean = dataset[0]
# t = torch.arange(0, clean.shape[-1], 0.01)
# clean = torch.sin((0.01*t)**3).unsqueeze(0).to('cuda')
denoised = denoise(clean.to(DEVICE))


torchaudio.save(os.path.join(INFERENCE_DIR, 'clean.wav'), clean.cpu(), SAMPLE_RATE)
torchaudio.save(os.path.join(INFERENCE_DIR, 'noisy.wav'), noisy.cpu(), SAMPLE_RATE)
torchaudio.save(os.path.join(INFERENCE_DIR, 'denoised.wav'), denoised.cpu(), SAMPLE_RATE)


# Get spectrograms for plotting
# spectrogram_function = torchaudio.transforms.MelSpectrogram(**SPECTROGRAM_PARAMS, power=1, n_mels=SPECTROGRAM_N_FREQS//3+4).to(DEVICE)
spectrogram_function = torchaudio.transforms.Spectrogram(**SPECTROGRAM_PARAMS, power=1).to(DEVICE)
clean_magnitude = spectrogram_function(clean)
noisy_magnitude = spectrogram_function(noisy)
denoised_magnitude = spectrogram_function(denoised)

# Plot
plt.rcParams['font.size'] = '6'
fig, axs = plt.subplots(3, figsize=(5, 6), dpi=300)

axs[0].imshow(clean_magnitude.squeeze().cpu(), aspect='auto')
axs[0].set_title('Clean')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Frequency bin')

axs[1].imshow(noisy_magnitude.squeeze().cpu(), aspect='auto')
axs[1].set_title('Noisy')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Frequency bin')

axs[2].imshow(denoised_magnitude.squeeze().detach().cpu(), aspect='auto')
axs[2].set_title('Denoised')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Frequency bin')

fig.suptitle('Spectrograms')
fig.tight_layout()

fig.savefig(os.path.join(INFERENCE_DIR, 'spectrograms.jpg'), format='jpg')