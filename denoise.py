import torch
import torchaudio
from hparams import DEVICE, SPECTROGRAM_PARAMS, MODEL_PATH
model = torch.load(MODEL_PATH).to(DEVICE)

def to_complex(mag, phase):
    return mag * torch.e**(1j*phase)

spectrogram_function = torchaudio.transforms.Spectrogram(**SPECTROGRAM_PARAMS, power=None).to(DEVICE)
inverse_spectrogram_function = torchaudio.transforms.InverseSpectrogram(**SPECTROGRAM_PARAMS).to(DEVICE)

def denoise(signal):

    spectrogram = spectrogram_function(signal)
    magnitude, phase = torch.abs(spectrogram), torch.angle(spectrogram)
    with torch.no_grad():
        denoised_magnitude = model(magnitude.unsqueeze(0)).squeeze().unsqueeze(0)
    back_to_complex = to_complex(denoised_magnitude, phase)
    denoised = inverse_spectrogram_function(back_to_complex)

    return denoised