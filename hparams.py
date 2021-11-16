import torch
import os

AUDIO_DIR = r"D:\It-Jim\small_speech_and_noise_dataset"
SAMPLE_RATE = 16000

SPECTROGRAM_N_FREQS = 128
SPECTROGRAM_N_TIMESTAMPS = 128
N_FFT = SPECTROGRAM_N_FREQS * 2 - 2
N_SAMPLES = N_FFT*SPECTROGRAM_N_TIMESTAMPS//2 - N_FFT//2

FRAME_OFFSET = 32000
BATCH_SIZE = 128
NUM_EPOCHS = 30000
LEARNING_RATE = 1e-3
MODEL_DIR = r"D:\It-Jim\spectrogram_to_audio\trained"
MODEL_PATH = os.path.join(MODEL_DIR, 'min_loss.pt')
INFERENCE_DIR = r"D:\It-Jim\spectrogram_to_audio\inference"

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


SPECTROGRAM_PARAMS = {
    'n_fft': N_FFT,
    #'hop_length': 168
}
