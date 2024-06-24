# Imports
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
from models.models import HAMM_SNN
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from scipy.io import savemat
my_pc = True
if my_pc:
    import torchaudio
else:
    import librosa

# Definitions
def read_kernels(directory):
    wav_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            filepath = os.path.join(directory, filename)
            sample_rate, data = wavfile.read(filepath)
            var_name = os.path.splitext(filename)[0]
            wav_dict[var_name] = data

    return wav_dict, sample_rate

def plot_dft(ax, true_signal, predicted_signal, title, fs):
    # To numpy
    true_signal = np.squeeze(true_signal.cpu().numpy())
    predicted_signal = np.squeeze(predicted_signal.detach().cpu().numpy())
    
    # Compute DFT and corresponding frequency vector
    true_dft = np.log(np.abs(np.fft.fft(true_signal)))
    predicted_dft = np.log(np.abs(np.fft.fft(predicted_signal)))
    
    n = len(true_signal)
    freq = np.fft.fftfreq(n, d=1/fs)
    
    # Only take the first half of the spectrum (up to Nyquist frequency)
    half_n = n // 2
    true_dft = true_dft[:half_n]
    predicted_dft = predicted_dft[:half_n]
    freq = freq[:half_n]
    
    ax.plot(freq, true_dft, label='True')
    ax.plot(freq, predicted_dft, label='Predicted')
    ax.set_title(title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Log Magnitude')
    ax.legend()


class CustomDataset(Dataset):
    def __init__(self, input_wavs_dir, output_wavs_dir):

        input_ess_dir = os.path.join(input_wavs_dir,'ESS')
        input_mls_dir = os.path.join(input_wavs_dir,'MLS')
        output_ess_dir = os.path.join(output_wavs_dir,'ESS')
        output_mls_dir = os.path.join(output_wavs_dir,'MLS')

        self.input_ess_files = [os.path.join(input_ess_dir, f).replace("\\","/") for f in os.listdir(input_ess_dir)]
        self.input_mls_files  = [os.path.join(input_mls_dir, f).replace("\\","/") for f in os.listdir(input_mls_dir)]
        self.output_ess_files = [os.path.join(output_ess_dir, f).replace("\\","/") for f in os.listdir(output_ess_dir)]
        self.output_mls_files = [os.path.join(output_mls_dir, f).replace("\\","/") for f in os.listdir(output_mls_dir)]

    def __len__(self):
        return min(len(self.input_ess_files), len(self.input_mls_files))

    def __getitem__(self, idx):
        # Waveforms
        if my_pc:
            input_ess, sr = torchaudio.load(self.input_ess_files[idx])
            input_ess = input_ess.contiguous()  # Ensure contiguous memory

            input_mls, sr = torchaudio.load(self.input_mls_files[idx])
            input_mls = input_mls.contiguous()  # Ensure contiguous memory

            output_ess, sr = torchaudio.load(self.output_ess_files[idx])
            output_ess = output_ess.contiguous()  # Ensure contiguous memory

            output_mls, sr = torchaudio.load(self.output_mls_files[idx])
            output_mls = output_mls.contiguous()  # Ensure contiguous memory
        else:
            input_ess, sr = librosa.load(self.input_ess_files[idx],sr=None)
            input_ess = torch.unsqueeze(torch.Tensor(input_ess),dim=0)
            input_ess = input_ess.contiguous()  # Ensure contiguous memory

            input_mls, sr = librosa.load(self.input_mls_files[idx],sr=None)
            input_mls = torch.unsqueeze(torch.Tensor(input_mls),dim=0)
            input_mls = input_mls.contiguous()  # Ensure contiguous memory

            output_ess, sr = librosa.load(self.output_ess_files[idx],sr=None)
            output_ess = torch.unsqueeze(torch.Tensor(output_ess),dim=0)
            output_ess = output_ess.contiguous()  # Ensure contiguous memory

            output_mls, sr = librosa.load(self.output_mls_files[idx],sr=None)
            output_mls = torch.unsqueeze(torch.Tensor(output_mls),dim=0)
            output_mls = output_mls.contiguous()  # Ensure contiguous memory

        return input_ess, input_mls, output_ess, output_mls

class SpectralMSE(torch.nn.Module):

    def __init__(self, alpha=1e-4):
        super(SpectralMSE, self).__init__()
        self.alpha = alpha

    def forward(self, y_hat, y):
        # Compute DFT domain signals
        nfft = y.shape[-1]
        dft_y_hat = torch.fft.fft(y_hat, n=nfft)
        dft_y = torch.fft.fft(y, n=nfft)

        # Apply frequency weights
        dft_y_hat = dft_y_hat
        dft_y = dft_y

        # Spectral MSE
        squared_mag_diff = torch.abs(dft_y_hat - dft_y)**2
        spectral_mse = torch.mean(squared_mag_diff)

        # Comparison plot
        # plt.plot(np.log(1e-5+np.abs(np.squeeze((torch.abs(dft_y)).detach().numpy()))))
        # plt.plot(np.log(1e-5+np.abs(np.squeeze((torch.abs(dft_y_hat)).detach().numpy()))))

        return spectral_mse
    
class responseSDR(torch.nn.Module):

    def __init__(self):
        super(responseSDR, self).__init__()

    def forward(self, h_hat, h):
        # Ensure h and h_hat are 1D arrays
        if h.ndim > 1:
            h = np.squeeze(h)
        if h_hat.ndim > 1:
            h_hat = np.squeeze(h_hat)

        # Zero-pad the shorter signal
        if len(h) > len(h_hat):
            h_hat = np.pad(h_hat, (0, len(h) - len(h_hat)), mode='constant')
        elif len(h_hat) > len(h):
            h = np.pad(h, (0, len(h_hat) - len(h)), mode='constant')

        # Compute SDR
        sdr = 10 * np.log10(np.linalg.norm(h, 2)**2 / np.linalg.norm(h - h_hat, 2)**2)
        return sdr
    
def normalize_signal(signal):
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    return (signal - mean_val) / std_val


## MAIN ##

def main():

    # Load the loss history from the .npy file
    LV_loss_history = np.load('models/LV/loss_history.npy')

    savemat('.\models\LV\LV_loss_history.mat', {'data': LV_loss_history})

    # Plot the loss history
    plt.figure(figsize=(7, 4))
    plt.plot(LV_loss_history, label='LV')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(r'Training Loss Evolution', fontsize=14)
    plt.xlim(0, len(LV_loss_history) - 1)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main() # aparentemente esto es necesario para windows