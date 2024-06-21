# Imports
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
from models.models import HAMM_SNN
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
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
    SNN_loss_history = np.load('models/SNN/loss_history.npy')
    NN_loss_history = np.load('models/NN/loss_history.npy')
    LV_loss_history = np.load('models/LV/loss_history.npy')

    # Plot the loss history
    plt.figure(figsize=(7,4))
    plt.plot(NN_loss_history, label='NN')
    plt.plot(SNN_loss_history, label='SNN')
    plt.plot(LV_loss_history, label='LV')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Evolution')
    plt.legend()
    plt.tight_layout()
    plt.show()


    ###### KERNEL COMPARISON ######

    # Load kernels
    NN_dir = os.path.join('kernels/NN')
    SNN_dir = os.path.join('kernels/SNN')
    LV_dir = os.path.join('kernels/LV')
    gt_dir = os.path.join('kernels/ground_truth')
    ker_NN_dict, sr = read_kernels(NN_dir)
    ker_SNN_dict, sr = read_kernels(SNN_dir)
    ker_LV_dict, sr = read_kernels(LV_dir)
    ker_gt_dict, sr = read_kernels(gt_dir)


    """ # Plot kernels """
    """ # Create a figure and a 1x3 subplot grid """
    """ fig, axs = plt.subplots(3, 3, figsize=(7, 7)) """
    """ # Iterate over the dictionary items and plot each signal """
    """ for ax, (title, signal) in zip(axs[0], ker_gt_dict.items()): """
    """     ax.plot(signal) """
    """     ax.set_title('True ' + title) """
    """     ax.set_xlabel('X-axis label') """
    """     ax.set_ylabel('Y-axis label') """
    """ for ax, (title, signal) in zip(axs[1], ker_NN_dict.items()): """
    """     ax.plot(signal) """
    """     ax.set_title('NN: ' + title) """
    """     ax.set_xlabel('X-axis label') """
    """     ax.set_ylabel('Y-axis label') """
    """ for ax, (title, signal) in zip(axs[2], ker_SNN_dict.items()): """
    """     ax.plot(signal) """
    """     ax.set_title('SNN: ' + title) """
    """     ax.set_xlabel('X-axis label') """
    """     ax.set_ylabel('Y-axis label') """

    """ plt.tight_layout() """
    """ plt.show() """


    # Initialize error dict and metric
    sdr_results = {}
    response_sdr = responseSDR()

    # Loop over each key and compute the errors
    for key in ker_gt_dict:
        true_signal = ker_gt_dict[key]
        estimated_ker_NN = ker_NN_dict[key]
        estimated_ker_SNN = ker_SNN_dict[key]
        estimated_ker_LV = ker_LV_dict[key]

        sdr_NN = response_sdr(true_signal, estimated_ker_NN)
        sdr_SNN = response_sdr(true_signal, estimated_ker_SNN)
        sdr_LV = response_sdr(true_signal, estimated_ker_LV)

        sdr_results[key] = {
            'NN': sdr_NN,
            'SNN': sdr_SNN,
            'LV': sdr_LV
        }

    # Print the error results
    for key, errors in sdr_results.items():
        print(f"SDR for {key}:")
        for method, error in errors.items():
            print(f"  {method}: {error}")

    # Plot the signals for visualization
    fig, axs = plt.subplots(3, 3, figsize=(6, 6))

    for i, key in enumerate(ker_gt_dict):
        true_signal = ker_gt_dict[key]
        estimated_ker_NN = ker_NN_dict[key]
        estimated_ker_SNN = ker_SNN_dict[key]
        estimated_ker_LV = ker_LV_dict[key]

        # Normalize the signals
        true_signal = normalize_signal(true_signal)
        estimated_ker_NN = normalize_signal(estimated_ker_NN)
        estimated_ker_SNN = normalize_signal(estimated_ker_SNN)
        estimated_ker_LV = normalize_signal(estimated_ker_LV)

        # Plot true vs NN
        axs[i, 0].plot(true_signal, label='True')
        axs[i, 0].plot(estimated_ker_NN, label='NN')
        axs[i, 0].set_title(f'True vs NN for {key}')
        axs[i, 0].legend()

        # Plot true vs SNN
        axs[i, 1].plot(true_signal, label='True')
        axs[i, 1].plot(estimated_ker_SNN, label='SNN')
        axs[i, 1].set_title(f'True vs  SNN for {key}')
        axs[i, 1].legend()

        # Plot true vs LV
        axs[i, 2].plot(true_signal, label='True')
        axs[i, 2].plot(estimated_ker_LV, label='LV')
        axs[i, 2].set_title(f'True vs  LV for {key}')
        axs[i, 2].legend()

    plt.tight_layout()
    plt.show() 



    ####### OUTPUT COMPARISON #######

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Error metric (spectral MSE)
    spectral_mse = SpectralMSE()

    # Load data
    input_wavs_dir = './inputs_wav/'
    output_wavs_dir =  './outputs_wav/'
    train_dataset = CustomDataset(input_wavs_dir,output_wavs_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)

    # Load trained models
    NN_model_path = os.path.join('models/NN/model.pth')
    model_NN = HAMM_SNN(use_snn=False,ablation_model=False)
    model_NN.load_state_dict(torch.load(NN_model_path, map_location=torch.device('cpu')))
    model_NN.eval()

    SNN_model_path = os.path.join('models/SNN/model.pth')
    model_SNN = HAMM_SNN(use_snn=True, ablation_model=False)
    model_SNN.load_state_dict(torch.load(SNN_model_path, map_location=torch.device('cpu')))
    model_SNN.eval()

    LV_model_path = os.path.join('models/LV/model.pth')
    model_LV = HAMM_SNN(use_snn=False, ablation_model=True)
    model_LV.load_state_dict(torch.load(LV_model_path, map_location=torch.device('cpu')))
    model_LV.eval()

    error_NN_ess = []
    error_NN_mls = []
    error_SNN_ess = []
    error_SNN_mls = []
    error_LV_ess = []
    error_LV_mls = []
    plot_flag = True

    for input_ess, input_mls, output_ess, output_mls in tqdm(train_dataloader):
        # I/O signals: normalized to [-1 1]
        x_ess =  input_ess
        y_ess = output_ess
        x_mls =  input_mls
        y_mls = output_mls
        max_y_mod_ess = torch.max(torch.abs(y_ess))
        y_ess = y_ess / max_y_mod_ess
        max_y_mod_mls = torch.max(torch.abs(y_mls))
        y_mls = y_mls / max_y_mod_mls
        # Signals to device
        x_ess = x_ess.to(device)
        y_ess = y_ess.to(device)
        x_mls = x_mls.to(device)
        y_mls = y_mls.to(device)
        # Forward pass
        y_hat_NN_ess, _, _, _ = model_NN(x_ess)
        y_hat_NN_mls, _, _, _ = model_NN(x_mls)
        y_hat_SNN_ess, _, _, _ = model_SNN(x_ess)
        y_hat_SNN_mls, _, _, _ = model_SNN(x_mls)
        y_hat_LV_ess, _, _, _ = model_LV(x_ess)
        y_hat_LV_mls, _, _, _ = model_LV(x_mls)
        # Compute loss
        error_NN_ess.append(spectral_mse(y_hat_NN_ess, y_ess))
        error_NN_mls.append(spectral_mse(y_hat_NN_mls, y_mls))
        error_SNN_ess.append(spectral_mse(y_hat_SNN_ess, y_ess))
        error_SNN_mls.append(spectral_mse(y_hat_SNN_mls, y_mls))
        error_LV_ess.append(spectral_mse(y_hat_LV_ess, y_ess))
        error_LV_mls.append(spectral_mse(y_hat_LV_mls, y_mls))

        # Plot one example
        if plot_flag:
            ## Time domain signals ##
            fig, axs = plt.subplots(3, 2, figsize=(7, 5))
            # NN with ESS input
            axs[0, 0].plot(np.squeeze(y_ess.cpu().numpy())[140000:150000], label='True')
            axs[0, 0].plot(np.squeeze(y_hat_NN_ess.detach().cpu().numpy())[140000:150000], label='Predicted')
            axs[0, 0].set_title('NN with ESS input')
            axs[0, 0].legend()

            # NN with MLS input
            axs[0, 1].plot(np.squeeze(y_mls.cpu().numpy())[140000:140300], label='True')
            axs[0, 1].plot(np.squeeze(y_hat_NN_mls.detach().cpu().numpy())[140000:140300], label='Predicted')
            axs[0, 1].set_title('NN with MLS input')
            axs[0, 1].legend()

            # SNN with ESS input
            axs[1, 0].plot(np.squeeze(y_ess.cpu().numpy())[140000:150000], label='True')
            axs[1, 0].plot(np.squeeze(y_hat_SNN_ess.detach().cpu().numpy())[140000:150000], label='Predicted')
            axs[1, 0].set_title('SNN with ESS input')
            axs[1, 0].legend()

            # SNN with MLS input
            axs[1, 1].plot(np.squeeze(y_mls.cpu().numpy())[140000:140300], label='True')
            axs[1, 1].plot(np.squeeze(y_hat_SNN_mls.detach().cpu().numpy())[140000:140300], label='Predicted')
            axs[1, 1].set_title('SNN with MLS input')
            axs[1, 1].legend()

            # LV with ESS input
            axs[2, 0].plot(np.squeeze(y_ess.cpu().numpy())[140000:150000], label='True')
            axs[2, 0].plot(np.squeeze(y_hat_LV_ess.detach().cpu().numpy())[140000:150000], label='Predicted')
            axs[2, 0].set_title('LV with ESS input')
            axs[2, 0].legend()

            # LV with MLS input
            axs[2, 1].plot(np.squeeze(y_mls.cpu().numpy())[140000:140300], label='True')
            axs[2, 1].plot(np.squeeze(y_hat_LV_mls.detach().cpu().numpy())[140000:140300], label='Predicted')
            axs[2, 1].set_title('LV with MLS input')
            axs[2, 1].legend()

            plt.tight_layout()
            plt.show()

            ## Frequency domain signals ##
            fig, axs = plt.subplots(3, 2, figsize=(7, 5))
            # NN with ESS input
            plot_dft(axs[0, 0], y_ess, y_hat_NN_ess, 'NN with ESS input', sr)

            # NN with MLS input
            plot_dft(axs[0, 1], y_mls, y_hat_NN_mls, 'NN with MLS input', sr)

            # SNN with ESS input
            plot_dft(axs[1, 0], y_ess, y_hat_SNN_ess, 'SNN with ESS input', sr)

            # SNN with MLS input
            plot_dft(axs[1, 1], y_mls, y_hat_SNN_mls, 'SNN with MLS input', sr)

            # LV with ESS input
            plot_dft(axs[2, 0], y_ess, y_hat_LV_ess, 'LV with ESS input', sr)

            # LV with MLS input
            plot_dft(axs[2, 1], y_mls, y_hat_LV_mls, 'LV with MLS input', sr)

            plt.tight_layout()
            plt.show()


            plot_flag = False

    # ESS -> NN
    error_NN_ess = np.array([element.detach().numpy() for element in error_NN_ess])
    mu_NN_ess = np.mean(error_NN_ess)
    std_NN_ess = np.std(error_NN_ess)

    # MLS -> NN
    error_NN_mls = np.array([element.detach().numpy() for element in error_NN_mls])
    mu_NN_mls = np.mean(error_NN_mls)
    std_NN_mls = np.std(error_NN_mls)

    # ESS -> SNN
    error_SNN_ess = np.array([element.detach().numpy() for element in error_SNN_ess])
    mu_SNN_ess = np.mean(error_SNN_ess)
    std_SNN_ess = np.std(error_SNN_ess)

    # MLS -> SNN
    error_SNN_mls = np.array([element.detach().numpy() for element in error_SNN_mls])
    mu_SNN_mls = np.mean(error_SNN_mls)
    std_SNN_mls = np.std(error_SNN_mls)

    # ESS -> LV
    error_LV_ess = np.array([element.detach().numpy() for element in error_LV_ess])
    mu_LV_ess = np.mean(error_LV_ess)
    std_LV_ess = np.std(error_LV_ess)

    # MLS -> LV
    error_LV_mls = np.array([element.detach().numpy() for element in error_LV_mls])
    mu_LV_mls = np.mean(error_LV_mls)
    std_LV_mls = np.std(error_LV_mls)

    print("Results:")
    print("========")
    print("ESS -> NN:")
    print(f"Mean Spectral Error: {mu_NN_ess:.4f}")
    print(f"Standard Deviation: {std_NN_ess:.4f}")
    print("========")
    print("MLS -> NN:")
    print(f"Mean Spectral Error: {mu_NN_mls:.4f}")
    print(f"Standard Deviation: {std_NN_mls:.4f}")
    print("========")
    print("ESS -> SNN:")
    print(f"Mean Spectral Error: {mu_SNN_ess:.4f}")
    print(f"Standard Deviation: {std_SNN_ess:.4f}")
    print("========")
    print("MLS -> SNN:")
    print(f"Mean Spectral Error: {mu_SNN_mls:.4f}")
    print(f"Standard Deviation: {std_SNN_mls:.4f}")
    print("========")
    print("ESS -> LV:")
    print(f"Mean Spectral Error: {mu_LV_ess:.4f}")
    print(f"Standard Deviation: {std_LV_ess:.4f}")
    print("========")
    print("MLS -> LV:")
    print(f"Mean Spectral Error: {mu_LV_mls:.4f}")
    print(f"Standard Deviation: {std_LV_mls:.4f}")


if __name__ == '__main__':
    main() # aparentemente esto es necesario para windows