import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
from models.models import HAMM_SNN
from SNN import CustomDataset, DataLoader
import torch
from tqdm import tqdm

def read_kernels(directory):

    wav_dict = {}

    for filename in os.listdir(directory):

        if filename.endswith('.wav'):

            filepath = os.path.join(directory, filename)
            sample_rate, data = wavfile.read(filepath)
            var_name = os.path.splitext(filename)[0]
            wav_dict[var_name] = data

    return wav_dict, sample_rate

class CustomErrorMetric(torch.nn.Module):

    def __init__(self, alpha=1e-4):
        super(CustomErrorMetric, self).__init__()
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


# Load the loss history from the .npy file
SNN_loss_history = np.load('models/SNN/loss_history.npy')
NN_loss_history = np.load('models/NN/loss_history.npy')

# Load kernels
NN_dir = os.path.join('kernels/NN')
SNN_dir = os.path.join('kernels/SNN')
gt_dir = os.path.join('kernels/ground_truth')
ker_NN_dict, sr = read_kernels(NN_dir)
ker_SNN_dict, sr = read_kernels(SNN_dir)
ker_gt_dict, sr = read_kernels(gt_dir)

# Plot the loss history
plt.figure(1)
plt.plot(NN_loss_history, label='NN')
plt.plot(SNN_loss_history, label='SNN')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Evolution')
plt.legend()
plt.show()

# Plot kernels
# Create a figure and a 1x3 subplot grid
fig, axs = plt.subplots(3, 3, figsize=(15, 10))
# Iterate over the dictionary items and plot each signal
for ax, (title, signal) in zip(axs[0], ker_gt_dict.items()):
    ax.plot(signal)
    ax.set_title('True ' + title)
    ax.set_xlabel('X-axis label')
    ax.set_ylabel('Y-axis label')
for ax, (title, signal) in zip(axs[1], ker_NN_dict.items()):
    ax.plot(signal)
    ax.set_title('NN: ' + title)
    ax.set_xlabel('X-axis label')
    ax.set_ylabel('Y-axis label')
for ax, (title, signal) in zip(axs[2], ker_SNN_dict.items()):
    ax.plot(signal)
    ax.set_title('SNN: ' + title)
    ax.set_xlabel('X-axis label')
    ax.set_ylabel('Y-axis label')

plt.tight_layout()
plt.show()

# Forward all signals in dataset and compute MSE...

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Error metric (train loss)
error_metric = CustomErrorMetric()

# Load data
input_wavs_dir = './inputs_wav/'
output_wavs_dir =  './outputs_wav/'
train_dataset = CustomDataset(input_wavs_dir,output_wavs_dir)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)

# Load trained models
NN_model_path = os.path.join('models/NN/model.pth')
model_NN = HAMM_SNN(use_snn=False)
model_NN.load_state_dict(torch.load(NN_model_path, map_location=torch.device('cpu')))
model_NN.eval()

SNN_model_path = os.path.join('models/SNN/model.pth')
model_SNN = HAMM_SNN(use_snn=True)
model_SNN.load_state_dict(torch.load(SNN_model_path, map_location=torch.device('cpu')))
model_SNN.eval()

error_NN_ess = []
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
    y_hat_NN_ess, _, _, _ = model_NN(x_ess,max_y_mod_ess)
    # Compute loss
    error_NN_ess.append(error_metric(y_hat_NN_ess, y_ess))

error_NN_ess = np.array(error_NN_ess)
mu_NN_ess = np.mean(error_NN_ess)
std_NN_ess = np.std(error_NN_ess)



hey=0