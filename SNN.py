import sys, os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
import snntorch.spikegen as spikegen
import numpy as np
import os
from tqdm import tqdm
from models.models import HAMM_SNN
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from monai.networks.layers import HilbertTransform
my_pc = False
if my_pc:
    import torchaudio
else:
    import librosa


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


""" class CustomLoss(torch.nn.Module):

    def __init__(self, alpha=1e-4):
        super(CustomLoss, self).__init__()
        self.alpha = alpha

    def generate_frequency_weights(self,n_pts):
        weights_half = torch.log(torch.linspace(1, 1e6, int(n_pts/2)))
        weights_half = weights_half/torch.max(weights_half)
        weights = torch.cat((weights_half,torch.flip(weights_half,dims=[-1])),dim=-1)
        return weights

    def forward(self, y_hat, y, ker1, ker2, ker3):
        # Compute DFT domain signals
        nfft = y.shape[-1]
        dft_y_hat = torch.fft.fft(y_hat, n=nfft)
        dft_y = torch.fft.fft(y, n=nfft)

        # Apply frequency weights
        freq_weights = self.generate_frequency_weights(y.shape[-1])
        #freq_weights = torch.ones(freq_weights.shape) # This eliminates the effect of frequency weighting
        dft_y_hat = dft_y_hat
        dft_y = dft_y

        # Spectral MSE
        squared_mag_diff = torch.log(torch.abs(dft_y_hat - dft_y)**2)
        spectral_mse = torch.mean(squared_mag_diff)

        # L1 regularization for the learnt kernels
        L1_reg = torch.norm(ker1,p=1) + torch.norm(ker2,p=1) + torch.norm(ker3,p=1)

        # Comparison plot
        # plt.plot(np.log(1e-5+np.abs(np.squeeze((torch.abs(dft_y)).detach().numpy()))))
        # plt.plot(np.log(1e-5+np.abs(np.squeeze((torch.abs(dft_y_hat)).detach().numpy()))))

        return spectral_mse #+ self.alpha*L1_reg """
    
class CustomLoss(torch.nn.Module):

    def __init__(self, alpha=1e-4, beta=1e-3):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ht = HilbertTransform()

    def generate_frequency_weights(self, n_pts):
        weights_half = torch.log(torch.linspace(1, 1e6, int(n_pts / 2)))
        weights_half = weights_half / torch.max(weights_half)
        weights = torch.cat((weights_half, torch.flip(weights_half, dims=[-1])), dim=-1)
        return weights

    def compute_analytic_components(self, signal):
        analytic_signal = self.ht(signal)
        envelope = torch.abs(analytic_signal)
        phase = torch.angle(analytic_signal)
        return envelope, phase

    def forward(self, y_hat, y, ker1, ker2, ker3):
        # Compute DFT domain signals
        nfft = y.shape[-1]
        dft_y_hat = torch.fft.fft(y_hat, n=nfft)
        dft_y = torch.fft.fft(y, n=nfft)

        # Apply frequency weights
        freq_weights = self.generate_frequency_weights(y.shape[-1])
        # freq_weights = torch.ones(freq_weights.shape) # This eliminates the effect of frequency weighting
        dft_y_hat = dft_y_hat
        dft_y = dft_y

        # Spectral MSE
        squared_mag_diff = torch.log(torch.abs(dft_y_hat - dft_y) ** 2)
        spectral_mse = torch.mean(squared_mag_diff)

        # L1 regularization for the learnt kernels
        L1_reg = torch.norm(ker1, p=1) + torch.norm(ker2, p=1) + torch.norm(ker3, p=1)

        # Compute envelopes
        envelope_y_hat, phase_y_hat = self.compute_analytic_components(y_hat)
        envelope_y, phase_y = self.compute_analytic_components(y)

        # Envelope MSE
        envelope_mse = F.mse_loss(envelope_y_hat, envelope_y)
        phase_mse = F.mse_loss(phase_y_hat, phase_y)

        return self.beta*spectral_mse + envelope_mse + (1/20)*phase_mse #+ self.alpha * L1_reg
    
def save_tensor_as_wav(tensor, filename, sample_rate=44100):
    # Convert the tensor to a numpy array
    array = tensor.cpu().detach().numpy()
    # Normalize the array to the range -1 to 1
    array = array / np.max(np.abs(array))
    # Scale to 16-bit integer values
    array_int24 = np.int32(array * 2147483647)  # Scale to 24-bit range
    # Save the array as a WAV file
    write(filename, sample_rate, array_int24)
        


## MAIN ##
def main():
    # User parameters
    use_snn = False
    ablation_model = True

    # Paths
    input_wavs_dir = './inputs_wav/'
    output_wavs_dir =  './outputs_wav/'

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    # Load data
    train_dataset = CustomDataset(input_wavs_dir,output_wavs_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)

    # Load model for training
    model = HAMM_SNN(use_snn=use_snn, ablation_model=ablation_model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = CustomLoss()

    # Train!
    epochs = 260
    loss_history = []

    # Define function to freeze/unfreeze branches
    def freeze_unfreeze_branches(model, branch_to_train=None):
        for name, param in model.named_parameters():
            if branch_to_train:
                if f'branch{branch_to_train}' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = True

    for epoch in range(epochs):
        total_loss = 0
        model.train()

        # Define branch training strategy
        if epoch < 100:
            freeze_unfreeze_branches(model, branch_to_train=1)
            train_order = 1
        elif epoch < 160:
            freeze_unfreeze_branches(model, branch_to_train=2)
            train_order = 2
        elif epoch < 200:
            freeze_unfreeze_branches(model, branch_to_train=3)
            train_order = 3
        else:
            freeze_unfreeze_branches(model)
            train_order = 0 # Train all

        if epoch % 2 == 0:
            for name, param in model.named_parameters():
                if 'branch' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:
            for name, param in model.named_parameters():
                if 'get_gains' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        for input_ess, input_mls, output_ess, output_mls in tqdm(train_dataloader):
            # ESS
            x = input_ess
            y = output_ess
            max_y_mod = torch.max(torch.abs(y))
            y = y / max_y_mod
            y = y - torch.mean(y) # Remove DC
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat, ker1, ker2, ker3 = model(x, train_order)
            loss = loss_fn(y_hat, y, ker1, ker2, ker3)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            # MLS
            x = input_mls
            y = output_mls
            max_y_mod = torch.max(torch.abs(y))
            y = y / max_y_mod
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat, ker1, ker2, ker3 = model(x, train_order)
            loss = loss_fn(y_hat, y, ker1, ker2, ker3)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_loss:.4f}")

    # Final forward to save the estimated kernels
    _, ker1, ker2, ker3 = model(x,max_y_mod)
    # Rotate to center
    ker1 = model.ifftshift(ker1)
    ker2 = model.ifftshift(ker2)
    ker3 = model.ifftshift(ker3)
    # Save
    if ablation_model:
        save_path_1 = os.path.join('kernels/LV','ker1.wav')
        save_path_2 = os.path.join('kernels/LV','ker2.wav')
        save_path_3 = os.path.join('kernels/LV','ker3.wav')
        save_path_model = os.path.join('models/LV','model.pth')
        save_path_loss = os.path.join('models/LV','loss_history.npy')
    elif use_snn:
        save_path_1 = os.path.join('kernels/SNN','ker1.wav')
        save_path_2 = os.path.join('kernels/SNN','ker2.wav')
        save_path_3 = os.path.join('kernels/SNN','ker3.wav')
        save_path_model = os.path.join('models/SNN','model.pth')
        save_path_loss = os.path.join('models/SNN','loss_history.npy')
    else:
        save_path_1 = os.path.join('kernels/NN','ker1.wav')
        save_path_2 = os.path.join('kernels/NN','ker2.wav')
        save_path_3 = os.path.join('kernels/NN','ker3.wav')
        save_path_model = os.path.join('models/NN','model.pth')
        save_path_loss = os.path.join('models/NN','loss_history.npy')

    # Save the identified kernels
    save_tensor_as_wav(ker1, save_path_1, sample_rate=144000)
    save_tensor_as_wav(ker2, save_path_2, sample_rate=144000)
    save_tensor_as_wav(ker3, save_path_3, sample_rate=144000)

    # Save the model
    torch.save(model.state_dict(), save_path_model)

    # Save the loss history
    np.save(save_path_loss, np.array(loss_history))

if __name__ == '__main__':
    main() # aparentemente esto es necesario para windows