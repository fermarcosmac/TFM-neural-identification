# TODO

# Something important is to define the maximum memory of the kernel diagonals (length of t-mode/batch size of data tensor)
# While developing, I will assume that I consider 512 samples (input time steps)

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
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
        input_ess, sr = torchaudio.load(self.input_ess_files[idx])
        input_ess = input_ess.contiguous()  # Ensure contiguous memory

        input_mls, sr = torchaudio.load(self.input_mls_files[idx])
        input_mls = input_mls.contiguous()  # Ensure contiguous memory

        output_ess, sr = torchaudio.load(self.output_ess_files[idx])
        output_ess = output_ess.contiguous()  # Ensure contiguous memory

        output_mls, sr = torchaudio.load(self.output_mls_files[idx])
        output_mls = output_mls.contiguous()  # Ensure contiguous memory

        return input_ess, input_mls, output_ess, output_mls


class CustomLoss(torch.nn.Module):

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
        squared_mag_diff = torch.abs(dft_y_hat - dft_y)**2
        spectral_mse = torch.mean(squared_mag_diff)

        # L1 regularization for the learnt kernels
        L1_reg = torch.norm(ker1,p=1) + torch.norm(ker2,p=1) + torch.norm(ker3,p=1)

        # Comparison plot
        # plt.plot(np.log(1e-5+np.abs(np.squeeze((torch.abs(dft_y)).detach().numpy()))))
        # plt.plot(np.log(1e-5+np.abs(np.squeeze((torch.abs(dft_y_hat)).detach().numpy()))))

        return spectral_mse #+ self.alpha*L1_reg
        




## MAIN ##
def main():
    # Paths
    input_wavs_dir = './inputs_wav/'
    output_wavs_dir =  './outputs_wav/'

    # Load data
    train_dataset = CustomDataset(input_wavs_dir,output_wavs_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)

    # Load model for training
    model = HAMM_SNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = CustomLoss()

    # Train! -> Original
    epochs = 100
    branch_idx = 1
    for epoch in range(epochs):

        total_loss = 0
        model.train()

        # Narendra-Gallman inspiration
        if epoch % 2 == 0:  # Freeze branches every 2 epochs
            for name, param in model.named_parameters():
                if 'branch' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:  # Freeze gains in every other epoch
            for name, param in model.named_parameters():
                if 'get_gains' in name:
                    param.requires_grad = False
                # Uncomment to freeze all branches but one consecutively (through epochs)
                #elif f'branch{branch_idx}' not in name:
                    #param.requires_grad = False
                else:
                    param.requires_grad = True
            # Update branch index (branch to update in next iteration) 
            branch_idx = branch_idx % 3 + 1

        for input_ess, input_mls, output_ess, output_mls in tqdm(train_dataloader):

            if epoch % 2 != 0:
                hey=0
            # I/O signals: normalized to [-1 1]
            x =  input_ess
            y = output_ess
            max_y_mod = torch.max(torch.abs(y))
            y = y / max_y_mod

            optimizer.zero_grad()

            # Forward pass
            y_hat, ker1, ker2, ker3 = model(x,max_y_mod)

            if epoch == 10 or epoch == 19:
                hey=0

            # Compute loss
            loss = loss_fn(y_hat, y, ker1, ker2, ker3) # Where should this loss be defined
            total_loss += loss.item()

            # Backpropagate
            loss.backward()
            # Check model.branchn.weight -> Gradient and whether it changes or not when performing optimizer.step()
            optimizer.step()
            


        avg_loss = total_loss / len(train_dataloader)


        # Plot estimated output evolution
        #if epoch == 1:
        #    plt.plot(np.squeeze((y).detach().numpy()))
        #plt.plot(np.squeeze((y_hat).detach().numpy()))


        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")

if __name__ == '__main__':
    main() # aparentemente esto es necesario para windows