


my_pc = True

import torch
from torch.utils.data import Dataset, DataLoader
if my_pc:
    import torchaudio
else:
    import librosa
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

    # Paths
    input_wavs_dir = './inputs_wav/'
    output_wavs_dir =  './outputs_wav/'

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    # Load data
    train_dataset = CustomDataset(input_wavs_dir,output_wavs_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)

    # Load model for training
    model = HAMM_SNN(use_snn=use_snn)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = CustomLoss()

    # Train!
    epochs = 100
    branch_idx = 1
    loss_history = []
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

            # I/O signals: normalized to [-1 1]
            x =  input_ess
            y = output_ess
            max_y_mod = torch.max(torch.abs(y))
            y = y / max_y_mod
            # Signals to device
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            # Forward pass
            y_hat, ker1, ker2, ker3 = model(x,max_y_mod)

            # Compute loss
            loss = loss_fn(y_hat, y, ker1, ker2, ker3) # Where should this loss be defined
            total_loss += loss.item()

            # Backpropagate
            loss.backward()
            # Check model.branchn.weight -> Gradient and whether it changes or not when performing optimizer.step()
            optimizer.step()
            
        avg_loss = total_loss / len(train_dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")

    # Final forward to save the estimated kernels
    _, ker1, ker2, ker3 = model(x,max_y_mod)
    # Rotate to center
    ker1 = model.ifftshift(ker1)
    ker2 = model.ifftshift(ker2)
    ker3 = model.ifftshift(ker3)
    # Save
    if use_snn:
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