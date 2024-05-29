import torch
import torch.nn.functional as F
from torch import nn
import snntorch as snn
from snntorch import surrogate
import snntorch.spikegen as spikegen
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# See kernel estimates
# plt.plot(np.squeeze((ker1).detach().numpy()))
# plt.plot(np.squeeze((ker2).detach().numpy()))
# plt.plot(np.squeeze((ker3).detach().numpy()))
# plt.show()

ker_length  = 19*2 # Number of samples in identified kernels (divided by 4)
num_ff = 20 # Number of Fourier Features
x_length = 570000
 

# Branch SNN
class BranchSNN(nn.Module):

    def __init__(self, input_size: int = num_ff*ker_length, neurons_per_layer: int = 64, **kwargs):
        super(BranchSNN, self).__init__()
        # Define Network architecture
        spike_grad = surrogate.fast_sigmoid(slope=25) # For backpropagation
        # Common layers
        self.fc1  = nn.Linear(in_features=input_size, out_features=neurons_per_layer)
        self.lif1 = snn.Leaky(beta=0.99, threshold=0, learn_beta=False, reset_mechanism='subtract', spike_grad=spike_grad)
        self.fc2  = nn.Linear(in_features=neurons_per_layer, out_features=neurons_per_layer)
        self.lif2 = snn.Leaky(beta=0.99, threshold=0, learn_beta=False, reset_mechanism='subtract', spike_grad=spike_grad)
        self.fc3  = nn.Linear(in_features=neurons_per_layer, out_features=neurons_per_layer)
        self.lif3 = snn.Leaky(beta=0.99, threshold=0, learn_beta=False, reset_mechanism='subtract', spike_grad=spike_grad)
        # Modulus prediction layers
        self.fc4  = nn.Linear(in_features=neurons_per_layer, out_features=neurons_per_layer)
        self.lif4 = snn.Leaky(beta=0.99, threshold=0.2, learn_beta=False, reset_mechanism='subtract', spike_grad=spike_grad)
        self.fc5  = nn.Linear(in_features=neurons_per_layer, out_features=neurons_per_layer)
        self.lif5 = snn.Leaky(beta=0.99, threshold=0.2, learn_beta=False, reset_mechanism='subtract', spike_grad=spike_grad)
        self.fc6  = nn.Linear(in_features=neurons_per_layer, out_features=1)
        self.lif6 = snn.Leaky(beta=0.99, threshold=0.2, learn_beta=False, reset_mechanism='subtract', spike_grad=spike_grad, output=True)
        # Phase prediction layers
        self.fc7  = nn.Linear(in_features=neurons_per_layer, out_features=neurons_per_layer)
        self.lif7 = snn.Leaky(beta=0.99, threshold=0.2, learn_beta=False, reset_mechanism='subtract', spike_grad=spike_grad)
        self.fc8  = nn.Linear(in_features=neurons_per_layer, out_features=neurons_per_layer)
        self.lif8 = snn.Leaky(beta=0.99, threshold=0.2, learn_beta=False, reset_mechanism='subtract', spike_grad=spike_grad)
        self.fc9  = nn.Linear(in_features=neurons_per_layer, out_features=1)
        self.lif9 = snn.Leaky(beta=0.99, threshold=0.2, learn_beta=False, reset_mechanism='subtract', spike_grad=spike_grad, output=True)
        # Activations
        self.relu  = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # Set requires_grad=True for all parameters
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x, **kwargs):
        """
        Forward pass of the branch.

        Args:
            x (torch.Tensor): The Fourier Feature matrix of the time vector (t), represented through rate coding
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        #x = x.view(-1, num_ff*ker_length)
        x = x.float()
        x = x.to(device)

        mem1 = self.lif1.init_leaky() # reset_mem()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        mem6 = self.lif6.init_leaky()
        mem7 = self.lif7.init_leaky()
        mem8 = self.lif8.init_leaky()
        mem9 = self.lif9.init_leaky()

        # Common layers
        x = self.fc1(x)
        x, mem1 = self.lif1(x, mem1)
        x = self.fc2(x)
        x, mem2 = self.lif2(x, mem2)
        x = self.fc3(x)
        x, mem3 = self.lif3(x, mem3)
        # Modulus prediction
        mod = self.fc4(x)
        mod, mem4 = self.lif4(mod, mem4)
        mod = self.fc5(mod)
        mod, mem5 = self.lif5(mod, mem5)
        mod = self.fc6(mod)
        mod, mem6 = self.lif6(mod, mem6)
        # Phase prediction
        pha = self.fc7(x)
        pha, mem7 = self.lif7(pha, mem7)
        pha = self.fc8(pha)
        pha, mem8 = self.lif8(pha, mem8)
        pha = self.fc9(pha)
        pha, mem9 = self.lif9(pha, mem9)

        mod = torch.squeeze(mod)
        pha = -2*torch.pi*self.sigmoid(torch.squeeze(pha))

        return mod, pha
    
# Branch NN
class BranchNN(nn.Module):

    def __init__(self, input_size: int = num_ff, neurons_per_layer: int = 512, **kwargs):
        super(BranchNN, self).__init__()
        # Define Network architecture
        # Common layers
        self.fc1  = nn.Linear(in_features=input_size, out_features=neurons_per_layer)
        self.fc2  = nn.Linear(in_features=neurons_per_layer, out_features=neurons_per_layer)
        self.fc3  = nn.Linear(in_features=neurons_per_layer, out_features=neurons_per_layer)
        # Modulus prediction layers
        self.fc4  = nn.Linear(in_features=neurons_per_layer, out_features=neurons_per_layer)
        self.fc5  = nn.Linear(in_features=neurons_per_layer, out_features=neurons_per_layer)
        self.fc6  = nn.Linear(in_features=neurons_per_layer, out_features=1)
        # Phase prediction layers
        self.fc7  = nn.Linear(in_features=neurons_per_layer, out_features=neurons_per_layer)
        self.fc8  = nn.Linear(in_features=neurons_per_layer, out_features=neurons_per_layer)
        self.fc9  = nn.Linear(in_features=neurons_per_layer, out_features=1)
        
        # Activations
        self.relu  = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # Set requires_grad=True for all parameters
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x, **kwargs):
        """
        Forward pass of the branch.

        Args:
            x (torch.Tensor): The Fourier Feature matrix of the time vector (t), represented through rate coding
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        #x = x.view(-1, num_ff)
        x = x.float()
        x = x.to(device)

        # Common layers -> I saw that some branches were only training either modulus or phase
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        # Modulus prediction
        mod = self.relu(self.fc4(x)) 
        mod = self.relu(self.fc5(mod))
        mod = self.sigmoid(self.fc6(mod))               # log-modulus ranges (-inf 0]
        mod = torch.exp(mod)                            # Network will predict the log-modulus and output the modulus
        # Phase prediction
        pha = self.relu(self.fc7(x))
        pha = self.relu(self.fc8(pha))
        pha = -2*torch.pi * self.sigmoid(self.fc9(pha))       # Predict inter-sample phase differences [-2pi 0]

        mod = torch.squeeze(mod)
        pha = torch.squeeze(pha)

        return mod, pha
    

# Learnable gain vector
class LearnableGains(nn.Module):
    def __init__(self):
        super(LearnableGains, self).__init__()
        self.gains = nn.Parameter(torch.randn(3))   # Initialize learnable gains vector

    def forward(self):
        return torch.abs(self.gains)  # Nonnegative gains
    

class LearnableDFT(nn.Module):
    def __init__(self, length):
        super(LearnableDFT, self).__init__()
        self.modulus = nn.Parameter(torch.abs(torch.randn(length)))
        self.phase = nn.Parameter(torch.randn(length)-2)

    def forward(self,x):
        # I include an input (x) for consistency with the other models, even though these are simply learnable vectors
        return self.modulus, self.phase
    

# Define the HAMM_SNN2 class
class HAMM_SNN(nn.Module):
    def __init__(self, use_snn: bool = False, ablation_model: bool = False, input_size: int = num_ff):
        super(HAMM_SNN, self).__init__()
        # Learnable attributes
        if ablation_model:
            self.branch1 = LearnableDFT(length=2*ker_length)
            self.branch2 = LearnableDFT(length=2*ker_length)
            self.branch3 = LearnableDFT(length=2*ker_length)
        elif use_snn:
            self.branch1 = BranchSNN(input_size=input_size)
            self.branch2 = BranchSNN(input_size=input_size)
            self.branch3 = BranchSNN(input_size=input_size)
        else:
            self.branch1 = BranchNN(input_size=input_size)
            self.branch2 = BranchNN(input_size=input_size)
            self.branch3 = BranchNN(input_size=input_size)
        self.get_gains   = LearnableGains()
        # Other attributes
        self.use_snn = use_snn
        self.input_size = input_size
        self.sigmoid = nn.Sigmoid()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    def forward(self, x, train_order: int = 0):

        # Invariant fourier features
        t = np.linspace(start=0,stop=2,num=ker_length*2)
        ff = self.compute_fourier_features(t,num_features=num_ff)
        ff = torch.transpose(torch.from_numpy(ff),0,1)

        if self.use_snn:
            # Convert rates to spike trains
            num_steps = int(1000 * 1e-3 * 5)
            spike_ff = spikegen.rate(ff, num_steps=num_steps)
            # Forward
            Ker1_mod, Ker1_pha_diff = self.branch1(spike_ff)
            Ker2_mod, Ker2_pha_diff = self.branch2(spike_ff)
            Ker3_mod, Ker3_pha_diff = self.branch3(spike_ff)
            # Decoding
            Ker1_mod = torch.mean(Ker1_mod,dim=[0])
            Ker1_pha_diff = torch.mean(Ker1_pha_diff,dim=[0])
            Ker1_mod = torch.exp(self.sigmoid(Ker1_mod))
            Ker2_mod = torch.mean(Ker2_mod,dim=[0])
            Ker2_pha_diff = torch.mean(Ker2_pha_diff,dim=[0])
            Ker2_mod = torch.exp(self.sigmoid(Ker2_mod))
            Ker3_mod = torch.mean(Ker3_mod,dim=[0])
            Ker3_pha_diff = torch.mean(Ker3_pha_diff,dim=[0])
            Ker3_mod = torch.exp(self.sigmoid(Ker3_mod))
        else:
            # Forward
            Ker1_mod, Ker1_pha_diff = self.branch1(ff)
            Ker2_mod, Ker2_pha_diff = self.branch2(ff)
            Ker3_mod, Ker3_pha_diff = self.branch3(ff)

        # Let us predict the phase differences between samples and here build back the unwrapped phase information
        Ker1_pha = torch.cumsum(Ker1_pha_diff, dim=0)
        Ker2_pha = torch.cumsum(Ker2_pha_diff, dim=0)
        Ker3_pha = torch.cumsum(Ker3_pha_diff, dim=0)

        # Extend spectra in Hermitian form (so time-domain equivalent is real-valued)
        KER1_mod_extended, KER1_pha_extended = self.hermitian_extend(Ker1_mod, Ker1_pha)
        KER2_mod_extended, KER2_pha_extended = self.hermitian_extend(Ker2_mod, Ker2_pha)
        KER3_mod_extended, KER3_pha_extended = self.hermitian_extend(Ker3_mod, Ker3_pha)

        # Go from polar to cartesian form (Euler's formula)
        KER1_real_extended = KER1_mod_extended * torch.cos(KER1_pha_extended)
        KER2_real_extended = KER2_mod_extended * torch.cos(KER2_pha_extended)
        KER3_real_extended = KER3_mod_extended * torch.cos(KER3_pha_extended)
        KER1_imag_extended = KER1_mod_extended * torch.sin(KER1_pha_extended)
        KER2_imag_extended = KER2_mod_extended * torch.sin(KER2_pha_extended)
        KER3_imag_extended = KER3_mod_extended * torch.sin(KER3_pha_extended)

        # Build complex DFT spectra
        KER1 = torch.complex(KER1_real_extended, KER1_imag_extended)
        KER2 = torch.complex(KER2_real_extended, KER2_imag_extended)
        KER3 = torch.complex(KER3_real_extended, KER3_imag_extended)
        
        
        # IDFT for time domain representation
        ker1 = torch.real(torch.fft.ifft(KER1))
        ker2 = torch.real(torch.fft.ifft(KER2))
        ker3 = torch.real(torch.fft.ifft(KER3))

        # This is necesary for vanilla branch NNs
        ker1 = torch.squeeze(ker1)
        ker2 = torch.squeeze(ker2)
        ker3 = torch.squeeze(ker3)

        # Rate decoding:
        # Kernels must span [-1 1]
        #ker1 = 2*((torch.sum(ker1, dim=0)/num_steps) - 0.5)
        #ker2 = 2*((torch.sum(ker2, dim=0)/num_steps) - 0.5)
        #ker3 = 2*((torch.sum(ker3, dim=0)/num_steps) - 0.5)
        # SNNS ARE SATURATING AT -1 OR +1. I DEFINITELY NEED AN EXTERNAL LEARNABLE GAIN

        # Set gains to activate selected branches
        gains = 0.01 * self.get_gains()
        g1 = 0
        g2 = 0
        g3 = 0
        match train_order:
            case 0:
                g1   = gains[0]
                g2   = gains[1]
                g3   = gains[2]
            case 1:
                g1   = gains[0]
            case 2:
                g2   = gains[1]
            case 3:
                g3   = gains[2]

        # Hammerstein logic -> went back to previous approach but assigning self.ker3_conv.weight.data (without the [0])
        x_branch1 = F.conv1d(x,   weight=ker1.unsqueeze(0).unsqueeze(0))
        x_branch2 = F.conv1d(x**2,weight=ker2.unsqueeze(0).unsqueeze(0))
        x_branch3 = F.conv1d(x**3,weight=ker3.unsqueeze(0).unsqueeze(0))

        output = g1*x_branch1 + g2*x_branch2 + g3*x_branch3
        # Pad the output tensor with zeros at the end of the third mode
        pad_length = x.size(2) - output.size(2)
        output = F.pad(output, (0, pad_length), mode='constant', value=0)
        # Normalize output
        # max_output_mod = torch.max(torch.abs(output))
        # output = output / max_output_mod

        return output, ker1, ker2, ker3

    # Compute Fourier feature matrix for an input vector of time indices (t)
    def compute_fourier_features(self,t,num_features):
        l = np.array(range(int(num_features/2)))
        a = (2 ** np.reshape(l,(len(l),1))) * np.pi
        b = np.reshape(t,(1,len(t)))
        feature_matrix = np.concatenate((np.sin(a @ b),np.cos(a @ b)),0)

        return feature_matrix
    
    def hermitian_reflect_spectrum(self,mod_part, pha_part):
        N = mod_part.shape[-1]
        reflected_mod =  torch.flip(mod_part[..., 1:N-1], dims=(-1,))
        reflected_pha = -torch.flip(pha_part[..., 1:N-1], dims=(-1,))
        return reflected_mod, reflected_pha

    def hermitian_extend(self,Ker_mod, Ker_pha):
        reflected_mod, reflected_pha = self.hermitian_reflect_spectrum(Ker_mod, Ker_pha)
        extended_mod = torch.cat((Ker_mod, reflected_mod), dim=-1)
        extended_pha = torch.cat((Ker_pha, reflected_pha), dim=-1)
        return extended_mod, extended_pha
    
    def ifftshift(self, tensor, dim=None):
        if dim is None:
            dim = list(range(tensor.dim()))  # Shift all dimensions by default
        elif not isinstance(dim, (list, tuple)):
            dim = [dim]  # Convert to list if a single dimension is provided

        shifted_tensor = tensor
        for d in dim:
            mid = shifted_tensor.shape[d] // 2
            shifted_tensor = torch.roll(shifted_tensor, shifts=mid, dims=d)
            if shifted_tensor.shape[d] % 2 == 0:  # For even-sized dimensions, flip the first half
                shifted_tensor = torch.flip(shifted_tensor, dims=[d])
        return shifted_tensor
