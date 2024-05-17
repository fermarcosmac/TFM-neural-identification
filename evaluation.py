import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile

def read_kernels(directory):

    wav_dict = {}

    for filename in os.listdir(directory):

        if filename.endswith('.wav'):

            filepath = os.path.join(directory, filename)
            sample_rate, data = wavfile.read(filepath)
            var_name = os.path.splitext(filename)[0]
            wav_dict[var_name] = data

    return wav_dict, sample_rate


# Load the loss history from the .npy file
SNN_loss_history = np.load('models/SNN/loss_history.npy')
NN_loss_history = np.load('models/NN/loss_history.npy')

# Load kernels
NN_dir = os.path.join('kernels/NN')
SNN_dir = os.path.join('kernels/SNN')
ker_NN_dict, sr = read_kernels(NN_dir)
ker_SNN_dict, sr = read_kernels(NN_dir)

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
# TODO
plt.figure(2)

hey=0