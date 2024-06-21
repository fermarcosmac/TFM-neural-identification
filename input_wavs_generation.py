import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import chirp, max_len_seq
from scipy.io.wavfile import write

# Seed pseudo-randomness 
np.random.seed(123)

# Save paths
save_path_ess = "./inputs_wav/ESS/"
save_path_mls = "./inputs_wav/MLS/"

# External parameters
fs = 64000
Norder = 3
FACTOR_UP = Norder
fs_up = int(fs * FACTOR_UP)
num_signals = 20

# Sweep parameters
f0 = 1
f1 = 24000
f1 = round(min([f1, 0.95 * fs / 2]))
f1 = f1*FACTOR_UP
duration = 2

# Chirp generation function (deterministic)
def generate_chirp_signal(f0, f1, fs, duration, phi0=0):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    ess = chirp(t, f0, duration, f1, method='logarithmic', phi=phi0)
    return ess, fs

# MLS generation function (random)
def generate_MLS_signal(length):
    nbits = 20
    init_state = generate_random_initial_state(nbits)
    [mls, final_state] = max_len_seq(nbits, state=init_state, length=length)
    mls = mls*2 -1
    return mls

def generate_random_initial_state(length):
    return np.random.choice([False, True], size=length)

# Generate a few signals with different initial phases (ESS) and initial states (MLS)
phi0_values = np.linspace(0, 180, num_signals, endpoint=False)
for i, phi0 in enumerate(phi0_values):
    # Generate signals
    ess, fs_ess = generate_chirp_signal(f0, f1, fs_up, duration, phi0)
    mls = generate_MLS_signal(len(ess))
    # Scale chirp to save it as 32-bit floating point (bit-depth)
    ess = np.float32(ess)
    # Save the signals as .wav files
    write(os.path.join(save_path_ess,f"ess_{i}.wav"), fs_up, ess)
    write(os.path.join(save_path_mls,f"mls_{i}.wav"), fs_up, mls)
