# An example of ESS and MLS signals generation in python

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, max_len_seq

# External parameters
fs = 48000
Norder = 3
FACTOR_UP = Norder*2
fs_up = fs*FACTOR_UP

# Sweep parameters
f0 = 1
f1 = 24000
f1 = round(min([f1, 0.95 * fs / 2]))
duration = 2
phi0 = -np.pi * 0.5

# Chirp generation function
def generate_chirp_signal(f0, f1, fs, duration, phi0=0):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    ess = chirp(t, f0, duration, f1, method='logarithmic', phi=phi0)
    return ess, fs

def generate_MLS_signal(length):
    nbits = 20  # More than 16 bits can take a long time!
    [mls, final_state] = max_len_seq(nbits, length=length)
    mls = mls*2 -1
    return mls

[ess, fs_ess] = generate_chirp_signal(f0, f1, fs_up, duration, phi0)

mls = generate_MLS_signal(len(ess))

# Plot generated signals
fig,ax = plt.subplots(2,1)
ax[0].plot(ess)
ax[0].title.set_text('Exponential Swept Sine')
ax[1].plot(mls)
ax[1].title.set_text('Maximum Length Sequence')
plt.tight_layout()
plt.show()