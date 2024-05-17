import numpy as np
import matplotlib.pyplot as plt


# Load the loss history from the .npy file
loaded_loss_history = np.load('loss_history.npy')

# Plot the loss history
plt.plot(loaded_loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.show()