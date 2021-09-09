

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

def plot(history):
    sns.set()
    plt.plot(history.history['loss'], label='Training set', color='blue', linestyle='-')
    plt.plot(history.history['val_loss'], label='Validation set', color='green', linestyle='--')
    plt.xlabel("Epochs", fontsize=30)
    plt.ylabel("Loss", fontsize=30)
    plt.xlim(0, len(history.history['loss']))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=30)
    plt.show()

