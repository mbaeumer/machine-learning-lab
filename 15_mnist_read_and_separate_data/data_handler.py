import numpy as np
import gzip
import struct

def load_images(filename):
    # Open and unzip the file of images:
    with gzip.open(filename, 'rb') as f:
        # Read the header information into a bunch of variables:
        _ignored, n_images, columns, rows = struct.unpack('>IIII', f.read(16))
        # Read all the pixels into a NumPy array of bytes:
        all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape the pixels into a matrix where each line is an image:
        return all_pixels.reshape(n_images, columns * rows)


def load_labels(filename):
    # Open and unzip the file of images:
    with gzip.open(filename, 'rb') as f:
        # Skip the header bytes:
        f.read(8)
        # Read all the labels into a list:
        all_labels = f.read()
        # Reshape the list of labels into a one-column matrix:
        return np.frombuffer(all_labels, dtype=np.uint8).reshape(-1, 1)


def one_hot_encode(Y):
    n_labels = Y.shape[0]
    n_classes = 10
    encoded_Y = np.zeros((n_labels, n_classes))
    for i in range(n_labels):
        label = Y[i]
        encoded_Y[i][label] = 1
    return encoded_Y

if __name__ == "__main__":
    X_train = load_images("train-images-idx3-ubyte.gz")
    X_train_all = load_images("t10k-images-idx3-ubyte.gz")
    X_validation, X_test = np.split(X_train_all, 2)

    X_train_unencoded = load_labels("train-labels-idx1-ubyte.gz")
    Y_train = one_hot_encode(X_train_unencoded)

    Y_test_all = load_labels("t10k-labels-idx1-ubyte.gz")
    Y_validation, Y_test = np.split(Y_test_all, 2)
