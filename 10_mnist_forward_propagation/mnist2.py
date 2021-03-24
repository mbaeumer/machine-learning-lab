import numpy as np
import gzip
import struct

def load_images(filename):
    with gzip.open(filename,'rb') as f:
        _ignored, n_images, columns, rows = struct.unpack('>IIII', f.read(16))
        all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
        print("Columns: %d, Rows: %d " % (columns,rows))
        return all_pixels.reshape(n_images, columns * rows)

X_train = load_images("train-images-idx3-ubyte.gz")
X_test = load_images("t10k-images-idx3-ubyte.gz")

def load_labels(filename):
  with gzip.open(filename,'rb') as f:
    f.read(8)
    all_labels = f.read()
    A = np.frombuffer(all_labels, dtype=np.uint8).reshape(-1, 1)
    return A


def one_hot_encode(Y):
  n_labels = Y.shape[0]
  n_classes = 10
  encoded_Y = np.zeros((n_labels, n_classes))
  for i in range(n_labels):
    label = Y[i]
    encoded_Y[i][label] = 1
  return encoded_Y


Y_train_unencoded = load_labels("train-labels-idx1-ubyte.gz")
Y_train = one_hot_encode(Y_train_unencoded)
Y_test = load_labels("t10k-labels-idx1-ubyte.gz")
print("test")



