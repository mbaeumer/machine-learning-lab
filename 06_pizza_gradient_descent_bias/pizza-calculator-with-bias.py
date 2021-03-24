#!/usr/bin/python

import numpy as np

def predict(X, w):
  return np.matmul(X, w)

def loss(X, Y, w):
  return np.average((predict(X, w) - Y) ** 2)

def gradient(X, Y, w):
  return 2 * np.matmul(X.T, (predict(X, w) - Y)) / X.shape[0]
  #w_gradient = 2 * np.average(X * (predict(X, w, b) - Y))
  #b_gradient = 2 * np.average(predict(X, w, b) - Y)
  #return (w_gradient, b_gradient)

def train(X, Y, iterations,lr):
  w = np.zeros((X.shape[1], 1))
  for i in range(iterations):
    current_loss = loss(X, Y, w)
    print("Iteration %4d => Loss: %.6f" % (i, current_loss))
    #w_gradient, b_gradient = gradient(X, Y, w, b)
    w -= gradient(X, Y, w) * lr
    #b -= b_gradient * lr
  return w

x1, x2, x3, y = np.loadtxt("pizza_3_vars.txt", skiprows=1, unpack=True)
X = np.column_stack((np.ones(x1.size), x1, x2, x3))
Y = y.reshape(-1, 1)

w = train(X, Y, iterations=100000, lr=0.001)
print("\nw=%.10f, b=%.10f" % (w,b))

print("Prediction: x=%d => y=%.2f" % (20, predict(20, w)))


