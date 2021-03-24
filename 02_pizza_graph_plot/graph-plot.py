#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

sb.set()
plt.axis([0, 55, 0, 55])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Reservations", fontsize=30)
plt.ylabel("Pizzas", fontsize=30)
X, Y = np.loadtxt("pizza.txt",skiprows=1,unpack=True)
plt.plot(X, Y, "bo")
plt.show()

