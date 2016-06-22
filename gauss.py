import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from random import gauss

x=[gauss(1, 0.5) for i in range(10000)]
y=[gauss(2, 0.5) for i in range(10000)]

plt.hist2d(x,y)
plt.colorbar()
plt.show()