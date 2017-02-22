import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from random import gauss

x=[gauss(85, 15) for i in range(1500)]
y=[gauss(15, 15) for i in range(1500)]

plt.hist2d(x,y)
plt.xlabel('Posicion [cm]')
plt.ylabel('Posicion [cm]')
plt.colorbar()
plt.show()