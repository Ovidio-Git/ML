

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm # para manejar colores 

############################# GRAFICA EN 3 DIMENSIONES ###############################

def f(x, y):
  return np.sin(x) + 2*np.cos(y)


resolution = 1000  # the resolution for vectors
vector_x = np.linspace(-4, 4, num=resolution) # make vectors for function z
vector_y = np.linspace(-4, 4, num=resolution)
vector_x, vector_y = np.meshgrid(vector_x, vector_y) #genera el plano teniendo como referencia los vectores x y y
#https://www.interactivechaos.com/es/manual/tutorial-de-numpy/la-funcion-meshgrid <-- documentation of function meshgrid

z = f(vector_x, vector_y)

fig, ax = plt.subplots(subplot_kw={"projection":"3d"}) #char in 3d
surf = ax.plot_surface(vector_x, vector_y, z, cmap=cm.summer)
fig.colorbar(surf)


################################  CURVAS DE NIVEL ####################################

fig2, ax2 = plt.subplots()
level_map = np.linspace(np.min(z), np.max(z), num=resolution)

#countur = curvas de nivel sin relleno
#counturf = curvas de nivel con relleno
cp = ax2.contourf(vector_x, vector_y, z, level=level_map, cmap=cm.summer)
fig2.colorbar(cp)
