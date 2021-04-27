
from matplotlib import cm  #  colors
import numpy as np 
import matplotlib.pyplot as plt


# GRAPH 3D FUNCTION COST

fig, ax = plt.subplots(subplot_kw={"projection":"3d"})

def f(x,y):
  return x**2 + y**2

resolution = 1000
vector_x = np.linspace(-4, 4, num=resolution)
vector_y = np.linspace(-4, 4, num=resolution)

X, Y = np.meshgrid(vector_x, vector_y)
Z = f(X, Y)

sup = ax.plot_surface(X, Y, Z, cmap=cm.summer)
fig.colorbar(sup)

# LEVEL CURV

level_map = np.linspace(np.min(Z), np.max(Z), num=100)
plt.contourf(X, Y, Z, levels=level_map, cmap=cm.summer)
plt.colorbar()

#GRADIENT DESCENT

plt.contourf(X, Y, Z, levels=level_map, cmap=cm.summer)
plt.colorbar()
plt.title("gradient descent")

#point random 

point = np.random.rand(2) * 8 - 4 #para que el vector quede entre [-4 ,4] se hace lo del *8-4 ya que la funcion manda un numero desde 0 a 1
steps = 0.01
learning_rate = 0.01
plt.plot(point[0], point[1], 'o', c='r')

def derivate(copy_point, point):
  return (f(copy_point[0], copy_point[1]) - f(point[0], point[1])) / steps



def gradient(point):
  grad = np.zeros(2) #inicializing 0 the vector "memory"
  for idx, val in enumerate(point):
    copy_point = np.copy(point)
    copy_point[idx] = copy_point[idx] + steps

    derivate_partial = derivate(copy_point, point)
    grad[idx] = derivate_partial 
  return grad


for i in range(100):
  point = point - learning_rate*gradient(point)
  if (i % 10 == 0):
    plt.plot(point[0], point[1], 'o', c='w')   
