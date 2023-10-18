# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email: valdecy.pereira@gmail.com
# Lesson: pyMetaheuristic - Jaya (Sanscrit Word for Victory)
# GitHub Repository: <https://github.com/Valdecy>

# Required Libraries
import numpy as np
# Jaya
from pyMetaheuristic.algorithm import victory
from pyMetaheuristic.utils import graphs

# Target Function - It can be any function that needs to be minimize, However it has to have only one argument: 'variables_values'. This Argument must be a list of variables.
# For Instance, suppose that our Target Function is the Easom Function (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)
# Target Function: Easom Function
def easom(variables_values = [0, 0]):
    x1, x2 = variables_values
    func_value = -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi) ** 2 - (x2 - np.pi) ** 2)
    return func_value
def sphere(variables_values):
    return sum(x**2 for x in variables_values)

# Target Function - Values
plot_parameters = {
    'min_values': (-5, -5),
    'max_values': (5, 5),
    'step': (0.1, 0.1),
    'solution': [],
    'proj_view': '3D',
    'view': 'notebook'
}
graphs.plot_single_function(target_function = sphere, **plot_parameters)

# jaya - Parameters
parameters = {
    'size': 500,
    'min_values': (-5, -5),
    'max_values': (5, 5),
    'iterations': 1000,
    'verbose': False
}

# Jaya - Algorithm
jy = victory(target_function = sphere, **parameters)

# Jaya - Solution
variables = jy[:-1]
minimum = jy[ -1]
print('Variables: ', np.around(variables, 4) , ' Minimum Value Found: ', round(minimum, 4) )

from matplotlib import pyplot as plt
# jaya - Plot Solution
plot_parameters = {
    'min_values': (-5, -5),
    'max_values': (5, 5),
    'step': 0.1,
    'solution': [variables[0], variables[1]],
}
x = np.arange(plot_parameters['min_values'][0], plot_parameters['max_values'][0] + plot_parameters['step'], plot_parameters['step'])
y = np.arange(plot_parameters['min_values'][1], plot_parameters['max_values'][1] + plot_parameters['step'], plot_parameters['step'])
X, Y = np.meshgrid(x, y)
Z = np.array([[sphere([x, y]) for x, y in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
min_x, min_y = plot_parameters['solution']
min_z = sphere([min_x, min_y])
ax.scatter([min_x], [min_y], [min_z], c='red', marker='o', label='Global Minimum')
ax.text(min_x, min_y, min_z, f'({min_x:.2f}, {min_y:.2f}, {min_z:.2f})', color='red', fontsize=12, zorder=10)
ax.view_init(elev=40, azim=20)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Function Value')
ax.set_title('Sphere Function (3D Surface Plot)')
plt.legend()
plt.show()

# Jaya je algoritmus pro nalezení nejlepšího řešení optimalizačního problému, např. v našem případě
# pro nalezení globálního minima funkce. Používá populaci kandidátních řešení a iterativně
# je vylepšuje, aby se přiblížila k optimálnímu výsledku. Jaya využívá koncept posouvání všech
# řešení blíže k nejlepšímu řešení (exploatace), zatímco je zároveň posouvá dál od nejhoršího řešení
# (vyhýbání se). Z grafu lze vidět, že algoritmus správně nalezl globální minimum funkce (s menší
# odchylkou). Graf reflektoval změny vstupních parametrů, a to například zvolený počet iterací.
