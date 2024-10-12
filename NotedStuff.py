import numpy as np
import matplotlib.pyplot as plt

# Datenpunkte
#x = [1, 2, 3, 4, 5]
#y = [2, 3, 5, 7, 11]
#[x,y] = [10, 8.53553391, 0], [0, 8.53553391, 10]
[x,y] = [10, 1.4651732, 0], [0, 1.4651732, 10]
# Plotten der Punkte mit 'scatter'
plt.scatter(x, y, color='red', label='Punkte')

# Zeichnen der Linien zwischen den Punkten mit 'plot'
plt.plot(x, y, color='blue', label='Linie')

# Labels und Titel
plt.xlabel('X-Achse')
plt.ylabel('Y-Achse')
plt.title('Plot von Punkten und Linien')

# Legende anzeigen
plt.legend()

# Anzeige des Plots
plt.show()
