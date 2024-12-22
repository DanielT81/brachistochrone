import numpy as np
import matplotlib.pyplot as plt

'''
# Assume x_data and y_data are your input points
x_data = np.linspace(-10, 10, 50)  # Example x values (10 to 100 points)
y_data = np.sin(x_data)  # Example y values (just an example)

# Fit a polynomial of degree n (for Taylor-like approximation, use a low degree like 3 or 4)
degree = 3
coeffs = np.polyfit(x_data, y_data, degree)

# Create a polynomial from the coefficients
poly = np.poly1d(coeffs)

# Generate a smooth range of x values to plot the fitted polynomial
x_smooth = np.linspace(min(x_data), max(x_data), 500)
y_smooth = poly(x_smooth)

# Plot the original data and the polynomial approximation
plt.scatter(x_data, y_data, color='red', label='Data points')
plt.plot(x_smooth, y_smooth, label=f'Polynomial fit (degree {degree})', color='blue')
plt.legend()
plt.show()
'''


r = 5 / 0.87248  # radius of the cycloids wheel for comparison
t = np.linspace(-1, np.pi, 1000)  # parameter t, range for several cycles

def cycloid(r, t):
	x = r * (t - np.sin(t))
	y = r * (1 - np.cos(t))
	return x, -y + 10


x_c, y_c = cycloid(r, t)
plt.plot(x_c, y_c, label='cycloid_graph')




coeffs =[-0.00091, 0.0254, -0.26356, 1.3015, -3.9468, 10]

degree = len(coeffs)-1
poly = np.poly1d(coeffs)
x_smooth = np.linspace(0, 10, 500)
y_smooth = poly(x_smooth)

plt.axis((-1, 11, -1, 11))
plt.plot(x_smooth, y_smooth, label=f'Polynomial fit (degree {degree})', color='blue')
plt.show()
