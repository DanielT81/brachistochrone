import numpy as np
import matplotlib.pyplot as plt


def cycloid(0, 10, 10, 0, ):
    # Calculate the range of the cycloid (end minus start)
    delta_x = x_end - x_start
    delta_y = y_end - y_start

    # Use the formula of a cycloid but scale it to the specified range
    # Find a suitable scaling factor for the radius
    r = delta_x / (2 * np.pi)  # scaling factor to fit x range

    # Parametric cycloid equations
    def cycloid_function(t):
        x = r * (t - np.sin(t))
        y = r * (1 - np.cos(t))
        return x, y

    # Find the t values corresponding to x_start and x_end
    t_start, t_end = 0, 2 * np.pi  # Default range for the cycloid

    # Rescale the parameter range for the given x_start and x_end
    def rescale_t(x_input):
        t = np.linspace(t_start, t_end, 1000)
        x_t_values = np.array([cycloid_function(t_i)[0] for t_i in t])
        t_idx = np.searchsorted(x_t_values, x_input)
        return t[t_idx]

    # Now, find the t corresponding to the input x_val
    t_val = rescale_t(x_val)

    # Get the y value at this t
    x_res, y_res = cycloid_function(t_val)

    # Scale y to fit between y_start and y_end
    y_res = y_start + (y_res / r) * delta_y

    return y_res


# Example Usage
x_start = 0
x_end = 10
y_start = 0
y_end = 5
x_val = 5

y_at_x = cycloid(x_start, x_end, y_start, y_end, x_val)
print(f"The y-value at x = {x_val} is: {y_at_x}")

# Plotting the cycloid
x_vals = np.linspace(x_start, x_end, 500)
y_vals = [cycloid(x_start, x_end, y_start, y_end, x) for x in x_vals]

plt.plot(x_vals, y_vals, label='Cycloid curve')
plt.scatter(x_val, y_at_x, color='red', label=f"Point at x={x_val}, y={y_at_x}")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cycloid Curve')
plt.legend()
plt.grid(True)
plt.show()
