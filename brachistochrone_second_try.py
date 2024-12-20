import time
from math import sqrt, sin, cos, pi
import numpy as np
from matplotlib import pyplot as plt
from numpy import asarray as aarr
start_time = time.perf_counter()  # to track the computation-time


'''set the start and end points of the calculation'''
set_start_point = aarr([0,10]) # the set start point for the computation
set_end_point = aarr([10,0]) # the set end point for the computation
amount_of_total_points = 20 # amount of total points in the system
precision_value = 15 # the precision value for the calculation


'''defining all the functions'''
def sqr(def_var) -> float: # function that returns the square of a float
	return def_var ** 2
def vec_length(def_vec) -> float: # function that puts out the cartesian norm for a vector
	return sqrt(sqr(def_vec[0]) + sqr(def_vec[1]))
def vec(start_point, end_point) -> np.ndarray: # function that returns the vector between two points
	result = end_point - start_point
	return result
def mid_point(start_point, end_point) -> np.ndarray: # function that returns the point directly in the middle of two points
	def_point = start_point + 0.5 * vec(start_point, end_point)
	return def_point
def dot_product(def_vec1: np.ndarray, *def_vec2: np.ndarray) -> np.ndarray:
	if def_vec2:
		x1_sum = def_vec1[0] * def_vec2[0]
		x2_sum = def_vec1[1] * def_vec2[1]
	else:
		x1_sum = def_vec1[0] * def_vec1[0]
		x2_sum = def_vec1[1] * def_vec1[1]
	return x1_sum + x2_sum
def normalize_vec(def_vec) -> np.ndarray: # function that returns the normalization of a given vector
	return def_vec / vec_length(def_vec)
def normal_vec(def_vec) -> np.ndarray: # function that returns the normalized normal vector to a vector
	def_norm_vec = normalize_vec(np.array([float(-def_vec[0]), float(def_vec[1])])) # normalvector to def_vec
	return def_norm_vec
def cycloid(r, t):
	x = r * (t - np.sin(t))
	y = r * (1 - np.cos(t))
	return x, -y + 10
def plot(good_or_bad):
	x = arr[:,0]
	y = arr[:,1]
	x_c, y_c = cycloid(r,t)
	plt.plot(x_c, y_c, label='cycloid_graph')


	new_x = new_arr[:,0]
	new_y = new_arr[:,1]
	plt.plot(x, y, label='graph', marker='o')
	#plt.plot(new_x, new_y, label='new_graph', marker='x')
	plt.plot(initial_x_values, initial_y_values, label='initial line', marker='o')
	plt.xlabel('X')
	plt.ylabel('Y')
	#if good_or_bad:
	#	if good_or_bad == 1:
	#		plt.title('Plot von dem Scheiß, aber gut')
	#		print('\n' * 5, 'GUUUT')
	#	if good_or_bad == 2:
	#		plt.title('Plot von dem Scheiß, aber schlecht')
	#		print('\n' * 5, 'GUUUT')
	plt.title(good_or_bad)
	plt.grid(True)
	plt.show()
def physics(def_start_vel: float, def_start_point: np.ndarray, def_end_point: np.ndarray, *debugger) -> np.ndarray: # function that calculates the time taken
	# for a point to roll down a vector
	if debugger:
		print(f'this is the {debugger} physics term')
	if def_start_vel > 0:
		print('physics: the starting velocity doesnt make sense here')
		plot()
		return
	def_vec = vec(def_start_point, def_end_point)
	delta_x, delta_y = def_vec # setting the differences in coordinates
	delta_s = sqrt(sqr(delta_x) + sqr(delta_y)) # setting the length of the vector
	acceleration_angle_factor = sqrt(1 / (1 + sqr(delta_x / delta_y))) # factor for the acceleration based on the rolling angle
	a_coefficient = ((-0.5) * g * acceleration_angle_factor * (delta_y / sqrt(delta_y ** 2)))  # - (1/2 * 0.1 (air drag coefficient) * area of the object * velocity**2)
	b_coefficient = def_start_vel
	c_coefficient = delta_s
	#print(f'a: {a_coefficient}, b: {b_coefficient}, c: {c_coefficient}')
	possible_time_arr = np.roots([a_coefficient, b_coefficient, c_coefficient])
	if type(possible_time_arr[0]) == np.complex128 or type(possible_time_arr[1]) == np.complex128:
		print('physics: problem with the imaginary unity',  debugger)
		return
	possible_time_arr = possible_time_arr[np.argsort(possible_time_arr)]
	if possible_time_arr[1] < 0:
		print('physics: both t values are negative',  debugger)
		print(possible_time_arr)
		return
	if possible_time_arr[0] > 0:
		print('physics: both t values are positive',  debugger)
		print(possible_time_arr)
		return
	time_result = possible_time_arr[1]
	velocity_result = def_start_vel + 2 * a_coefficient * time_result
	return aarr([time_result, velocity_result])
def calc_arr_time(def_name) -> np.ndarray:
	def_time = 0
	def_vel = 0
	for def_index in range(amount_of_total_points - 1):
		(def_new_time, def_new_vel) = physics(def_vel, aarr(def_name[def_index]), aarr(def_name[def_index+1]))
		def_time += def_new_time #/ 2
		def_vel = def_new_vel
	return def_time
def move_random_point() -> None: # rng point in arr gets moved by amount and stored in same column in new_arr
	global new_arr
	def_index = np.random.randint(1, amount_of_total_points - 1)
	def_old_point = arr[def_index]
	def_vec = vec(arr[def_index-1],arr[def_index+1])
	def_norm_vec = normal_vec(def_vec)
	def_new_point = def_old_point + def_norm_vec * optimizing_factor * np.random.choice([1,-1])
	new_arr[def_index] = def_new_point
def loop() -> None: # loop that creates the new points and compares whether they are better that way
	global arr
	global arr_time
	global new_arr
	global waiting
	global total_iterations
	while waiting < safeguard:#sqr(ATP * 5): # Term is to make sure that there is no more optimizing possible
		new_arr = np.array(arr, copy=True)
		move_random_point()
		new_time = calc_arr_time(new_arr)
		#print(f'new_time: {new_time} \narr_time: {arr_time}')
		total_iterations += 1
		if new_time < arr_time:
			arr = new_arr
			arr_time = new_time
			print(waiting)
			waiting = 0
		else:
			new_arr = arr
			waiting += 1
	print(f'the safeguard is: {safeguard}')
	print(f'the amount of total iterations is: {total_iterations}')
	print(f'the time for the optimized points is: {arr_time} second')



'''defining the variables'''
initial_x_values = np.linspace(set_start_point[0], set_end_point[0], amount_of_total_points) # initial x values in the line
initial_y_values = np.linspace(set_start_point[1], set_end_point[1], amount_of_total_points) # initial y values in the line
global_vec = vec(set_start_point, set_end_point) # setting the boundary vector for easy debugging

waiting = 0 # amount of iterations that have passed since the last change
total_iterations = 0 # the amount of total iterations
g = -9.81 # constant for gravitational acceleration on earth
safeguard = amount_of_total_points ** 2 * 2 # the safeguard for the loop function
optimizing_factor = dot_product(global_vec) / sqr(amount_of_total_points * precision_value) # the global factor for the normal vector

arr = np.array([initial_x_values,initial_y_values]).transpose() # array with yet optimal points
arr_time = calc_arr_time(arr) # the (yet initial) time take for the points in arr
new_arr = arr # array with the random change to compare the time with

r = 5 / 0.87248  # radius of the cycloids wheel for comparison
t = np.linspace(0, np.pi, 1000)  # parameter t, range for several cycles



'''running the function'''
print('\n' * 5)
loop()



'''information about the runtime'''
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"\n\nElapsed time: {elapsed_time} seconds")



'''plotting the graph'''
plot('after')
