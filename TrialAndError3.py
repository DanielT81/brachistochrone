import time
from math import sqrt, sin, cos, pi
import numpy as np
from matplotlib import pyplot as plt
from numpy import asarray as aarr
start_time = time.perf_counter()  # to track the computation-time

'''defining all the variables'''
changes = 0
g = -9.81
ATP = 10 # amount of total points in the system
set_start_point = aarr([0,10]) #the set start point for the computation
set_end_point = aarr([10,0]) #the set end point for the computation
initial_x_values = np.linspace(set_start_point[0], set_end_point[0], ATP)
initial_y_values = np.linspace(set_start_point[1], set_end_point[1], ATP)
arr = np.array([initial_x_values,initial_y_values]).transpose()
new_arr = arr


'''defining all the functions'''
def sqr(def_var) -> float: # function that returns the square of a float
	return def_var ** 2
def cart_norm(def_vec) -> float: # function that puts out the cartesian norm for a vector
	return sqrt(sqr(def_vec[0]) + sqr(def_vec[1]))
def vec(start_point, end_point) -> np.ndarray: # function that returns the vector between two points
	result = end_point - start_point
	return result
def vec_len(def_vec) -> float:
	def_len = sqrt(sqr(def_vec[0]) + sqr(def_vec[1]))
	return def_len
def mid_point(start_point, end_point) -> np.ndarray: # function that returns the point directly in the middle of two points
	def_point = start_point + 0.5 * vec(start_point, end_point)
	return def_point
def dot_product(def_vec1: np.ndarray, def_vec2: np.ndarray) -> np.ndarray:
	x1_sum = def_vec1[0] * def_vec2[0]
	x2_sum = def_vec1[1] * def_vec2[1]
	return x1_sum + x2_sum
def normalize_vec(def_vec) -> np.ndarray: # function that returns the normalization of a given vector
	return def_vec / cart_norm(def_vec)
def normal_vec(def_vec) -> np.ndarray: # function that returns the normalized normal vector to a vector
	def_norm_vec = normalize_vec(np.array([float(-def_vec[0]), float(def_vec[1])])) # normalvector to def_vec
	return def_norm_vec
def plot(good_or_bad):
	x = arr[:,0]
	y = arr[:,1]
	new_x = new_arr[:,0]
	new_y = new_arr[:,1]
	plt.plot(x, y, label='graph', marker='o')
	plt.plot(new_x, new_y, label='new_graph', marker='x')
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
	print(f'a: {a_coefficient}, b: {b_coefficient}, c: {c_coefficient}')
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
		#return
	time_result = possible_time_arr[1]
	velocity_result = def_start_vel + 2 * a_coefficient * time_result
	return aarr([time_result, velocity_result])


def calc_arr_time(def_name) -> np.ndarray:
	def_time = 0
	def_vel = 0
	for def_index in range(ATP-1):
		(def_new_time, def_new_vel) = physics(def_vel, aarr(def_name[def_index]), aarr(def_name[def_index+1]), def_index)
		print(def_time, def_vel)
		def_time += def_new_time #/ 2
		def_vel = def_new_vel
	#print([def_time, def_vel])
	#return np.array([def_time, def_vel], dtype=object)
	return def_time


def move_random_point() -> None: # rng point in arr gets moved by amount and stored in same column in new_arr
	global new_arr
	def_index = np.random.randint(1, ATP - 1)
	def_old_point = arr[def_index]
	def_vec = vec(arr[def_index-1],arr[def_index+1])
	#def_norm_vec = normal_vec(def_vec)
	def_norm_vec = np.array([1,1])
	def_new_point = def_old_point + def_norm_vec * optimizing_factor * np.random.choice([1,-1])
	new_arr[def_index] = def_new_point


def loop(def_iterations: int or float) -> None:
	global arr
	global arr_time
	global new_arr
	for def_iteration in range(def_iterations):
		new_arr = arr
		plot('before move_random_point')
		move_random_point()
		plot('after move_random_point')
		new_time = calc_arr_time(new_arr)
		#print(f'arr is {arr}')

		if new_time < arr_time:
			print('gut')
			plot('after good')
			arr = new_arr
			arr_time = new_time
		else:
			plot('after bad')
			new_arr = arr



'''defining the variables that depend on functions'''
arr_time = calc_arr_time(arr)
global_vec = vec(set_start_point, set_end_point) # setting the boundary vector for easy debugging
optimizing_factor = np.dot(global_vec, global_vec) * 0.001 # the global factor for the normal vector

#print(physics(aarr([0,0]),aarr([0,10]), aarr([10,0])))
print('\n' * 5)
'''
for i in range(20000):
	move_random_point()
'''
loop(10000)
print(arr)



'''
vecc1 = np.array([0,50])
vecc2 = np.array([10,0])
vell = np.array([10,-50])
(timee, velly) = physics(vell, vecc1, vecc2)
print(f'\n \n \nphysics term normally is {timee, velly} or {timee, vec_len(velly)}')
#print(test_val / physics(aarr([0,0]), set_start_point, set_end_point, 2)[0] - 1.9992130052793873)
'''


plot()

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"\n \n Elapsed time: {elapsed_time} seconds")
