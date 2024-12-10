import numpy as np
g = np.array([0,-9.81])
def_vec = np.array([10,-10])
def sqrt(def_val):
	return np.sqrt(def_val)

def sqr(def_val):
	return def_val**2

def dot_product(def_vec1: np.ndarray, def_vec2: np.ndarray) -> np.ndarray:
	x1_sum = def_vec1[0] + def_vec2[0]
	x2_sum = def_vec1[1] + def_vec2[1]
	return x1_sum + x2_sum

def cart_norm(def_vec) -> float: # function that puts out the cartesian norm for a vector
	return sqrt(sqr(def_vec[0]) + sqr(def_vec[1]))



half_acceleration = 0.5 * (dot_product(g, def_vec / cart_norm(def_vec)) * (def_vec / cart_norm(def_vec)))  # - (1/2 * 0.1 (air drag coefficient) * area of the
# object *
# velocity**2)
print(half_acceleration)
a = [34, -34]
b = [4.00,-4]
c = [-10, 10]

resol1 = np.roots([a[0],b[0],c[0]])
resol2 = np.roots([a[1],b[1],c[1]])
#print(np.sqrt(1/3.468358762))
##print(resol1, resol2, '\n')
###print(resol1[1]-resol2[1], resol1[0]-resol2[0])

#print(0.5 * np.sqrt(2) * 9.81)