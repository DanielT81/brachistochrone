import numpy as np

a = [34, -34]
b = [4.00,-4]
c = [-10, 10]

resol1 = np.roots([a[0],b[0],c[0]])
resol2 = np.roots([a[1],b[1],c[1]])
print(np.sqrt(1/3.468358762))
print(resol1, resol2, '\n')
print(resol1[1]-resol2[1], resol1[0]-resol2[0])