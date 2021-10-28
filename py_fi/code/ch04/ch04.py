# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# <img src="http://hilpisch.com/tpq_logo.png" alt="The Python Quants" width="35%" align="right" border="0"><br>
# %% [markdown]
# # Python for Finance (2nd ed.)
# 
# **Mastering Data-Driven Finance**
# 
# &copy; Dr. Yves J. Hilpisch | The Python Quants GmbH
# 
# <img src="http://hilpisch.com/images/py4fi_2nd_shadow.png" width="300px" align="left">
# %% [markdown]
# # Numerical Computing with NumPy
# %% [markdown]
# ## Python Lists

# %%
v = [0.5, 0.75, 1.0, 1.5, 2.0]  


# %%
m = [v, v, v]  
m  


# %%
m[1]


# %%
m[1][0]


# %%
v1 = [0.5, 1.5]
v2 = [1, 2]
m = [v1, v2]
c = [m, m]  
c


# %%
c[1][1][0]


# %%
v = [0.5, 0.75, 1.0, 1.5, 2.0]
m = [v, v, v]
m


# %%
v = [0.5, 0.75, 1.0, 1.5, 2.0]

vv =  [v]*3
vv


# %%
v[0] = 'Python'
vv


# %%
from copy import deepcopy
v = [0.5, 0.75, 1.0, 1.5, 2.0]
print(deepcopy(v))
m = 3 * [deepcopy(v), ]  
m


# %%
v[0] = 'Python'  
m  

# %% [markdown]
# ### Python Array Class

# %%
v = [0.5, 0.75, 1.0, 1.5, 2.0]


# %%
import array


# %%
a = array.array('f', v)  
a


# %%
a.append(0.5)  
a


# %%
a.extend([5.0, 6.75])  
a


# %%
2 * a  


# %%
# causes intentional error
# a.append('string')  


# %%
a.tolist()  


# %%
f = open('array.apy', 'wb')  
a.tofile(f)  
f.close()  


# %%
with open('array.apy', 'wb') as f:  
    a.tofile(f)  


 

# %%
b = array.array('f')  


# %%
with open('array.apy', 'rb') as f:  
    b.fromfile(f, 5)  


# %%
b  


# %%
b = array.array('d')  


# %%
with open('array.apy', 'rb') as f:
    b.fromfile(f, 2)  


# %%
b  

# %% [markdown]
# ## NumPy Arrays
# %% [markdown]
# ### The Basics

# %%
import numpy as np  


# %%
#we can convert any iterable into numpy arrays.
a = np.array([0, 0.5, 1.0, 1.5, 2.0])  
a

b = np.array({3:3,4:65,6:4,3:'df'})
print(a, b)


# %%
print(type(a), type(b))


# %%
a = np.array(['a', 'b', 'c'])  
a


# %%
a = np.arange(2, 20, 2)  
print(a)

#this is equivalent to
lst = []
for i in range(2,20,2):
    lst.append(i)

print(np.array(lst))

#this is equivalent to
lst2 = np.array([i for i in range(2, 20, 2)])
print(lst2)


# %%
a = np.arange(8, dtype=np.float)  

print(a)
np_array = np.arange(20, dtype = np.float)

print(np_array)


# %%
a[5:]  


# %%
a[:2]  


# %%
a.sum()  


# %%
a.std()  


# %%
a.cumsum()  


# %%
l = [0., 0.5, 1.5, 3., 5.]
2 * l  


# %%
a


# %%
print(2 * a)  
np.concatenate([a]*3)


# %%
a ** 2  


# %%
2 ** a  


# %%
a ** a  


# %%
np.exp(a)  


# %%
np.sqrt(a)  


# %%



# %%
import math


# %%
import math  


# %%
math.sqrt(2.5)  


# %%
# causes intentional error
#math.sqrt(a)  





# %% [markdown]
# ### Multiple Dimensions

# %%
b = np.array([a, a * 2])  
b


# %%
b[0]  


# %%
b[0][2]


# %%
b[0, 2]  


# %%
b[:, 1]  


# %%
x1 = b.sum()  
x2 = b[1].sum()
print(x1, x2)


# %%
b.sum(axis=0)  


# %%
b.sum(axis=1)  

# %% [markdown]
# fill values to an array template

# %%
c = np.zeros((2, 3), dtype='i', order='C')  
c


# %%
c = np.zeros((2, 3, 4))  
c


# %%
c = np.ones((2, 3, 4), dtype='i', order='C')  
c


# %%
c = np.ones((2, 3, 4))
c






# %%
e = np.empty((2, 3, 2))  
e


# %%
f = np.zeros_like(c)  
f[1][1]


# %%
np.eye(5)  
np.eye(12)


# %%
g = np.linspace( [2, 244, 23],[4,3,222],20)  #geneate 20 values to fill in an array, the starting value is [2, 244, 23], the ending avlue is [4,3,222].
print(g)

# %% [markdown]
# ### Meta-Information

# %%
g.size  


# %%
g.itemsize  #the number of bytes to represent an element.


# %%
g.ndim  


# %%
g.shape  


# %%
g.dtype  


# %%
g.nbytes  

# %% [markdown]
# ### Reshaping, Resizing, Stacking, Flattening

# %%
g = np.arange(15)


# %%
g


# %%
print('{} {}'.format(type(g.shape), g.shape ))
np.ndim(g.shape)


# %%
np.shape(g) 


# %%
g.reshape((3, 5))  


# %%
h = g.reshape((5, 3))  
h


# %%
h.T  


# %%
h.transpose()  


# %%
g


# %%
np.resize(g, (3, 1))  #resize the numpy array into an array of (3,1) dimension, this leads to truncatation of the first three elements.


# %%
np.resize(g, (1, 5))  


# %%
np.resize(g, (2, 5))  


# %%
n = np.resize(g, (5, 4))  #this leads to data (duplilcation)
n


# %%
h


# %%
np.hstack((h, 2 * h))  


# %%
np.vstack((h, 0.5 * h))  


# %%
h


# %%
h.flatten()  


# %%
h.flatten(order='C')  
h.flatten()


# %%
h.flatten


# %%
for i in h.flat:  #this is equivalent to h.flatten()
    print(i, end=',')


# %%
for i in h.ravel(order='C'):  #this is equivalent to h.flatten()
    print(i, end=',')


# %%
for i in h.ravel(order='F'):  
    print(i, end=',')

# %% [markdown]
# ### Boolean Arrays

# %%
h


# %%
h > 8  


# %%
xx  = h > 8
print(type(xx.astype(int)))
xx.astype(int)


# %%
h <= 7  


# %%
h == 5  


# %%
(h == 5).astype(int)  


# %%
(h > 4) & (h <= 12)  


# %%
h[h > 8]  


# %%
h[(h > 4) & (h <= 12)]  


# %%
h[(h < 4) | (h >= 12)]  


# %%
h


# %%
np.where(h > 7, 1, 0)  


# %%
np.where(h % 2 == 0, 'even', 'odd')  


# %%
np.where(h <= 7, h * 2, h / 2)  

# %% [markdown]
# ### Speed Comparison

# %%
import random
I = 100


# %%


# %%
x = [[random.gauss(0, 1) for j in range(I)]              for i in range(I)]  
x


# %%
I = 3

[[random.gauss(0, 1) for j in range(I)]              for i in range(I)]  


# %%


# %%


# %%
import sys



# %% [markdown]
# ### Structured Arrays, so that we no longer require all elements in a numpy array to be of the same type - we only require all elements in the same columnn of the numpy array to be of the same type.

# %%
dt = np.dtype([('Name', 'S10'), ('Age', 'i4'),
               ('Height', 'f'), ('Children/Pets', 'i4', 2)])  


# %%
dt  


# %%
dt = np.dtype({'names': ['Name', 'Age', 'Height', 'Children/Pets'],
             'formats':'O int float int,int'.split()})  


# %%
dt  


# %%
s = np.array([('Smith', 45, 1.83, (0, 1)),
              ('Jones', 53, 1.72, (2, 2))], dtype=dt)  


# %%
s  


# %%
type(s)  


# %%
s['Name']  


# %%
s['Height'].mean()  


# %%
s[0]  


# %%
s[1]['Age']  

# %% [markdown]
# ## Vectorization of Code

# %%
np.random.seed(100)
r = np.arange(12).reshape((4, 3))  
s = np.arange(12).reshape((4, 3)) * 0.5  


# %%
r  


# %%
s  


# %%
r + s  


# %%
#np.concatenate([r,s])
np.hstack((r,s))


# %%
r + 3  


# %%
2 * r  


# %%
2 * r + 3  


# %%
r


# %%
r.shape


# %%
s = np.arange(0, 12, 4)  
s  


# %%
r + s  


# %%
s = np.arange(0, 12, 3)  
s  


# %%
# causes intentional error
# r + s  


# %%
r.transpose() + s  


# %%
sr = s.reshape(-1, 1)  
sr


# %%
sr.shape  


# %%
r + s.reshape(-1, 1)  


# %%
def f(x):
    return 3 * x + 5  


# %%
f(0.5)  


# %%
f(r)  

# %% [markdown]
# ## Memory Layout
# %% [markdown]
# Cf. http://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays/

# %%
x = np.random.standard_normal((1000000, 5))  
x


# %%
y = 2 * x + 3  


# %%
C = np.array((x, y), order='C')  
C


# %%
F = np.array((x, y), order='F')  


# %%
x = 0.0; y = 0.0  


# %%
C[:2].round(2)  

 
# %%
F = 0.0; C = 0.0  

# %% [markdown]
# <img src="http://hilpisch.com/tpq_logo.png" alt="The Python Quants" width="35%" align="right" border="0"><br>
# 
# <a href="http://tpq.io" target="_blank">http://tpq.io</a> | <a href="http://twitter.com/dyjh" target="_blank">@dyjh</a> | <a href="mailto:training@tpq.io">training@tpq.io</a>

