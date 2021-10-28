
# 这个文档作为py的一个模版reference，我会将python data science handbook过一遍， 这个文档基于

# Python for Datascience by VanderPlas, O'Reilly

# OS, kernel, applications, and shell

# OSes are usually a collection of a kernel and a bunch of applications. Examples of OSes are Window$, Mac OS X, UNIX, Ubuntu, Solaris. Note that Linux is not an OS, but rather a kernel, which is the most important part of an OS. A shell is an application that runs on the OS and provides the user interface to the OS.

# In computing, a shell is a computer program which exposes an operating system's services to a human user or other program. In general, operating system shells use either a command-line interface (CLI) or graphical user interface (GUI), depending on a computer's role and particular operation. It is named a shell because it is the outermost layer around the operating system

# IPython (or IPython notebook) is a shell - originally developed to connect the python engine and the user but now for used to work with multiple languages (e.g., Julia and R). Ipython was from 2014 split into two projects: the IPython (e.g., the the kernel, which now becomes rather language independent, though it is still named related to 'python') we have now and the Jupyter notebook (e.g., format, msg protocal, and notebook application etc.). Jupyter notebook is based on IPython kernel. Jupyter notebook can also be used with multiple languages and is extremely popular in DS. It is a web browser based on development environment.

# We can run R in Jupyter notebook or R notebook or R Markdown - neglegible difference.

#%%
import numpy as np
np.__version__

# %%


# Python data are not defined but inferred, e.g., when we say, x=2, x is then considered as integer, and y= 'x', so y is then considered as string. This increases flexibility and convenience at a cost of efficiency as in python the data will have to not only contain the value but also additional information such as the type and size of the data. This is particularly true with list in python.

# To overcome this limitation, we can use array type where all the data inside an array are of the same type. We can use the build-in package 'array', e.g.,
import array
example1 = list(range(20))
example1a = array.array('i', example1)# 'i' indicates the contents are integers.

print(example1a)

# 
# but we prefer numpy array with superior performance in terms of storage, flexiblity, and efficiency.
# %%

import numpy as np

x1 = np.array([1,2,3,5,5,6,7, 8])
print(x1)

x2 = np.array([range(11) for i in x1], dtype='int16')
print(x2)

print(np.eye(4))


# %% each numpy array has some attributes, e.g., 

print(x2.size)
print(x2.ndim)
print(x2.shape)
# %% we can access numpy elements, e.g.,

print(x2[0,3])
print(x2[0,-1])
print(x2[0,::2])

# %% sub arrays are 'views' not 'copies', if we change the sub array, we will also have the original array values changed.

print(x2)
x2_1 = x2[0,1:3]

print(x2_1)
x2_1[0]= 9999


print(x2)

# %% thus, a propery way to have a copy of the sub array is:
x3 = np.array([range(11) for i in x1], dtype='int16')
x3_1 = x3[0,1:3].copy()

x3_1[0] = 9999

print(x3_1)
print(x3)

# %% we can rearrange the values in an array into several dimensions, e.g.,

x4 = x2.reshape((2,4,11))
print(x4)



# %% we can concatenate and split arrays.

x3a, x3b = np.vsplit(x3,[2])

print('vsplit is', x3a, x3b)

# we split horizontally x3 into 3 sub arrays
x3c, x3d, x3e = np.hsplit(x3,[2,5])

print('hsplit is', x3c, x3d, x3e)
# %%

# python is implemented via CPython - some alternatives include PyPy, Cython, and Numba etc. They all have pros and cons but the original CPython is still the most popular because of its flexibility though at a cost of slow operations.

# Numpy can be very fast because it takes advantages of vectorized operations. e.g., we can apply a function to the whole vector/array rather than applying the function to each value of the array respectively. The former approach which numpy takes will push the loop which is done by the latter to the compile layer that underlies numpy, leading to faster execution. e.g.,

# numpy achieves this via unray and ufuncs
# e.g., ufuncs
x = np.array([1,2,3,4,-5,6,-6])
print(abs(x))
print(np.absolute(x))
print(np.power(4,np.abs(x)))
# unary

print(np.add.accumulate(x))

# %% 
# advanced skills in numpy:
# 
# we can also use 'broadcasting' functions in numpy. e.g., to apply binary ufuncs (e.g., addition, subtraction, multiplication etc.) on arrays of different sizes.
# 


# we can perform operations on two arrays of the same size,
a = np.array([0,1,2])
b = np.array([5,6,7])
print(a + b)
# and on arrays of different sizes (e.g., this is called broadcasting) 我们用broadcasting来对多个array进行操作例如加减乘除，它做的事情其实就是把这几个数组各个‘补全’然后‘凑’成相同唯独的几个数组，然后在加减乘除

print( a + 5) #这里其实把5凑成了和a维度等同的[5,5,5]
# this broadcasting can be extended to arrays of higher dimensions:

M = np.ones((3,3))
print(M)
print(M+a)# 这里把a‘补全/自我复制’成了[[0,1,2],[0,1,2],[0,1,2]]
print(M+1)

a = np.arange(3)
b = np.arange(3)[:, np.newaxis]
print(a, b)

print(a + b) # 这里也是相互补全see p65, Figure 2.4 for illustration

# rules broadcasting:
# 1. 
# 2.
# 3. if in any dimension the sizes disagree and neither is equal to 1, an error is raised. see p65

# %% 












# comparision, masks and boolean logic

# masking comes up when we want to extract, modify, count, and do any manipulation based on some criterion.

# [omitted here]

# 'Fancy indexing'



# %%
