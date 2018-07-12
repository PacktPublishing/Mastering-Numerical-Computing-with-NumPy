from __future__ import print_function

import time
from datetime import datetime

import numpy as np
from numpy.random import rand
from numpy.linalg import qr
from numpy.linalg import eig
from scipy.linalg import lu
from scipy.linalg import cholesky


def timer(*args, operation, n):
    """
    Returns average time spent 
    for given operation and arguments.
    
    Parameters
    ----------
        *args: list (of numpy.ndarray, numpy.matrixlib.defmatrix.matrix or both)
            one or more numpy vectors or matrices
        operation: function
            numpy or scipy operation to be applied to given arguments
        n: int 
            number of iterations to apply given operation

    Returns
    -------
        avg_time_spent: double
            Average time spent to apply given operation
        std_time_spent: double
            Standard deviation of time spent to apply given operation
        
    Examples
    --------
    >>> import numpy as np

    >>> vec1 = np.array(np.random.rand(1000))
    >>> vec2 = np.array(np.random.rand(1000))

    >>> args = (vec1, vec2)
    
    >>> timer(*args, operation=np.dot, n=1000000)
    8.942582607269287e-07
    """
    
    # Following list will hold the
    # time spent value for each iteration
    time_spent = []
    
    # Configuration info
    print("""
    -------------------------------------------
    
    ### {} Operation ###
    
    Arguments Info
    --------------
    args[0] Dimension: {},
    args[0] Shape: {},
    args[0] Length: {}
    """.format(operation.__name__,
        args[0].ndim,
        args[0].shape,
        len(args[0])))
    
    # If *args length is greater than 1, 
    # print out the info for second argument
    args_len = 0
    for i, arg in enumerate(args):
        args_len += 1
        
    if args_len > 1:
        print("""
    args[1] Dimension: {},
    args[1] Shape: {},
    args[1] Length: {}
        """.format(args[1].ndim,
            args[1].shape,
            len(args[1])))
   
    print("""
    Operation Info
    --------------
    Name: {},
    Docstring: {}
    
    Iterations Info
    ---------------
    # of iterations: {}""".format(
        operation.__name__,
        operation.__doc__[:100] + 
        "... For more info type 'operation?'",
        n))
    
    print("""
    -> Starting {} of iterations at: {}""".format(n, datetime.now()))
    
    if args_len > 1:
        for i in range(n):
            start = time.time()
            operation(args[0], args[1])
            time_spent.append(time.time()-start)
    else:
        for i in range(n):
            start = time.time()
            operation(args[0])
            time_spent.append(time.time()-start)
        
    avg_time_spent = np.sum(time_spent) / n
    std_time_spent = np.std(time_spent)
    
    print("""
    -> Average time spent: {} seconds,
    -> Std. deviation time spent: {} seconds
    
    -------------------------------------------
    """.format(avg_time_spent, std_time_spent))
    
    return avg_time_spent, std_time_spent



# Seed for reproducibility
np.random.seed(8053)

dim = 100
n = 10000

v1, v2 = np.array(rand(dim)), np.array(rand(dim))
m1, m2 = rand(dim, dim), rand(dim, dim)

# Vector - Vector Product
args = [v1, v2]
vv_product = timer(*args, operation=np.dot, n=n)

# Vector - Matrix Product
args = [v1, m1]
vm_product = timer(*args, operation=np.dot, n=n)

# Matrix - Matrix Product
args = [m1, m2]
mm_product = timer(*args, operation=np.dot, n=n)

# Singular-value Decomposition
args = [m1]
sv_dec = timer(*args, operation=np.linalg.svd, n=n)

# LU Decomposition
args = [m1]
lu_dec = timer(*args, operation=lu, n=n)

# QR Decomposition
args = [m1]
qr_dec = timer(*args, operation=qr, n=n)

# Cholesky Decomposition
M = np.array([[1, 3, 4], 
     [2, 13, 15], 
     [5, 31, 33]])
args = [M]
cholesky_dec = timer(*args, operation=cholesky, n=n)

# Eigenvalue Decomposition
args = [m1]
eig_dec = timer(*args, operation=eig, n=n)

print("""
V-V Product: {},
V-M Product: {},
M-M Product: {},
SV  Decomp.: {},
LU  Decomp.: {},
QR  Decomp.: {},
Cholesky D.: {},
Eigval Dec.: {}
""".format(vv_product,
           vm_product,
           mm_product,
           sv_dec,
           lu_dec,
           qr_dec,
           cholesky_dec,
           eig_dec))

print("""
NumPy Configuration:
--------------------
""")
np.__config__.show()