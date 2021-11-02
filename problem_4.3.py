"""
Code that solves HW #3, problem 4.3 from the text.
"""

import numpy as np
from matplotlib import pyplot as plt

from matplotlib.cm import jet

#from descent_algorithm_utils import good_init, good_descent, good_stepsize, good_termination
#from descent_algorithm_utils import  bad_init,  bad_descent,  bad_stepsize,  bad_termination

def exact_linesearch_for_quadratics(d, xy0, gfxy, f, A):
    # Exact linesearch for functions of the form
    # f(x) = x^T A x + 2 b^T A x + c

    return np.inner(-d, gfxy)/(2 * np.inner(d, A @ d))

def backtracking_linesearch(d, xy0, gfxy, f, s, alpha, beta):
    # Backtracking linesearch algorithm.

    assert np.inner(d, gfxy) < 0, "Direction d is not a descent direction"

    fold = f(xy0)
    xy = xy0 + s*d
    fnew = f(xy)
    i = 0

    while (fold - fnew) < -alpha*(s*beta**i)*(np.inner(gfxy, d)):
        i += 1
        xy = xy0 + (s*beta**i)*d
        fnew = f(xy)

    return s*(beta**i)

def gradient_descent(f, gradf, x0, stepsize, gradnorm_tolerance, max_iter=5000):
    """
    Generic gradient descent algorithm.
    """

    xy_k = np.zeros([max_iter+1, x0.size])  # iterates
    f_k  = np.zeros(max_iter+1)             # function values
    t_k  = np.zeros(max_iter+1)             # stepsizes
    gf_k = np.zeros(max_iter+1)             # gradient norms

    k = 0 # Iteration count

    # Initialize
    xy_k[0,:] = x0
    f_k[0] = f(xy_k[0,:])
    gf = gradf(xy_k[0,:])
    gf_k[0] = np.linalg.norm(gf)

    while (k < max_iter) and (np.linalg.norm(gf) > gradnorm_tolerance):

        # Descent direction: gradient descent
        d_k = -gf

        # Stepsize
        t_k[k] = stepsize(d_k, xy_k[k,:], gf, f)

        # Update
        xy_k[k+1,:] = xy_k[k,:] + t_k[k]*d_k

        # Compute metrics for iterate k+1
        f_k[k+1] = f(xy_k[k+1,:])
        gf = gradf(xy_k[k+1,:])
        gf_k[k+1] = np.linalg.norm(gf)

        k += 1

    # Truncate unused allocation
    xy_k = xy_k[:k+1,:]
    f_k = f_k[:k+1]
    gf_k = gf_k[:k+1]
    t_k = t_k[:k]

    return xy_k, f_k, gf_k, t_k

# Returns an (n x n) Hilbert matrix
def hilb(n):
    assert n > 0
    A = np.zeros((n,n))
    # i
    A += np.tile(np.reshape(np.arange(1,n+1), (n,1)), (1, n))
    # j
    A += A.T

    return 1/(A-1)

# The function and its gradient
def f_template(A,x):
    if len(x.shape)==1:
        return np.inner(x, A @ x)
    else:
        return np.inner(x, A @ x)

def gradf_template(A,x):
    return 2*(A@x)

# Set up problem
n = 5   # Defines the size of A
gradnorm_tolerance = 1e-4

x0 = np.array([1., 2., 3., 4., 5.])

A = hilb(n)

f = lambda x: f_template(A,x)
gradf = lambda x: gradf_template(A,x)

# Backtracking linesearch parameters
s = 1
alpha = 0.5
beta = 0.5

solutions = [None]*3
titles = [None]*3

stepsize = lambda d, xy0, gfxy, f: backtracking_linesearch(d, xy0, gfxy, f, s, alpha, beta)
solutions[0] = gradient_descent(f, gradf, x0, stepsize, gradnorm_tolerance)
titles[0] = "Backtracking, $s = 1$, $\\alpha = 0.5$, $\\beta= 0.5$"

#f2 = lambda x: x[0]**2 + 2*x[1]**2
#B = np.eye(2)
#B[0,0] = 1
#B[1,1] = 2
#def grad2(x):
#    return 2*B @ x
#
#x02 = np.array([2., 1.])
#stepsize = lambda d, xy0, gfxy, f: exact_linesearch_for_quadratics(d, xy0, gfxy, f, B)
#testing = gradient_descent(f2, grad2, x02, stepsize, 1e-5)

alpha = 0.1
stepsize = lambda d, xy0, gfxy, f: backtracking_linesearch(d, xy0, gfxy, f, s, alpha, beta)
solutions[1] = gradient_descent(f, gradf, x0, stepsize, gradnorm_tolerance)
titles[1] = "Backtracking, $s = 1$, $\\alpha = 0.1$, $\\beta= 0.5$"

stepsize = lambda d, xy0, gfxy, f: exact_linesearch_for_quadratics(d, xy0, gfxy, f, A)
solutions[2] = gradient_descent(f, gradf, x0, stepsize, gradnorm_tolerance)
titles[2] = "Exact linesearch"

## Metrics
fig = plt.figure(figsize=(12, 6))
for col in range(3):

    k = solutions[col][1].size-1

    #plt.subplot(2,3,col+1)
    #plt.plot(range(k), solutions[col][3], 'k.-')
    #plt.xlabel('Iteration index $k$')
    #plt.title('Step size $t_k$')

    plt.subplot(2,3,col+1)
    plt.semilogy(range(k+1), solutions[col][1], 'k.-')
    plt.xlabel('Iteration index $k$')
    plt.title(titles[col]+'\n{0:d} iterations'.format(k) + '\nFunction value $f(x_k)$')

    plt.subplot(2,3,col+4)
    plt.semilogy(range(k+1), solutions[col][2], 'k.-')
    plt.xlabel('Iteration index $k$')
    plt.title('Gradient norm $||\\nabla f(x_k)||$')

plt.subplots_adjust(wspace=0.5, hspace=0.5)

plt.show()

