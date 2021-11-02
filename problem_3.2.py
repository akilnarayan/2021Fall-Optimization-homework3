# Solves Problem 3.2 from the book.

import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(0, 1, 30)
y = 2*x**2 - 3*x + 1 + 0.05*np.random.randn(*x.shape)

A = np.zeros((x.size, 3))
for i in range(A.shape[1]):
    A[:,i] = x**i

# Solve the least squares problem using built-in solvers instead of explicitly
# solving the normal equations
coeffs = np.linalg.lstsq(A, y)[0]

print("The least squares fitted polynomial has the form {0:1.4f} x^2 + {1:1.4f} x + {2:1.4f}".format(coeffs[2], coeffs[1], coeffs[0]))

plt.plot(x, y, 'r.', x, A @ coeffs, 'b')
plt.xlim([0, 1])
plt.ylim([-0.4, 1])
plt.show()
