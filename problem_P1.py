
import numpy as np
from matplotlib import pyplot as plt

mu, sig = 3, 0.4

M = 100
ym = mu + sig*np.random.randn(M)

# Maximum likelihood estimate
mu_ast = 1/M * np.sum(ym)
sig_ast = np.sqrt(1/M * np.sum((ym - mu_ast)**2))

y = np.linspace(mu -3*sig, mu + 3*sig, 100)
pdf_ast = 1/(sig_ast * np.sqrt(2*np.pi)) * np.exp(-(y - mu_ast)**2/(2*sig_ast**2))

plt.figure()
plt.hist(ym, bins=20, density=True, label='Histogram data')
plt.plot(y, pdf_ast, 'r', label='Max likelihood estimate')
plt.xlabel('$y$')
plt.legend(frameon=False)
plt.show()
