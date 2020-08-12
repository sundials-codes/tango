import numpy as np
import matplotlib.pyplot as plt
import os

f1 = 'a1'
f2 = 'a2'

os.system('python3 kinsol_simple.py alpha 0.4 depth 0 save ' + f1)
os.system('python3 kinsol_simple.py alpha 0.3 depth 1 save ' + f2)

resid1 = np.loadtxt(f1 + 'resid')
resid2 = np.loadtxt(f2 + 'resid')

it1 = range(0,len(resid1))
it2 = range(0,len(resid2))

# plot residuals
plt.figure()
plt.semilogy(it1, resid1, it2, resid2)
plt.xlabel('iteration number')
plt.title('Max-norm residual')
plt.savefig('overlay.png')
