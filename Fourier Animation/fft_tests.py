import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

p_i = 25
m_i = 15
std = 0.1
A = 20

@np.vectorize
def x_gaussian(x, mean=0, std=1):
	normal = A / (std * np.sqrt(2 * np.pi))
	return normal * np.e**(-(x / std)**2 / 2) * np.cos(p_i * x)

@np.vectorize
def p_gaussian(p, carrier_p=0, x_std=1):
	p = p - carrier_p
	B = A / np.sqrt(2 * np.pi)
	return B * np.e**(-(x_std * p)**2 / 2)

def even_terms(p, carrier_p=0, x_std=1):
	'''Fourier transform for even terms.'''
	return p_gaussian(p, carrier_p, x_std)


f = lambda x: x_gaussian(x, std=std)
A_terms = lambda k: even_terms(k, p_i, std)

# Space variables
rang = 25
x = np.linspace(-30 * std, 30 * std, 1000)
k = np.linspace(0, rang + p_i, 500)

fig, ax = plt.subplots()
ax.plot(x, f(x), label='Function')
ax.plot(k, A_terms(k), label='Transform')
ax.plot(x, np.fft.fft(f(x)).real)
plt.legend()
plt.show()
