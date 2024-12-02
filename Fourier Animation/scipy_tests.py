import scipy as sp
import numpy as np
import matplotlib.pyplot as plt


@np.vectorize
def A_term(k, carrier_k=1, std=1):
	k = k - carrier_k
	a = 1 / std**2
	return np.sqrt(np.pi / a) * np.e**(-(np.pi * k)**2 / a)


@np.vectorize
def A_term(k):
	return np.sinc(carrier_k)


f = lambda k, x: A_term(k) * np.cos(k * x)

int_limit = 100
subdivisions = 100
result = lambda x: sp.integrate.quad(f, -int_limit, int_limit,
									  args=(x, ), limit=subdivisions)

@np.vectorize
def vector_result(x):
	return result(x)

k = np.linspace(-10, 10, 500)
x = np.linspace(-20, 20, 500)
# approximate result, approximate error
y = vector_result(x)
fig, ax = plt.subplots()

# ax.plot(k, A_term(k))
ax.plot(x, y[0])
#ax.plot(ts, y[1])

plt.show()
