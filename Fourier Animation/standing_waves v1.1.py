import matplotlib.pyplot as plt
import numpy as np


class FourierFunction(object):

	def __init__(self, name, f, wl, A_term=lambda m: 0, B_term=lambda m: 0):
		self.name = name
		self.func = f
		self.wl = wl
		self.k_0 = 2 * np.pi / wl

		self.A_term = A_term
		self.B_term = B_term
		self.A_cos = lambda m, x: A_term(m) * np.cos(m * self.k_0 * x)
		self.B_sin = lambda m, x: B_term(m) * np.sin(m *self.k_0 * x)

		self.M = 10
		self.fourier_expand(self.M)


	def fourier_expand(self, M):
		self.M = M
		self.fseries = lambda x: sum([self.A_cos(m, x) + self.B_sin(m, x) for m in range(M + 1)])


	def graph(self, plt_range=None, k_space=True):
		if not plt_range:
			plt_range = (0, self.wl)

		# rcParams
		rc_update = {'font.size': 18, 'font.family': 'serif',
					 'font.serif': ['Times New Roman', 'FreeSerif'], 'mathtext.fontset': 'cm'}
		plt.rcParams.update(rc_update)

		# Plots
		fig, axs = plt.subplot_mosaic([['x-space'], ['k-space']])
		fig.suptitle(f"Fourier series of {self.name} up to term {self.M}")
		plt.subplots_adjust(hspace=0.5)

		# x-space
		x = np.linspace(plt_range[0], plt_range[1], 1000)
		axs['x-space'].set(title='$x$-space', xlabel='$x$', ylabel='$E(x)$')
		axs['x-space'].plot(x, self.func(x), label='Original function', color='#FA3719')
		axs['x-space'].plot(x, self.fseries(x), label=f'Fourier up to term {self.M}', color='#18CCF5')

		for m in range(self.M + 1):
			axs['x-space'].plot(x, self.A_cos(m, x), alpha=0.2)
			axs['x-space'].plot(x, self.B_sin(m, x), alpha=0.2)

		# k-space
		if k_space != True:
			return fig, axs

		integers = [m for m in range(self.M + 1)]
		A_terms = [self.A_term(m) for m in range(self.M + 1)]
		B_terms = [self.B_term(m) for m in range(self.M + 1)]
		x_axis = (self.M + 1) * [0]
		
		axs['k-space'].set_xticks(integers)
		axs['k-space'].set(title='$k$-space', xlabel='$m$', ylabel='$A_m$/$B_m$')
		axs['k-space'].scatter(integers, A_terms, marker='*', label='$A_m$: Cosine terms', color='#FA3719')
		axs['k-space'].scatter(integers, B_terms, label='$B_m$: Sine terms', color='#18CCF5')
		axs['k-space'].vlines(integers, x_axis, A_terms, color='#FAB9AF', zorder=0)
		axs['k-space'].vlines(integers, x_axis, B_terms, color='#ABE5F5', zorder=0)

		for ax_name in axs:
			axs[ax_name].legend(fontsize=10)

		figManager = plt.get_current_fig_manager()
		figManager.full_screen_toggle()

		return fig, axs


A = 5
wl = 10

@np.vectorize
def f(x):
	return A + A * int((x % wl) < wl / 2) + -A * int((x % wl) > wl / 2)


def A_term(m):
	'''mth cosine term. DC term included.'''
	if m == 0:
		return A
	return 0


def B_term(m):
	'''mth sine term.'''
	if m == 0:
		return 0
	if m % 2 == 1:
		return 4 * A / (m * np.pi)
	return 0


f = FourierFunction("square wave", f, wl, A_term, B_term)
f.fourier_expand(50)
f.graph(plt_range=(wl / 4 - wl / 2.5, wl / 4 + wl / 2.5))
plt.show()