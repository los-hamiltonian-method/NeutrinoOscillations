import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as FA
from matplotlib.animation import FFMpegWriter
import numpy as np
import scipy as sp
from scipy import stats
from time import time

# rcParams
mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Program Files\\ffmpeg-7.1-full_build\\bin\\ffmpeg.exe'
rc_update = {'font.size': 7, 'font.family': 'serif',
			 'font.serif': ['Times New Roman', 'FreeSerif'], 'mathtext.fontset': 'cm'}
plt.rcParams.update(rc_update)


class FourierFunction(object):

	def __init__(self, name, func, A_terms=lambda k: 0, B_terms=lambda k: 0, w=lambda k: 0):
		'''
		Parameters
		----------
		func: function
			Function to be Fourier expanded.
		A_terms: function
			Fourier transform for cosine terms.
		B_terms: function
			Fourier transform for sine terms.
		w: function
			Function of k, dispersion relation. Defaults to 0.
		'''
		self.name = name
		self.func = func
		self.w = w

		self.A_terms = A_terms
		self.B_terms = B_terms
		self.phase = lambda k, x, t: k * x - w(k) * t
		self.A_cos = lambda k, x, t: A_terms(k) * np.cos(self.phase(k, x, t))
		self.B_sin = lambda k, x, t: B_terms(k) * np.sin(self.phase(k, x, t))


	def FIntegrate(self, x, t=0, expansion_range=(-10, 10)):
		integrand = lambda k, x, t: self.A_cos(k, x, t) + self.B_sin(k, x, t)

		@np.vectorize
		def FIntegral(x, t):
			integral = lambda x, t: sp.integrate.quad(integrand, expansion_range[0],
													  expansion_range[1], args=(x, t))
			return integral(x, t)

		return FIntegral(x, t)[0]


	def graph(self, fig, axs, x_plt_range, expansion_range=(-10, 10),
			  k_space=False, t=0, plot_original=False, show=True):

		# Plots
		# x-space
		axs['x-space'].clear()
		x = np.linspace(x_plt_range[0], x_plt_range[1], 1000)
		padding = 0.5
		y_plt_range = (min(self.func(x)) - padding, max(self.func(x)) + padding)
		axs['x-space'].set(xlim=x_plt_range, ylim=y_plt_range)

		# x-space
		axs['x-space'].set(title='$x$-space', xlabel='$x$', ylabel='$E(x)$')
		if plot_original:
			axs['x-space'].plot(x, self.func(x),
								label='Original function', color='#FA3719')
		axs['x-space'].plot(x, self.FIntegrate(x, t, expansion_range),
							label=f'Fourier integral', color='#18CCF5')
		axs['x-space'].legend()

		# Sinusoidal constituents
		# for m in range(*expansion_range):
			# axs['x-space'].plot(x, self.A_cos(m, x, t), alpha=0.2)
			# axs['x-space'].plot(x, self.B_sin(m, x, t), alpha=0.2)

		# k-space
		if not k_space:
			if show:
				FourierFunction.show_plot()
			return fig, axs
		
		k = np.linspace(expansion_range[0], expansion_range[1], 1000)
		axs['k-space'].set(xlim=expansion_range)
		axs['k-space'].set(title='$k$-space', xlabel='$k$', ylabel='$A(k)$/$B(k)$')
		axs['k-space'].axhline(0, color='#ccc', ls='--')

		axs['k-space'].plot(k, self.A_terms(k), label='$A(k)$: Cosine terms',
							   linestyle='--', color='#FA3719', zorder = 2)
		axs['k-space'].plot(k, self.B_terms(k), label='$B(k)$: Sine terms', color='#18CCF5')
		#axs['k-space'].vlines(integers, x_axis, A_terms, color='#FAB9AF', zorder=0)
		#axs['k-space'].vlines(integers, x_axis, B_terms, color='#ABE5F5', zorder=0)
		plt.legend()

		if show:
			FourierFunction.show_plot()

		return fig, axs


	@staticmethod
	def show_plot():
		figManager = plt.get_current_fig_manager()
		figManager.full_screen_toggle()
		plt.tight_layout()
		plt.show()


	def crude_animator(self, x_plt_range):
		for t in np.arange(0, 2*np.pi, 0.1):
			self.core_grapher(x_plt_range, t=t)
			plt.show()


	def animate(self, filename, fps=10, slowdown=20, x_plt_range=None, k_space=False, total_frames=None):
		if not total_frames:
			total_frames = int(np.ceil(slowdown * 20 * np.pi / self.w(1)))
		subplot_mosaic = [['x-space']]
		if k_space:
			subplot_mosaic.append(['kspace'])
		fig, axs = plt.subplot_mosaic(subplot_mosaic, dpi=300)
		fig.suptitle(f"Fourier series of {self.name} up to term {self.M}")
		plt.subplots_adjust(hspace=0.5)

		fps = 10
		def updater(frame):
			t = frame / (slowdown * n_fps)
			self.graph(fig, axs, x_plt_range, k_space, t=t)
			self.animation_progress(total_frames, frame)

		animation = FA(plt.gcf(), updater, frames=list(range(total_frames)), repeat=False)
		writer = FFMpegWriter(fps=fps)
		animation.save(filename, writer=writer)


	@staticmethod
	def animation_progress(total_frames, frame):
		# Progress bar
		frame += 1
		done = frame / total_frames
		remaining = (total_frames - frame) / total_frames
		done = int(np.ceil(20 * done)) * '#'
		remaining = int(np.ceil(20 * remaining)) * '_'
		progress_bar = "[" + done + remaining + "]"

		# Print
		print(f"Animated: {frame} / {total_frames} frames")
		print(progress_bar + '\n')

		if frame == total_frames:
			print("Animation done!\n")


def main():
	# F{e**(-a * x**2)} = np.sqrt(np.pi / a) e**((np.pi * k)**2 / a)
	@np.vectorize
	def x_gaussian(x, mean=0, std=1):
		a = 1 / (2 * std**2)
		return np.e**(-a * x**2)


	@np.vectorize
	def p_gaussian(p, carrier_p=0, x_std=1):
		a = 1 / (2 * std**2)
		p = p - carrier_p
		return np.sqrt(np.pi / a) * np.e**(-(np.pi * p)**2 / a)


	def even_terms(p, carrier_p=0, x_std=1):
		'''Fourier transform for even terms.'''
		return p_gaussian(p, carrier_p, x_std)


	@np.vectorize
	def B_terms(m):
		'''Fourier transform for odd terms.'''
		return 0


	w0 = 30
	def w(k):
		return abs(k) * w0


	# Pulse parameters
	p_i = 1
	std = 1

	f = lambda x: x_gaussian(x, std=std)
	A_terms = lambda k: even_terms(k, p_i, std)

	
	# Plot parameters
	subplot_mosaic = [['x-space', 'k-space']]
	# subplot_mosaic = [['x-space']]
	fig, axs = plt.subplot_mosaic(subplot_mosaic, dpi=300)
	x_plt_range = (-30, 30)
	expansion_range = (-10 + p_i, 10 + p_i)

	# FourierFunction
	gaussian_pulse = FourierFunction("Gaussian pulse", f, A_terms, B_terms, w)
	gaussian_pulse.graph(fig, axs, x_plt_range, expansion_range=expansion_range,
						 plot_original=True, show=True, k_space=True)
	

if __name__ == '__main__':
	main()
