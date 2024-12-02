import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as FA
from matplotlib.animation import FFMpegWriter
import numpy as np
from scipy import stats
from time import time

# rcParams
mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Program Files\\ffmpeg-7.1-full_build\\bin\\ffmpeg.exe'
rc_update = {'font.size': 18, 'font.family': 'serif',
			 'font.serif': ['Times New Roman', 'FreeSerif'], 'mathtext.fontset': 'cm'}
plt.rcParams.update(rc_update)


class FourierFunction(object):

	def __init__(self, name, func, wl, A_term=lambda m: 0, B_term=lambda m: 0, w=lambda k: 0):
		'''
		Parameters
		----------
		func: function
			Function to be Fourier expanded.
		wl: float
			Wavelength.
		A_term: function
			Returns mth value of cosine coefficient (may take negative m).
		B_term: function
			Returns mth value of sine coefficient (may take negative m).
		w: function
			Function of k, dispersion relation. Defaults to 0.
		self.neg: Bool
			Whether or not to expand with negative m.
		self.M: int
			Nonnegative, maximum Fourier term at which f has been expanded.
		'''
		self.name = name
		self.func = func
		self.wl = wl
		self.w = w
		self.k0 = 2 * np.pi / wl
		print(self.k0)

		self.A_term = A_term
		self.B_term = B_term
		self.phase = lambda m, x, t: m * self.k0 * x - w(m * self.k0) * t
		self.A_cos = lambda m, x, t: A_term(m) * np.cos(self.phase(m, x, t))
		self.B_sin = lambda m, x, t: B_term(m) * np.sin(self.phase(m, x, t))

		self.M = 10
		self.neg = True
		self.fourier_expand(self.M)


	def fourier_expand(self, M, negatives=True):
		self.neg = negatives
		self.M = M
		self.fseries = lambda x, t: sum([self.A_cos(m, x, t) + self.B_sin(m, x, t)
									  	 for m in range(-int(self.neg) * M, M + 1)])


	def graph(self, fig, axs, x_plt_range=None,
			  k_space=False, t=0, plot_original=False, show=True):
		if not x_plt_range:
			x_plt_range = (-self.wl, self.wl)
		expansion_range = (-int(self.neg) * self.M, self.M + 1)

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
		axs['x-space'].plot(x, self.fseries(x, t),
							label=f'Fourier up to term {self.M}', color='#18CCF5')

		# Sinusoidal constituents
		for m in range(*expansion_range):
			axs['x-space'].plot(x, self.A_cos(m, x, t), alpha=0.2)
			axs['x-space'].plot(x, self.B_sin(m, x, t), alpha=0.2)

		# k-space
		if not k_space:
			if show:
				FourierFunction.show_plot()
			return fig, axs

		integers = [m for m in range(*expansion_range)]
		A_terms = [self.A_term(m) for m in range(*expansion_range)]
		B_terms = [self.B_term(m) for m in range(*expansion_range)]
		x_axis = ((1 + int(self.neg)) * self.M + 1) * [0]
		
		axs['k-space'].set(xlim=expansion_range)
		axs['k-space'].set_xticks(integers)
		axs['k-space'].set(title='$k$-space', xlabel='$mk_0$', ylabel='$A_m$/$B_m$')
		axs['k-space'].axhline(0, color='#ccc', ls='--')
		axs['k-space'].scatter(integers, A_terms, label='$A_m$: Cosine terms',
							   marker='*', color='#FA3719', zorder = 2)
		axs['k-space'].scatter(integers, B_terms, label='$B_m$: Sine terms', color='#18CCF5')
		axs['k-space'].vlines(integers, x_axis, A_terms, color='#FAB9AF', zorder=0)
		axs['k-space'].vlines(integers, x_axis, B_terms, color='#ABE5F5', zorder=0)

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
		a = 1 / std**2
		return np.e**(-a * x**2)


	def p_gaussian(p, carrier_p=0, x_std=1):
		a = 1 / std**2
		return np.sqrt(np.pi / a) * np.e**(-(np.pi * p)**2 / a)


	def even_terms(m, carrier_p=0, x_std=1):
		'''mth cosine term. DC term included.'''
		return 1E10 * p_gaussian(m, carrier_p, x_std)


	def B_term(m):
		'''mth sine term.'''
		return 0

	w0 = 30
	def w(k):
		return abs(k) * w0


	# Pulse parameters
	wl = 10
	p_i = 10
	std = 2

	f = lambda x: x_gaussian(x, std=std)
	A_term = lambda m: even_terms(m, 3, 3)

	
	# Plot parameters
	subplot_mosaic = [['x-space'], ['k-space']]
	fig, axs = plt.subplot_mosaic(subplot_mosaic, dpi=300)
	x_plt_range = (-100, 100)

	# FourierFunction
	gaussian_pulse = FourierFunction("Gaussian pulse", f, wl, A_term, B_term, w)
	gaussian_pulse.graph(fig, axs, x_plt_range=x_plt_range, plot_original=True, show=True, k_space=True)
	
	k0 = 0.6283185307179586
	print(even_terms(k0, 3, 3))
	#gaussian_pulse.fourier_expand(50)
	

if __name__ == '__main__':
	main()