import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as FA
from matplotlib.animation import FFMpegWriter
import numpy as np
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

		self.A_term = A_term
		self.B_term = B_term
		self.phase = lambda m, x, t: m * self.k0 * x - w(m * self.k0) * t
		self.A_cos = lambda m, x, t: A_term(m) * np.cos(self.phase(m, x, t))
		self.B_sin = lambda m, x, t: B_term(m) * np.sin(self.phase(m, x, t))

		self.M = 10
		self.neg = True
		self.fourier_expand(self.M)

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


	def fourier_expand(self, M, negatives=True):
		self.neg = negatives
		self.M = M
		self.fseries = lambda x, t: sum([self.A_cos(m, x, t) + self.B_sin(m, x, t)
									  	 for m in range(-int(self.neg) * M, M + 1)])


	def graph(self, fig, axs, x_plt_range=None, k_space=False, t=0):
		if not x_plt_range:
			x_plt_range = (-self.wl, self.wl)
		expansion_range = (-int(self.neg) * self.M, self.M + 1)

		# Plots
		axs['x-space'].clear()
		x = np.linspace(x_plt_range[0], x_plt_range[1], 1000)
		padding = 0.5
		y_plt_range = (min(self.func(x)) - padding, max(self.func(x)) + padding)
		axs['x-space'].set(xlim=x_plt_range, ylim=y_plt_range)

		# x-space
		axs['x-space'].set(title='$x$-space', xlabel='$x$', ylabel='$E(x)$')
		axs['x-space'].plot(x, self.func(x), label='Original function', color='#FA3719')
		axs['x-space'].plot(x, self.fseries(x, t), label=f'Fourier up to term {self.M}', color='#18CCF5')

		for m in range(*expansion_range):
			axs['x-space'].plot(x, self.A_cos(m, x, t), alpha=0.2)
			axs['x-space'].plot(x, self.B_sin(m, x, t), alpha=0.2)

		# k-space
		if not k_space:
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

		figManager = plt.get_current_fig_manager()
		figManager.full_screen_toggle()
		plt.tight_layout()

		return fig, axs


	def crude_animator(self, x_plt_range):
		for t in np.arange(0, 2*np.pi, 0.1):
			self.core_grapher(x_plt_range, t=t)
			plt.show()


	def animate(self, filename, slowdown=20, x_plt_range=None, k_space=False, total_frames=None):
		if not total_frames:
			total_frames = int(np.ceil(slowdown * 20 * np.pi / self.w(1)))
		subplot_mosaic = [['x-space']]
		if k_space:
			subplot_mosaic.append(['kspace'])
		fig, axs = plt.subplot_mosaic(subplot_mosaic, dpi=300)
		fig.suptitle(f"Fourier series of {self.name} up to term {self.M}")
		plt.subplots_adjust(hspace=0.5)

		def updater(frame):
			t = frame / (slowdown * 10)
			self.graph(fig, axs, x_plt_range, k_space, t=t)
			self.animation_progress(total_frames, frame)

		animation = FA(plt.gcf(), updater, frames=list(range(total_frames)), repeat=False)
		writer = FFMpegWriter(fps=10)
		animation.save(filename, writer=writer)


def main():
	A = 5
	wl = 10
	w0 = 5

	@np.vectorize
	def f(x):
		return A * int((x % wl) < wl / 2) + -A * int((x % wl) > wl / 2)


	def A_term(m):
		'''mth cosine term. DC term included.'''
		return 0


	def B_term(m):
		'''mth sine term.'''
		if m == 0:
			return 0
		if m % 2 == 1:
			return 2 * A / (m * np.pi)
		return 0


	def w(k):
		return abs(k) * w0


	filename = "./squre_wave.mp4"
	x_plt_range = (-2.5, 7.5)
	f = FourierFunction("square wave", f, wl, A_term, B_term, w)
	f.fourier_expand(50)
	f.animate(filename, x_plt_range=x_plt_range)


if __name__ == '__main__':
	main()