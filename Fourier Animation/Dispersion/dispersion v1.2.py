import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as FA
from matplotlib.animation import FFMpegWriter
import numpy as np
import scipy as sp
from scipy import stats
from time import time
#from playsound import playsound

# rcParams
mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Program Files\\ffmpeg-7.1-full_build\\bin\\ffmpeg.exe'
#mpl.rcParams['animation.ffmpeg_path'] = r'E:\Program Files\ffmpeg-7.1-full_build\bin\ffmpeg.exe'
rc_update = {'font.size': 7, 'font.family': 'serif',
			 'font.serif': ['Times New Roman', 'FreeSerif'], 'mathtext.fontset': 'cm'}
plt.rcParams.update(rc_update)

og_colors = {'blue': '#18CCF5', 'red': '#FA3719'}
colors = {'orange': '#ffab40', 'Lgrey': '#999', 'blue': '#7ea8a7',
		  'red': '#e58e8fff', 'yellow': '#fcf3cc', 'dark_orange': '#bf6f22'}

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
			integral = lambda x, t: sp.integrate.quad(
				integrand, expansion_range[0], expansion_range[1], args=(x, t))
			return integral(x, t)

		return FIntegral(x, t)[0] / (2 * np.pi)


	def graph_relevant(self, axs, k):
		relevant_A_k = []
		relevant_B_k = []
		for i in k:
			if self.A_terms(i) > max(self.A_terms(k)) / (2 * np.e):
				relevant_A_k.append(i)
			if self.A_terms(i) > max(max(self.A_terms(k)) / (2 * np.e), 0):
				relevant_B_k.append(i)

		relevant_A_k = np.array(relevant_A_k)
		relevant_B_k = np.array(relevant_B_k)
		axs['k-space'].plot(relevant_A_k, self.A_terms(relevant_A_k),
							color=og_colors['red'], label='Relevant $A(k)$')
		# axs['k-space'].plot(relevant_B_k, self.B_terms(relevant_A_k),
		#					  color=og_colors['blue'], label='Relevant $B(k)$')

		k_limits = relevant_A_k if len(relevant_A_k) > len(relevant_B_k) else relevant_B_k
		relevant_k = np.linspace(min(k_limits), max(k_limits), 500)

		axs['wk-relation'].plot(relevant_k, self.w(relevant_k),
								color=colors['orange'], label='Relevant $k$', zorder=1)
		


	def graph(self, fig, axs, x_plt_range, expansion_range=(-10, 10),
			  k_space=False, t=0, plot_original=False, show=False):

		# Plots
		fig.suptitle(f"Fourier integral of {self.name} "
					 f"on range $k \in [{expansion_range[0]}, {expansion_range[1]}]$",
					 fontsize=10)
		plt.subplots_adjust(hspace=0.25)

		# x-space
		axs['x-space'].clear()
		x = np.linspace(x_plt_range[0], x_plt_range[1], 1000)
		padding = 3
		y_plt_range = (-max(abs(self.func(x))) - padding, max(abs(self.func(x))) + padding)
		axs['x-space'].set(xlim=x_plt_range, ylim=y_plt_range)

		# x-space
		axs['x-space'].set(title='$x$-space', xlabel='$x$', ylabel='$E(x)$')
		if plot_original and (0 <= t < 1):
			axs['x-space'].plot(x, self.func(x),
								label='Original function',
								color=colors['blue'], zorder=1, linestyle='--')
		axs['x-space'].plot(x, self.FIntegrate(x, t, expansion_range),
							label=f'Fourier integral', color=colors['orange'], zorder=0)
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
		
		k = np.linspace(expansion_range[0], expansion_range[1], 500)
		axs['k-space'].set(xlim=expansion_range)
		axs['k-space'].set(title='$k$-space', xlabel='$k$', ylabel='$A(k)$/$B(k)$')
		axs['k-space'].axhline(0, color='#ccc', ls='--')

		axs['k-space'].plot(k, self.A_terms(k), label='$A(k)$: Cosine terms',
							   color=colors['red'], zorder = 2)
		#axs['k-space'].plot(k, self.B_terms(k), label='$B(k)$: Sine terms',
		#					   linestyle='--', color=og_colors['blue'])
		#axs['k-space'].vlines(integers, x_axis, A_terms, color='#FAB9AF', zorder=0)
		#axs['k-space'].vlines(integers, x_axis, B_terms, color='#ABE5F5', zorder=0)
		self.graph_relevant(axs, k)

		axs['wk-relation'].set(xlim=(0, expansion_range[1]))
		axs['wk-relation'].set(title='$\omega k $-relation', xlabel='$k$', ylabel='$\omega(k)$')

		axs['wk-relation'].plot(k, self.w(k), label='$\omega(k)$',
							   color=colors['blue'], zorder=0)
		
		'''
		for ax_label in axs:
			axs[ax_label].legend(fontsize=7)
		'''

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


	def animate(self, filename, x_plt_range, expansion_range, total_frames,
				k_space=False, fps=10, slowdown=20):
		subplot_mosaic = [['x-space', 'x-space']]
		if k_space:
			subplot_mosaic.append(['k-space', 'wk-relation'])
		fig, axs = plt.subplot_mosaic(subplot_mosaic, dpi=300)

		def updater(frame):
			t = frame / (slowdown * fps)
			self.graph(fig, axs, x_plt_range, expansion_range,
					   k_space, t=t)
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
	# Pulse parameters
	p_i = 800
	m_i = 400
	std = 0.0025
	A = 20


	@np.vectorize
	def x_gaussian(x, mean=0, std=1):
		normal = A / (std * np.sqrt(2 * np.pi))
		return normal * np.e**(-(x / std)**2 / 2) * np.cos(p_i * x)


	@np.vectorize
	def p_gaussian(p, carrier_p=0, x_std=1):
		p = p - carrier_p
		return A * np.e**(-(x_std * p)**2 / 2)


	def even_terms(p, carrier_p=0, x_std=1):
		'''Fourier transform for even terms.'''
		return p_gaussian(p, carrier_p, x_std)


	@np.vectorize
	def B_terms(m):
		'''Fourier transform for odd terms.'''
		return 0


	@np.vectorize
	def w(k):
		return np.sqrt(k**2 + m_i**2)


	f = lambda x: x_gaussian(x, std=std)
	A_terms = lambda k: even_terms(k, p_i, std)

	
	# Plot parameters
	subplot_mosaic = [['x-space', 'x-space'], ['k-space', 'wk-relation']]
	# subplot_mosaic = [['x-space']]
	fig, axs = plt.subplot_mosaic(subplot_mosaic, dpi=300)
	x_plt_range = (-10 * std, 30 * std)
	rang = 1000
	expansion_range = (-rang + p_i, rang + p_i)

	# FourierFunction
	gaussian_pulse = FourierFunction("gaussian pulse", f, A_terms, B_terms, w)
	#gaussian_pulse.graph(fig, axs, x_plt_range, expansion_range=expansion_range,
	#					 plot_original=True, show=True, k_space=True)
	
	# Need to add way so that old files can't be overwriten.
	filename = '../mp4/gaussian_pulse_high_dispersion3.mp4'
	gaussian_pulse.animate(filename, x_plt_range, expansion_range,
						   fps=10, slowdown=100, total_frames=100,
						   k_space=True)	

	done_sound = '../GW150914_H1_shifted.wav'
	#playsound(done_sound)


if __name__ == '__main__':
	main()
